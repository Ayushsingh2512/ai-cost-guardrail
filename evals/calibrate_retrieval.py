"""
Retrieval score calibration tool (dev split only).

Analyzes top-1 retrieval similarity score distributions for answerable vs
unanswerable questions from an evaluation report (e.g. generated with
--retrieval-only on the dev split) and computes the separation margin.

Rule:
    Calibration is strictly restricted to the 'dev' split. The 'test' split
    remains untouched until final configuration freeze.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _safe_percentile(values: Sequence[float], pct: float) -> float | None:
    """Return the *pct* percentile (0–100) of sorted *values*, or None if empty."""
    if not values:
        return None
    sorted_vals = sorted(values)
    idx = (pct / 100) * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_vals):
        return float(sorted_vals[-1])
    frac = idx - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def compute_distribution_stats(scores: list[float]) -> dict[str, Any]:
    """Compute summary statistics for a list of similarity scores."""
    if not scores:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "std": None,
            "scores": [],
        }

    sorted_scores = sorted(scores)
    mean_val = statistics.mean(sorted_scores)
    median_val = statistics.median(sorted_scores)
    std_val = statistics.stdev(sorted_scores) if len(sorted_scores) > 1 else 0.0

    return {
        "count": len(sorted_scores),
        "min": round(sorted_scores[0], 6),
        "max": round(sorted_scores[-1], 6),
        "mean": round(mean_val, 6),
        "median": round(median_val, 6),
        "p25": round(_safe_percentile(sorted_scores, 25) or 0.0, 6),
        "p75": round(_safe_percentile(sorted_scores, 75) or 0.0, 6),
        "std": round(std_val, 6),
        "scores": [round(s, 6) for s in sorted_scores],
    }


def analyze_retrieval_calibration(report_data: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze top-1 similarity score distributions for answerable vs unanswerable questions.

    Enforces that only the 'dev' split is analyzed.

    Parameters
    ----------
    report_data:
        Parsed evaluation report JSON dictionary.

    Returns
    -------
    dict
        Distribution statistics and separation metrics.
    """
    questions: list[dict[str, Any]] = report_data.get("per_question", [])
    if not questions:
        raise ValueError("Report contains no per-question evaluation results.")

    # Rule enforcement: Dev split only
    for q in questions:
        split = q.get("split")
        if split != "dev":
            raise ValueError(
                f"Rule violation: calibration is restricted to the 'dev' split only. "
                f"Question {q.get('id')!r} belongs to split {split!r}."
            )

    answerable_top1: list[float] = []
    unanswerable_top1: list[float] = []

    for q in questions:
        if q.get("failed"):
            continue

        retrieved = q.get("retrieved", [])
        if not retrieved:
            continue

        # Top-1 score is the similarity score of the first retrieved passage
        top1_score = float(retrieved[0]["score"])

        if q.get("answerable"):
            answerable_top1.append(top1_score)
        else:
            unanswerable_top1.append(top1_score)

    ans_stats = compute_distribution_stats(answerable_top1)
    unans_stats = compute_distribution_stats(unanswerable_top1)

    separation_margin: float | None = None
    mean_difference: float | None = None
    suggested_threshold: float | None = None
    separable = False

    if ans_stats["count"] > 0 and unans_stats["count"] > 0:
        ans_min: float = ans_stats["min"]
        unans_max: float = unans_stats["max"]
        ans_mean: float = ans_stats["mean"]
        unans_mean: float = unans_stats["mean"]

        # Separation margin: min(answerable) - max(unanswerable)
        separation_margin = round(ans_min - unans_max, 6)
        mean_difference = round(ans_mean - unans_mean, 6)
        separable = separation_margin > 0.0

        if separable:
            # Clean decision boundary midpoint
            suggested_threshold = round((ans_min + unans_max) / 2.0, 6)
        else:
            # Overlapping distributions: midpoint between medians or means
            ans_med: float = ans_stats["median"]
            unans_med: float = unans_stats["median"]
            suggested_threshold = round((ans_med + unans_med) / 2.0, 6)

    return {
        "split": "dev",
        "total_questions": len(questions),
        "evaluated_questions": ans_stats["count"] + unans_stats["count"],
        "answerable": ans_stats,
        "unanswerable": unans_stats,
        "separation": {
            "margin": separation_margin,
            "mean_difference": mean_difference,
            "separable": separable,
            "suggested_threshold": suggested_threshold,
        },
    }


def print_calibration_summary(result: dict[str, Any]) -> None:
    """Print formatted distribution and separation report to stdout."""
    ans = result["answerable"]
    unans = result["unanswerable"]
    sep = result["separation"]

    border = "=" * 68
    print(f"\n{border}")
    print("  RETRIEVAL SCORE CALIBRATION REPORT  (dev split only)")
    print(border)
    print(f"  Total questions analyzed : {result['evaluated_questions']}")
    print(f"  Answerable questions     : {ans['count']}")
    print(f"  Unanswerable questions   : {unans['count']}")

    print("\n  [Top-1 Score Distribution: Answerable]")
    if ans["count"] > 0:
        print(f"    Min    : {ans['min']:.4f}")
        print(f"    p25    : {ans['p25']:.4f}")
        print(f"    Median : {ans['median']:.4f}")
        print(f"    Mean   : {ans['mean']:.4f} (std={ans['std']:.4f})")
        print(f"    p75    : {ans['p75']:.4f}")
        print(f"    Max    : {ans['max']:.4f}")
    else:
        print("    (no answerable questions evaluated)")

    print("\n  [Top-1 Score Distribution: Unanswerable]")
    if unans["count"] > 0:
        print(f"    Min    : {unans['min']:.4f}")
        print(f"    p25    : {unans['p25']:.4f}")
        print(f"    Median : {unans['median']:.4f}")
        print(f"    Mean   : {unans['mean']:.4f} (std={unans['std']:.4f})")
        print(f"    p75    : {unans['p75']:.4f}")
        print(f"    Max    : {unans['max']:.4f}")
    else:
        print("    (no unanswerable questions evaluated)")

    print("\n  [Separation Analysis]")
    if sep["margin"] is not None:
        margin_str = f"{sep['margin']:+.4f}"
        print(f"    Separation margin (min(ans) - max(unans)) : {margin_str}")
        print(f"    Mean difference (mean(ans) - mean(unans)): {sep['mean_difference']:+.4f}")
        print(f"    Distributions separable without overlap  : {'YES' if sep['separable'] else 'NO'}")
        if sep["suggested_threshold"] is not None:
            print(f"    Suggested abstention score threshold    : {sep['suggested_threshold']:.4f}")
    else:
        print("    Insufficient data for separation calculation.")

    print(f"{border}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evals.calibrate_retrieval",
        description="Analyze top-1 retrieval score distributions and separation margin on the dev split.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path to evaluation report JSON (from --retrieval-only on dev split).",
    )

    args = parser.parse_args(argv)

    if not args.report.exists():
        print(f"Error: report file not found: {args.report}", file=sys.stderr)
        return 1

    try:
        with args.report.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        result = analyze_retrieval_calibration(data)
    except Exception as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 1

    print_calibration_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

