"""
Report serialisation for the RAG evaluation suite (Pass 1 and Pass 1.5).

The report is written as machine-readable JSON to a configurable output path.
Generated artifacts (reports) are listed in .gitignore and never committed.

Deterministic-mode reports carry an explicit disclaimer that retrieval and
answer metrics validate only the harness logic, not real model quality.

The report schema is versioned (``schema_version``) so future passes can add
fields without breaking consumers of older reports.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from evals.datasets.corpus import (
    CORPUS_CHUNK_COUNT,
    CORPUS_DOCUMENT_COUNT,
    CORPUS_VERSION,
)
from evals.metrics import QuestionResult, SplitMetrics

SCHEMA_VERSION = "1.5.0"
DEFAULT_PROMPT_VERSION = "rag-grounded-v1"
DEFAULT_GENERATOR_MODEL = "gemini-3-flash-preview"
DEFAULT_TEMPERATURE = 0.0


def sanitize_secrets(text: str | None) -> str | None:
    """Ensure API keys and credentials are never logged or saved to reports."""
    if text is None:
        return None
    sanitized = text
    # 1. Redact values of environment variables that resemble keys/tokens
    for key_env in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "API_KEY", "SECRET_KEY"):
        val = os.environ.get(key_env)
        if val and len(val) >= 6:
            sanitized = sanitized.replace(val, "[REDACTED_API_KEY]")
    # 2. Match common Google/Gemini API key patterns
    sanitized = re.sub(r"AIza[0-9A-Za-z-_]{20,}", "[REDACTED_API_KEY]", sanitized)
    # 3. Match generic key/token patterns in URLs or headers
    sanitized = re.sub(r"(?i)(key|token|auth|bearer)\s*[:=]\s*['\"]?[^\s&'\"]+", r"\1=[REDACTED]", sanitized)
    return sanitized


def get_git_commit() -> str:
    """Return the current git commit hash, or 'unknown' if unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def compute_file_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    if not path.exists():
        return "unknown"
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_config_fingerprint(
    *,
    dataset_sha256: str,
    corpus_version: str,
    prompt_version: str,
    generator_model: str,
    temperature: float,
    top_k: int,
    mode: str,
) -> str:
    """Compute a deterministic hash representing the evaluation run configuration."""
    raw = (
        f"{dataset_sha256}|{corpus_version}|{prompt_version}|"
        f"{generator_model}|{temperature:.4f}|{top_k}|{mode}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def append_checkpoint_record(checkpoint_path: Path, record: dict[str, Any]) -> None:
    """Append a JSON record to the checkpoint JSONL file and flush immediately."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, cls=_DecimalEncoder) + "\n")
        fh.flush()


def read_checkpoint(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """
    Read an existing checkpoint file (either JSONL checkpoint or JSON report).

    Returns (header_dict_or_None, list_of_question_records).
    """
    if not path.exists():
        return None, []

    if path.suffix == ".json":
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            json_header: dict[str, Any] = {
                "config_fingerprint": data.get("config_fingerprint"),
                "dataset_sha256": data.get("dataset_sha256"),
                "corpus_version": data.get("corpus_version"),
                "prompt_version": data.get("prompt_version"),
                "generator_model": data.get("generator_model"),
                "temperature": data.get("temperature"),
                "top_k": data.get("top_k"),
                "mode": data.get("mode"),
            }
            return json_header, data.get("per_question", [])
        except Exception:
            return None, []

    # JSONL checkpoint
    header: dict[str, Any] | None = None
    questions: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("type") == "header":
                    header = record
                else:
                    questions.append(record)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Failed to parse checkpoint %s: %s", path, exc)
    return header, questions

DETERMINISTIC_DISCLAIMER = (
    "DETERMINISTIC MODE: retrieval uses FakeEmbedder + FakeCorpusRetriever "
    "(in-memory cosine similarity) and generation uses FakeGenerator (no LLM "
    "call). Recall and abstention metrics validate the evaluation harness only "
    "and say nothing about real retrieval or generation quality."
)

CORPUS_DISCLAIMER = (
    f"Results are computed over a synthetic corpus ({CORPUS_DOCUMENT_COUNT} documents, "
    f"{CORPUS_CHUNK_COUNT} chunks, version {CORPUS_VERSION}). "
    "Differences of one or two questions are within the noise of this sample size. "
    "Human review is required before drawing conclusions about real-world quality."
)


class _DecimalEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        return super().default(o)


def _split_metrics_to_dict(m: SplitMetrics) -> dict[str, Any]:
    """Serialise a SplitMetrics dataclass to a JSON-compatible dict."""
    unanswerable_denominator = m.unanswerable_count
    answerable_denominator = m.answerable_count

    # --- Abstention fractions (avoid ZeroDivisionError) ---
    def _frac(numerator: int, denominator: int) -> float | None:
        if denominator == 0:
            return None
        return round(numerator / denominator, 4)

    abstention = {
        "correct_abstentions": m.correct_abstentions,
        "correct_abstention_rate": _frac(
            m.correct_abstentions, unanswerable_denominator
        ),
        "unsupported_answers": m.unsupported_answers,
        "unsupported_answer_rate": _frac(
            m.unsupported_answers, unanswerable_denominator
        ),
        "unnecessary_abstentions": m.unnecessary_abstentions,
        "unnecessary_abstention_rate": _frac(
            m.unnecessary_abstentions, answerable_denominator
        ),
        "denominator_unanswerable": unanswerable_denominator,
        "denominator_answerable": answerable_denominator,
        "note": (
            "Abstention mechanism: the pipeline returns status='no_context' when "
            "the retriever returns zero passages. No configurable threshold exists. "
            "Calibration on dev split: inspect unnecessary_abstention_rate and "
            "correct_abstention_rate to understand retrieval coverage."
        ),
    }

    # Pass 1.5 ranking metrics and analytic references
    retrieval: dict[str, Any] = {
        "recall_at_1": {
            "metric": "Recall@1",
            "aggregation": "macro average over answerable questions",
            "mean": (
                round(m.recall_at_1_mean, 4)
                if m.recall_at_1_mean is not None
                else None
            ),
            "random_chance_reference": (
                round(m.random_chance_recall_at_1, 4)
                if m.random_chance_recall_at_1 is not None
                else None
            ),
            "values": [round(v, 4) for v in m.recall_at_1_values],
        },
        "recall_at_3": {
            "metric": "Recall@3",
            "aggregation": "macro average over answerable questions",
            "mean": (
                round(m.recall_at_3_mean, 4)
                if m.recall_at_3_mean is not None
                else None
            ),
            "random_chance_reference": (
                round(m.random_chance_recall_at_3, 4)
                if m.random_chance_recall_at_3 is not None
                else None
            ),
            "values": [round(v, 4) for v in m.recall_at_3_values],
        },
        "recall_at_5": {
            "metric": "Recall@5",
            "aggregation": "macro average over answerable questions",
            "mean": (
                round(m.recall_at_5_mean, 4)
                if m.recall_at_5_mean is not None
                else None
            ),
            "random_chance_reference": (
                round(m.random_chance_recall_at_5, 4)
                if m.random_chance_recall_at_5 is not None
                else None
            ),
            "values": [round(v, 4) for v in m.recall_at_5_values],
        },
        "mrr": {
            "metric": "MRR",
            "aggregation": "macro average reciprocal rank over answerable questions",
            "definition": (
                "Reciprocal rank of the first expected-evidence chunk found in "
                "the retrieved list (1.0 / rank of first hit, or 0.0 if none found)"
            ),
            "mean": round(m.mrr_mean, 4) if m.mrr_mean is not None else None,
            "values": [round(v, 4) for v in m.mrr_values],
        },
        "random_chance_note": (
            "random_chance_reference is an analytic reference computed as "
            "min(1.0, k / corpus_size), representing the expected recall of a "
            "uniform random retriever without replacement. It is NOT a measurement."
        ),
        # Backward compatibility aliases for Pass 1 report consumers
        "metric": "Recall@5",
        "aggregation": "macro average over answerable questions",
        "mean": (
            round(m.recall_at_5_mean, 4)
            if m.recall_at_5_mean is not None
            else None
        ),
        "values": [round(v, 4) for v in m.recall_at_5_values],
        "answerable_questions_evaluated": len(m.recall_at_5_values),
    }

    latency: dict[str, Any] = {
        "unit": "milliseconds",
        "mean": (
            round(m.latency_ms_mean, 1) if m.latency_ms_mean is not None else None
        ),
        "p50": (
            round(m.latency_ms_p50, 1) if m.latency_ms_p50 is not None else None
        ),
        "p95": (
            round(m.latency_ms_p95, 1) if m.latency_ms_p95 is not None else None
        ),
        "sample_count": len(m.latency_ms_values),
    }

    cost: dict[str, Any] = {
        "total_cost_usd": str(m.total_cost_sum) if m.cost_available else "unavailable",
        "embedding_tokens_total": m.embedding_tokens_sum if m.cost_available else "unavailable",
        "note": (
            "Cost is only reported when the pipeline is run with a RAGBudget "
            "(live mode with valid credentials). In deterministic mode all cost "
            "fields are 'unavailable'."
        ),
    }

    return {
        "split": m.split,
        "total": m.total,
        "answerable_count": m.answerable_count,
        "unanswerable_count": m.unanswerable_count,
        "failed_count": m.failed_count,
        "retrieval": retrieval,
        "abstention": abstention,
        "latency": latency,
        "cost": cost,
    }


def _question_result_to_dict(qr: QuestionResult) -> dict[str, Any]:
    return {
        "id": qr.question_id,
        "split": qr.split,
        "answerable": qr.answerable,
        "generation_status": qr.generation_status,
        "recall_at_1": (
            round(qr.recall_at_1, 4) if qr.recall_at_1 is not None else None
        ),
        "recall_at_3": (
            round(qr.recall_at_3, 4) if qr.recall_at_3 is not None else None
        ),
        "recall_at_5": (
            round(qr.recall_at_5, 4) if qr.recall_at_5 is not None else None
        ),
        "reciprocal_rank": (
            round(qr.reciprocal_rank, 4) if qr.reciprocal_rank is not None else None
        ),
        "recall_at_k": (
            round(qr.recall_at_k, 4) if qr.recall_at_k is not None else None
        ),
        "latency_ms": qr.latency_ms,
        "total_cost": str(qr.total_cost) if qr.total_cost is not None else None,
        "embedding_tokens": qr.embedding_tokens,
        "retries": qr.retries,
        "failed": qr.failed,
        "error": sanitize_secrets(qr.error),
        "answer": sanitize_secrets(qr.answer),
        "citations": qr.citations,
        "retrieved": qr.retrieved_chunks,
        "citation_validity": qr.citation_validity,
    }


def write_report(
    *,
    output_path: Path,
    dataset_path: Path,
    dataset_size: int,
    mode: str,
    model_config: dict[str, Any],
    top_k: int,
    question_results: list[QuestionResult],
    split_metrics: dict[str, SplitMetrics],
    git_commit: str | None = None,
    dataset_sha256: str | None = None,
    corpus_version: str | None = None,
    prompt_version: str | None = None,
    generator_model: str | None = None,
    temperature: float | None = None,
    pending_question_ids: list[str] | None = None,
    quota_error: dict[str, Any] | None = None,
    complete: bool | None = None,
) -> dict[str, Any]:
    """
    Build and write the evaluation report to *output_path*.

    Returns the report dict (for testing and inspection).
    """
    timestamp = datetime.now(tz=UTC).isoformat()
    failed_question_ids = [qr.question_id for qr in question_results if qr.failed]
    
    if complete is None:
        has_pending = pending_question_ids is not None and len(pending_question_ids) > 0
        complete = len(failed_question_ids) == 0 and not has_pending

    # Resolve metadata headers
    effective_git_commit = git_commit or get_git_commit()
    effective_dataset_sha256 = dataset_sha256 or compute_file_sha256(dataset_path)
    effective_corpus_version = corpus_version or CORPUS_VERSION
    effective_prompt_version = (
        prompt_version
        or model_config.get("prompt_version")
        or DEFAULT_PROMPT_VERSION
    )
    effective_generator_model = (
        generator_model
        or model_config.get("generator")
        or model_config.get("generator_model")
        or DEFAULT_GENERATOR_MODEL
    )
    effective_temperature = (
        temperature
        if temperature is not None
        else model_config.get("temperature", DEFAULT_TEMPERATURE)
    )

    config_fingerprint = compute_config_fingerprint(
        dataset_sha256=effective_dataset_sha256,
        corpus_version=effective_corpus_version,
        prompt_version=effective_prompt_version,
        generator_model=effective_generator_model,
        temperature=effective_temperature,
        top_k=top_k,
        mode=mode,
    )

    caveats: list[str] = [CORPUS_DISCLAIMER]
    if mode == "deterministic":
        caveats.append(DETERMINISTIC_DISCLAIMER)

    if not complete:
        parts: list[str] = []
        if failed_question_ids:
            parts.append(
                f"{len(failed_question_ids)} question(s) failed after retries: {failed_question_ids}"
            )
        if pending_question_ids:
            parts.append(
                f"{len(pending_question_ids)} question(s) pending/stopped: {pending_question_ids}"
            )
        if quota_error:
            parts.append(f"quota details: {quota_error}")
        caveats.append(
            f"INCOMPLETE RUN: {'; '.join(parts)}. Do NOT use these metrics as a baseline."
        )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": timestamp,
        "mode": mode,
        "complete": complete,
        "git_commit": effective_git_commit,
        "dataset_sha256": effective_dataset_sha256,
        "corpus_version": effective_corpus_version,
        "prompt_version": effective_prompt_version,
        "generator_model": effective_generator_model,
        "temperature": effective_temperature,
        "top_k": top_k,
        "config_fingerprint": config_fingerprint,
        "failed_question_ids": failed_question_ids,
        "pending_question_ids": pending_question_ids or [],
        "dataset": {
            "path": str(dataset_path),
            "sha256": effective_dataset_sha256,
            "size": dataset_size,
            "corpus_version": effective_corpus_version,
            "corpus_documents": CORPUS_DOCUMENT_COUNT,
            "corpus_chunks": CORPUS_CHUNK_COUNT,
        },
        "configuration": {
            **model_config,
            "generator_model": effective_generator_model,
            "temperature": effective_temperature,
            "top_k": top_k,
            "prompt_version": effective_prompt_version,
            "corpus_version": effective_corpus_version,
            "dataset_sha256": effective_dataset_sha256,
            "git_commit": effective_git_commit,
            "config_fingerprint": config_fingerprint,
        },
        "caveats": caveats,
        "splits": {
            split: _split_metrics_to_dict(m) for split, m in split_metrics.items()
        },
        "per_question": [_question_result_to_dict(qr) for qr in question_results],
    }

    if quota_error:
        report["quota_error"] = quota_error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, cls=_DecimalEncoder)
        fh.write("\n")

    return report
