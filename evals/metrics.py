"""
Evaluation metrics for the RAG evaluation suite (Pass 1 and Pass 1.5).

All metric functions are pure Python — no side effects, no I/O.

Aggregation note
----------------
Recall@K and MRR are reported as **macro averages** over answerable questions:
each answerable question gets an equal weight regardless of how many expected
chunks it has. Unanswerable questions are excluded from retrieval metrics entirely.

Pass 1.5 Ranking Metrics
------------------------
- Recall@1: fraction of expected chunks retrieved in top 1.
- Recall@3: fraction of expected chunks retrieved in top 3.
- Recall@5: fraction of expected chunks retrieved in top 5.
- MRR (Mean Reciprocal Rank): reciprocal rank of the first expected-evidence chunk
  found in the retrieved list (0.0 if none found).
  Multi-evidence definition: For questions with multiple expected evidence chunks,
  the reciprocal rank is determined by the *first* (highest-ranked) expected chunk
  appearing in the retrieved list: 1.0 / min({rank(e) : e in expected_refs and e in retrieved_refs}).
- Random-chance Recall@K reference: analytic expected value min(1.0, k / corpus_size).
  Reported alongside measured recall to provide context for corpus size vs k.

Abstention metrics
------------------
The pipeline's abstention mechanism is implicit: the ``RAGService`` calls the
generator with the retrieved passages. When the retriever returns zero passages
the ``GenerationService`` short-circuits and returns ``status="no_context"``
without making an LLM call. This is the sole abstention signal available in the
existing pipeline. No threshold is involved; the behaviour is binary.

Three categories are tracked (denominators documented per metric):

* ``correct_abstentions``    — unanswerable questions where status is "no_context".
                               Denominator: all unanswerable questions.
* ``unsupported_answers``    — unanswerable questions where status is "answered"
                               (the pipeline produced an answer with no grounded
                               evidence). Denominator: all unanswerable questions.
* ``unnecessary_abstentions``— answerable questions where status is "no_context".
                               Denominator: all answerable questions.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from evals.datasets.corpus import VALID_EVIDENCE_REFS

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

GenerationStatus = Literal[
    "answered", "no_context", "blocked", "failed", "retrieval_only"
]


@dataclass(frozen=True)
class RetrievalResult:
    """What the retriever returned for one question."""

    #: Stable (document_id, chunk_index) pairs in retrieval-rank order.
    retrieved_refs: list[tuple[str, int]]
    #: Generation status returned by the pipeline.
    generation_status: GenerationStatus
    #: End-to-end wall-clock latency in milliseconds (None if unavailable).
    latency_ms: int | None = None
    #: Total monetary cost (None if budget was not attached).
    total_cost: Decimal | None = None
    #: Embedding token count (None if unavailable).
    embedding_tokens: int | None = None
    #: Number of retries performed for this question.
    retries: int = 0
    #: Retrieved chunks with similarity scores: (doc_id, chunk_idx, score).
    retrieved_scores: list[tuple[str, int, float]] = field(default_factory=list)
    #: Generated answer text (None if generation skipped or failed).
    answer: str | None = None
    #: Citations produced during generation.
    citations: list[dict[str, Any]] = field(default_factory=list)
    #: Citation validity evaluation result.
    citation_validity: dict[str, Any] | None = None
    #: Whether this failure was due to 429 Daily Quota exhaustion.
    is_daily_quota: bool = False
    #: Error details if daily quota was hit.
    quota_details: str | None = None


@dataclass
class QuestionResult:
    """Per-question evaluation outcome."""

    question_id: str
    split: Literal["dev", "test"]
    answerable: bool
    generation_status: GenerationStatus
    #: Recall@5 (or primary K) for backward compatibility. None for unanswerable.
    recall_at_k: float | None = None
    #: Recall@1 for this question. None for unanswerable questions.
    recall_at_1: float | None = None
    #: Recall@3 for this question. None for unanswerable questions.
    recall_at_3: float | None = None
    #: Recall@5 for this question. None for unanswerable questions.
    recall_at_5: float | None = None
    #: Reciprocal rank (1/rank of first hit, or 0.0). None for unanswerable.
    reciprocal_rank: float | None = None
    latency_ms: int | None = None
    total_cost: Decimal | None = None
    embedding_tokens: int | None = None
    #: Human-readable error message if the pipeline raised an exception.
    error: str | None = None
    #: Whether this question failed (exception during pipeline call).
    failed: bool = False
    #: Number of retries performed for this question.
    retries: int = 0
    #: Generated answer text (None if generation skipped or failed).
    answer: str | None = None
    #: Citations produced during generation.
    citations: list[dict[str, Any]] = field(default_factory=list)
    #: Retrieved items with similarity scores: [{"document_id": str, "chunk_index": int, "score": float}].
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    #: Citation validity evaluation result.
    citation_validity: dict[str, Any] | None = None
    #: Whether this failure was due to 429 Daily Quota exhaustion.
    is_daily_quota: bool = False
    #: Error details if daily quota was hit.
    quota_details: str | None = None


@dataclass
class SplitMetrics:
    """Aggregated metrics for one dataset split (dev or test)."""

    split: str
    total: int = 0
    answerable_count: int = 0
    unanswerable_count: int = 0
    failed_count: int = 0

    # --- Ranking / Retrieval Metrics ---
    recall_at_1_mean: float | None = None
    recall_at_1_values: list[float] = field(default_factory=list)

    recall_at_3_mean: float | None = None
    recall_at_3_values: list[float] = field(default_factory=list)

    recall_at_5_mean: float | None = None
    recall_at_5_values: list[float] = field(default_factory=list)

    mrr_mean: float | None = None
    mrr_values: list[float] = field(default_factory=list)

    #: Backward compatibility alias for Recall@5 macro mean
    recall_at_k_mean: float | None = None
    recall_at_k_values: list[float] = field(default_factory=list)

    #: Analytic random-chance references (min(1.0, k / corpus_size))
    random_chance_recall_at_1: float | None = None
    random_chance_recall_at_3: float | None = None
    random_chance_recall_at_5: float | None = None

    # --- Abstention ---
    correct_abstentions: int = 0  # unanswerable → no_context
    unsupported_answers: int = 0  # unanswerable → answered
    unnecessary_abstentions: int = 0  # answerable   → no_context

    # --- Latency ---
    latency_ms_values: list[int] = field(default_factory=list)
    latency_ms_mean: float | None = None
    latency_ms_p50: float | None = None
    latency_ms_p95: float | None = None

    # --- Cost / tokens ---
    total_cost_sum: Decimal = field(default_factory=lambda: Decimal(0))
    embedding_tokens_sum: int = 0
    cost_available: bool = False


# ---------------------------------------------------------------------------
# Citation Validity Evaluation
# ---------------------------------------------------------------------------


def evaluate_citation_validity(
    generation_status: str,
    citations: Sequence[tuple[str, int] | dict[str, Any]],
    retrieved_refs: Sequence[tuple[str, int]],
    invalid_markers_count: int = 0,
) -> dict[str, Any]:
    """
    Evaluate the validity of citations produced by the generator.

    A generation result has valid citations if:
    - No citation markers refer to unsupplied passage numbers (invalid_markers_count == 0).
    - Every cited passage reference was among the retrieved passages.
    - If status is 'no_context' or 'retrieval_only', no citations are expected.
    """
    if generation_status in ("no_context", "retrieval_only"):
        return {
            "valid": True,
            "cited_count": 0,
            "invalid_markers_count": 0,
            "unretrieved_citations_count": 0,
            "reason": f"No citations expected for status {generation_status!r}",
        }

    normalized_citations: list[tuple[str, int]] = []
    for c in citations:
        if isinstance(c, tuple) and len(c) == 2:
            normalized_citations.append((str(c[0]), int(c[1])))
        elif isinstance(c, dict) and "document_id" in c and "chunk_index" in c:
            normalized_citations.append((str(c["document_id"]), int(c["chunk_index"])))

    try:
        invalid_markers = int(invalid_markers_count)
    except (TypeError, ValueError):
        invalid_markers = 0

    retrieved_set = set(retrieved_refs)
    unretrieved = [ref for ref in normalized_citations if ref not in retrieved_set]

    is_valid = (invalid_markers == 0) and (len(unretrieved) == 0)

    if invalid_markers > 0:
        reason = f"Contains {invalid_markers} invalid citation marker(s)"
    elif unretrieved:
        reason = f"Contains {len(unretrieved)} citation(s) not in retrieved passages"
    else:
        reason = "All citations refer to retrieved passages"

    return {
        "valid": is_valid,
        "cited_count": len(normalized_citations),
        "invalid_markers_count": invalid_markers,
        "unretrieved_citations_count": len(unretrieved),
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Pure Ranking Metric Functions
# ---------------------------------------------------------------------------


def recall_at_k(
    expected_refs: list[tuple[str, int]],
    retrieved_refs: list[tuple[str, int]],
    k: int = 5,
) -> float:
    """
    Compute Recall@K for one answerable question.

    Parameters
    ----------
    expected_refs:
        Ground-truth (document_id, chunk_index) pairs. Must be non-empty for
        answerable questions.
    retrieved_refs:
        Refs returned by the retriever in rank order.
    k:
        Maximum number of retrieved results to consider (k >= 1).

    Returns
    -------
    float
        Fraction of expected_refs found in top-K retrieved_refs. Range [0.0, 1.0].
    """
    if not expected_refs:
        raise ValueError(
            "recall_at_k: expected_refs must be non-empty for answerable questions"
        )
    if k <= 0:
        raise ValueError("k must be greater than zero")

    top_k_set = set(retrieved_refs[:k])
    hits = sum(1 for ref in expected_refs if ref in top_k_set)
    return hits / len(expected_refs)


def recall_at_1(
    expected_refs: list[tuple[str, int]],
    retrieved_refs: list[tuple[str, int]],
) -> float:
    """Compute Recall@1 for one answerable question."""
    return recall_at_k(expected_refs=expected_refs, retrieved_refs=retrieved_refs, k=1)


def recall_at_3(
    expected_refs: list[tuple[str, int]],
    retrieved_refs: list[tuple[str, int]],
) -> float:
    """Compute Recall@3 for one answerable question."""
    return recall_at_k(expected_refs=expected_refs, retrieved_refs=retrieved_refs, k=3)


def reciprocal_rank(
    expected_refs: list[tuple[str, int]],
    retrieved_refs: list[tuple[str, int]],
) -> float:
    """
    Compute the Reciprocal Rank (RR) for a single answerable question.

    Definition for multi-evidence questions:
    ----------------------------------------
    If a question has multiple expected evidence chunks, its reciprocal rank is
    determined by the *first* (highest-ranked) expected chunk found in the
    retrieved list:
        RR = 1.0 / rank  (where rank is 1-based)
    If none of the expected evidence chunks appear anywhere in retrieved_refs,
    the reciprocal rank is 0.0.

    Parameters
    ----------
    expected_refs:
        Ground-truth (document_id, chunk_index) pairs. Must be non-empty for
        answerable questions.
    retrieved_refs:
        Refs returned by the retriever in rank order.

    Returns
    -------
    float
        Reciprocal rank in range [0.0, 1.0].
    """
    if not expected_refs:
        raise ValueError(
            "reciprocal_rank: expected_refs must be non-empty for answerable questions"
        )

    expected_set = set(expected_refs)
    for rank, ref in enumerate(retrieved_refs, start=1):
        if ref in expected_set:
            return 1.0 / rank

    return 0.0


def random_chance_recall_at_k(
    corpus_size: int,
    k: int,
    expected_evidence_counts: Sequence[int] | None = None,
) -> float:
    """
    Compute the analytic expected value of Recall@k under uniform random retrieval.

    Analytic Reference Derivation:
    ------------------------------
    Consider a corpus of N distinct chunks from which a retriever samples k items
    uniformly at random without replacement.
    For each distinct chunk in the corpus, the marginal probability of being
    included in a random sample of size k is:
        P(chunk in sample) = min(1.0, k / N)

    For an answerable question with E ground-truth expected chunks (where E >= 1),
    let X be the number of expected chunks included in the retrieved k items.
    By linearity of expectation:
        E[X] = sum_{j=1}^E P(chunk_j in sample) = E * min(1.0, k / N)

    Recall@k for that question is defined as X / E. Therefore, the expected Recall@k is:
        E[Recall@k] = E[X / E] = (1 / E) * E[X] = (1 / E) * (E * min(1.0, k / N))
                    = min(1.0, k / N)

    Notice that this expectation is independent of E (the count of expected chunks).
    Consequently, the macro-average over any collection of answerable questions is
    also identically min(1.0, k / N).

    This value is an analytic baseline reference representing random retrieval
    performance, NOT an empirical measurement.

    Parameters
    ----------
    corpus_size:
        Total number of chunks in the corpus (N).
    k:
        Number of items retrieved (k >= 1).
    expected_evidence_counts:
        Optional sequence of expected-evidence counts per question. If supplied,
        each count must be >= 1. The expectation remains min(1.0, k / corpus_size).

    Returns
    -------
    float
        Analytic expected recall in range [0.0, 1.0].
    """
    if corpus_size <= 0:
        return 0.0
    if k <= 0:
        return 0.0
    if k >= corpus_size:
        return 1.0

    if expected_evidence_counts is not None and len(expected_evidence_counts) > 0:
        for count in expected_evidence_counts:
            if count <= 0:
                raise ValueError("expected_evidence_counts must be positive integers")

    return min(1.0, float(k) / float(corpus_size))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _safe_percentile(values: Sequence[int | float], pct: float) -> float | None:
    """Return the *pct* percentile (0–100) of *values*, or None if empty."""
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


def compute_split_metrics(
    results: list[QuestionResult],
    split: str,
    top_k: int = 5,
    corpus_size: int | None = None,
) -> SplitMetrics:
    """
    Aggregate per-question results into split-level metrics.

    Only results matching *split* are included.
    """
    if corpus_size is None:
        corpus_size = len(VALID_EVIDENCE_REFS)

    split_results = [r for r in results if r.split == split]
    m = SplitMetrics(split=split)
    m.total = len(split_results)

    for qr in split_results:
        if qr.failed:
            m.failed_count += 1
            # Failed questions are not counted toward any quality metric.
            continue

        if qr.answerable:
            m.answerable_count += 1
        else:
            m.unanswerable_count += 1

        # Abstention.
        if not qr.answerable:
            if qr.generation_status == "no_context":
                m.correct_abstentions += 1
            elif qr.generation_status == "answered":
                m.unsupported_answers += 1
            # "blocked" is neither correct abstention nor unsupported answer.
        else:
            if qr.generation_status == "no_context":
                m.unnecessary_abstentions += 1

        # Ranking metrics.
        if qr.recall_at_1 is not None:
            m.recall_at_1_values.append(qr.recall_at_1)
        if qr.recall_at_3 is not None:
            m.recall_at_3_values.append(qr.recall_at_3)
        if qr.recall_at_5 is not None:
            m.recall_at_5_values.append(qr.recall_at_5)
        elif qr.recall_at_k is not None:
            m.recall_at_5_values.append(qr.recall_at_k)

        if qr.reciprocal_rank is not None:
            m.mrr_values.append(qr.reciprocal_rank)

        # Latency.
        if qr.latency_ms is not None:
            m.latency_ms_values.append(qr.latency_ms)

        # Cost.
        if qr.total_cost is not None:
            m.total_cost_sum += qr.total_cost
            m.cost_available = True

        if qr.embedding_tokens is not None:
            m.embedding_tokens_sum += qr.embedding_tokens

    # Means (macro averages over answerable questions).
    if m.recall_at_1_values:
        m.recall_at_1_mean = statistics.mean(m.recall_at_1_values)
    if m.recall_at_3_values:
        m.recall_at_3_mean = statistics.mean(m.recall_at_3_values)
    if m.recall_at_5_values:
        m.recall_at_5_mean = statistics.mean(m.recall_at_5_values)
    if m.mrr_values:
        m.mrr_mean = statistics.mean(m.mrr_values)

    # Backward compatibility aliases.
    m.recall_at_k_values = m.recall_at_5_values
    m.recall_at_k_mean = m.recall_at_5_mean

    # Analytic random-chance references.
    m.random_chance_recall_at_1 = random_chance_recall_at_k(corpus_size, 1)
    m.random_chance_recall_at_3 = random_chance_recall_at_k(corpus_size, 3)
    m.random_chance_recall_at_5 = random_chance_recall_at_k(corpus_size, 5)

    # Latency statistics.
    if m.latency_ms_values:
        m.latency_ms_mean = statistics.mean(m.latency_ms_values)
        m.latency_ms_p50 = _safe_percentile(m.latency_ms_values, 50)
        m.latency_ms_p95 = _safe_percentile(m.latency_ms_values, 95)

    return m
