"""
RAG Evaluation Runner — Pass 1 and Pass 1.5.

Usage
-----
Deterministic mode (no credentials required, validates harness):

    uv run python -m evals.evaluate_rag \\
        --dataset evals/datasets/rag_golden.jsonl \\
        --mode deterministic \\
        --output evals/results/latest.json

Live mode (requires GEMINI_API_KEY and a running PostgreSQL/pgvector instance):

    uv run python -m evals.evaluate_rag \\
        --dataset evals/datasets/rag_golden.jsonl \\
        --mode live \\
        --split dev \\
        --output evals/results/baseline_dev_before_abstention.json

Exit codes
----------
0  All questions processed and report written (complete run).
1  Dataset validation failed or a non-recoverable configuration error occurred.
2  One or more questions failed (pipeline exception / retry exhaustion / daily quota); report marked incomplete.

Abstention mechanism
--------------------
This pipeline has no configurable abstention threshold. The generator returns
``status="no_context"`` when the retriever supplies zero passages. This is the
sole abstention signal. To calibrate retrieval coverage, inspect
``unnecessary_abstention_rate`` and ``unsupported_answer_rate`` on the dev split.
Failed questions (exceptions during API calls) are kept strictly distinct from
abstentions and marked as ``generation_status="failed"``.

Security & Tenant Isolation
---------------------------
Live mode uses a dedicated evaluation tenant (default: ``rag-eval-suite-tenant``)
and strictly operates within that tenant ID. Prior to ingestion, existing documents
for that tenant are purged via ``cleanup_evaluation_documents``. Retrieval queries
are scoped by ``Document.tenant_id == tenant_id`` and
``DocumentChunkRecord.tenant_id == tenant_id``, ensuring complete isolation from
real tenant data. API keys and secrets are never printed or logged.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import re
import sys
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from evals.datasets.corpus import (
    CORPUS_CHUNK_COUNT,
    CORPUS_VERSION,
    SYNTHETIC_CORPUS,
)
from evals.fakes import (
    FakeCorpusRetriever,
    FakeGenerator,
    build_passages_from_corpus,
)
from evals.metrics import (
    QuestionResult,
    RetrievalResult,
    SplitMetrics,
    compute_split_metrics,
    evaluate_citation_validity,
    recall_at_1,
    recall_at_3,
    recall_at_k,
    reciprocal_rank,
)
from evals.report import (
    _question_result_to_dict,
    append_checkpoint_record,
    compute_config_fingerprint,
    compute_file_sha256,
    get_git_commit,
    read_checkpoint,
    sanitize_secrets,
    write_report,
)
from evals.schema import GoldenRecord, load_and_validate

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.rag.generation import ContextPassage
    from app.rag.pipeline import Embedder, RAGService

logger = logging.getLogger(__name__)

DEFAULT_EVAL_TENANT_NAME = "rag-eval-suite-tenant"
EVAL_TENANT_ID_DETERMINISTIC = 0
DEFAULT_REQUEST_DELAY_LIVE = 13.0
DEFAULT_REQUEST_DELAY_DETERMINISTIC = 0.0
MAX_LIVE_RETRIES = 4
BASE_BACKOFF_SECONDS = 2.0
MAX_CONSECUTIVE_RETRY_EXHAUSTIONS = 2


# ---------------------------------------------------------------------------
# Tenant isolation and corpus ingestion
# ---------------------------------------------------------------------------


def ensure_eval_tenant(db: Session, tenant_id: int | None = None) -> int:
    """
    Ensure a dedicated evaluation tenant exists in the database.

    If *tenant_id* is provided, checks that it exists (creating it if needed);
    otherwise searches for or creates a tenant named ``DEFAULT_EVAL_TENANT_NAME``.
    Guarantees that evaluation never writes to or alters production tenant data.
    """
    from app.services.models import Tenant

    if tenant_id is not None:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if tenant is None:
            tenant = Tenant(
                id=tenant_id,
                name=f"{DEFAULT_EVAL_TENANT_NAME}-{tenant_id}",
                monthly_budget=Decimal("1000.00"),
                current_spend=Decimal("0.00"),
            )
            db.add(tenant)
            db.commit()
            db.refresh(tenant)
        return tenant.id

    tenant = db.query(Tenant).filter(Tenant.name == DEFAULT_EVAL_TENANT_NAME).first()
    if tenant is None:
        tenant = Tenant(
            name=DEFAULT_EVAL_TENANT_NAME,
            monthly_budget=Decimal("1000.00"),
            current_spend=Decimal("0.00"),
        )
        db.add(tenant)
        db.commit()
        db.refresh(tenant)
    return tenant.id


def cleanup_evaluation_documents(db: Session, tenant_id: int) -> int:
    """
    Delete all documents belonging to the evaluation tenant.

    Foreign key CASCADE in PostgreSQL automatically deletes associated
    ``DocumentChunkRecord`` rows. Scoped strictly by tenant_id, so it cannot
    touch any other tenant's documents.
    """
    from app.services.models import Document

    deleted_count = (
        db.query(Document)
        .filter(Document.tenant_id == tenant_id)
        .delete(synchronize_session=False)
    )
    db.commit()
    logger.info(
        "Cleaned up %d existing document(s) for evaluation tenant %d",
        deleted_count,
        tenant_id,
    )
    return deleted_count


def ingest_evaluation_corpus(
    db: Session,
    tenant_id: int,
    embedder: Embedder,
    corpus: dict[str, list[str]],
) -> dict[str, str]:
    """
    Ingest the synthetic evaluation corpus into the database for *tenant_id*.

    Returns a mapping of ``str(document.id)`` (UUID string) -> canonical doc_id
    (e.g. 'synthetic_hr_policy') so retrieved UUID references can be resolved
    back to golden expected evidence references.
    """
    from app.services.models import Document, DocumentChunkRecord

    uuid_to_doc_id: dict[str, str] = {}
    fingerprint = embedder.fingerprint

    for doc_id, chunks in corpus.items():
        doc_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"rag-eval:{tenant_id}:{doc_id}")
        doc = Document(id=doc_uuid, tenant_id=tenant_id, source=doc_id)
        db.add(doc)
        db.flush()

        uuid_to_doc_id[str(doc.id)] = doc_id

        texts = list(chunks)
        embed_docs = getattr(embedder, "embed_documents", None)
        if callable(embed_docs):
            vectors: list[list[float]] = cast("list[list[float]]", embed_docs(texts))
        else:
            vectors = [embedder.embed_query(t) for t in texts]

        for chunk_index, (text, vector) in enumerate(zip(texts, vectors, strict=True)):
            chunk_record = DocumentChunkRecord(
                tenant_id=tenant_id,
                document_id=doc.id,
                chunk_index=chunk_index,
                page=chunk_index,
                text=text,
                metadata_={"source": doc_id},
                embedding=vector,
                fingerprint=fingerprint,
            )
            db.add(chunk_record)

    db.commit()
    logger.info(
        "Ingested %d documents (%d chunks) for evaluation tenant %d (fingerprint=%s)",
        len(corpus),
        sum(len(c) for c in corpus.values()),
        tenant_id,
        fingerprint,
    )
    return uuid_to_doc_id


# ---------------------------------------------------------------------------
# EvalRetriever with score recording
# ---------------------------------------------------------------------------


class EvalRetriever:
    """
    Retriever for evaluation that records similarity scores alongside passages.
    Satisfies Retriever protocol.
    """

    def __init__(self, db: Session) -> None:
        self._db = db
        self.last_search_results: list[tuple[ContextPassage, float]] = []

    def search_with_scores(
        self,
        *,
        tenant_id: int,
        vector: Sequence[float],
        k: int = 5,
        fingerprint: str,
        max_distance: float | None = None,
    ) -> list[tuple[ContextPassage, float]]:
        from sqlalchemy import select

        from app.rag.generation import ContextPassage
        from app.services.models import Document, DocumentChunkRecord

        if k <= 0:
            raise ValueError("k must be greater than zero")
        if not vector:
            raise ValueError("vector must not be empty")
        if not fingerprint:
            raise ValueError("fingerprint must not be empty")
        if max_distance is not None and max_distance < 0:
            raise ValueError("max_distance must not be negative")

        distance_expr = DocumentChunkRecord.embedding.cosine_distance(list(vector))

        statement = (
            select(DocumentChunkRecord, distance_expr.label("distance"))
            .join(
                Document,
                Document.id == DocumentChunkRecord.document_id,
            )
            .where(
                Document.tenant_id == tenant_id,
                DocumentChunkRecord.tenant_id == tenant_id,
                DocumentChunkRecord.fingerprint == fingerprint,
            )
            .order_by(distance_expr)
            .limit(k)
        )

        if max_distance is not None:
            statement = statement.where(distance_expr <= max_distance)

        rows = self._db.execute(statement).all()

        results: list[tuple[ContextPassage, float]] = []
        for chunk, dist in rows:
            metadata = chunk.metadata_ or {}
            passage = ContextPassage(
                ref=(str(chunk.document_id), chunk.chunk_index),
                text=chunk.text,
                source=metadata.get("source"),
                page=chunk.page,
            )
            # cosine distance = 1 - cosine similarity
            sim = 1.0 - float(dist)
            results.append((passage, sim))

        self.last_search_results = results
        return results

    def search(
        self,
        *,
        tenant_id: int,
        vector: Sequence[float],
        k: int = 5,
        fingerprint: str,
        max_distance: float | None = None,
    ) -> list[ContextPassage]:
        scored = self.search_with_scores(
            tenant_id=tenant_id,
            vector=vector,
            k=k,
            fingerprint=fingerprint,
            max_distance=max_distance,
        )
        return [p for p, _ in scored]


# ---------------------------------------------------------------------------
# Per-question evaluation logic and rate-limit retries
# ---------------------------------------------------------------------------


def _extract_retry_delay(exc: Exception) -> float | None:
    """
    Extract any retry delay (in seconds) indicated by the provider exception.
    """
    targets = [exc]
    cause = getattr(exc, "__cause__", None)
    if cause is not None and isinstance(cause, Exception):
        targets.append(cause)

    for target in targets:
        for attr in ("retry_after", "retry_delay", "delay"):
            val = getattr(target, attr, None)
            if isinstance(val, (int, float)) and val > 0:
                return float(val)

        # Check response headers if present
        resp = getattr(target, "response", None)
        if resp is not None and hasattr(resp, "headers"):
            headers = resp.headers
            h_val = headers.get("retry-after") or headers.get("Retry-After")
            if h_val is not None:
                try:
                    f = float(h_val)
                    if f > 0:
                        return f
                except ValueError:
                    pass

    # Regex search in error message
    msg = f"{exc} {cause}"
    match = re.search(
        r"retry(?:\s+after|\s+in)?\s+(\d+(?:\.\d+)?)\s*s?", msg, re.IGNORECASE
    )
    if match:
        try:
            delay = float(match.group(1))
            if delay > 0:
                return delay
        except ValueError:
            pass

    return None


def _is_daily_quota_error(exc: Exception) -> tuple[bool, str | None]:
    """
    Check if an exception represents a 429 Daily Quota exhaustion.

    Daily quotas cannot be resolved by short retries, requiring immediate stop.
    """
    targets = [exc]
    cause = getattr(exc, "__cause__", None)
    if cause is not None and isinstance(cause, Exception):
        targets.append(cause)

    daily_patterns = [
        re.compile(r"daily", re.IGNORECASE),
        re.compile(r"per\s*day", re.IGNORECASE),
        re.compile(r"perday", re.IGNORECASE),
        re.compile(r"free_tier.*requests_per_day", re.IGNORECASE),
        re.compile(r"quota\s*limit.*day", re.IGNORECASE),
        re.compile(r"RESOURCE_EXHAUSTED.*day", re.IGNORECASE),
    ]

    for target in targets:
        msg = str(target)
        for pat in daily_patterns:
            if pat.search(msg):
                return True, sanitize_secrets(msg)

        resp = getattr(target, "response", None)
        if resp is not None and hasattr(resp, "text"):
            resp_text = str(resp.text)
            for pat in daily_patterns:
                if pat.search(resp_text):
                    return True, sanitize_secrets(resp_text)

    return False, None


async def _run_one_question_deterministic(
    record: GoldenRecord,
    retriever: FakeCorpusRetriever,
    generator: FakeGenerator,
    top_k: int,
    *,
    retrieval_only: bool = False,
) -> tuple[RetrievalResult, str | None]:
    """
    Run one question through the deterministic pipeline.

    Returns (RetrievalResult, error_message_or_None).
    """
    from app.rag.embeddings import FakeEmbedder
    from app.rag.pipeline import RAGService

    embedder = FakeEmbedder(dimensions=8)

    if retrieval_only:
        started = time.monotonic()
        try:
            query_vec = embedder._embed_text(record["query"])
            scored = retriever.search_with_scores(
                tenant_id=EVAL_TENANT_ID_DETERMINISTIC,
                vector=query_vec,
                k=top_k,
                fingerprint="fake",
            )
            latency_ms = int((time.monotonic() - started) * 1000)
        except Exception as exc:
            latency_ms = int((time.monotonic() - started) * 1000)
            return (
                RetrievalResult(
                    retrieved_refs=[],
                    generation_status="failed",
                    latency_ms=latency_ms,
                    retries=0,
                ),
                traceback.format_exception_only(type(exc), exc)[-1].strip(),
            )

        canonical_retrieved = [(p.ref[0], p.ref[1]) for p, _ in scored]
        scored_chunks = [(p.ref[0], p.ref[1], s) for p, s in scored]
        cit_validity = evaluate_citation_validity(
            generation_status="retrieval_only",
            citations=[],
            retrieved_refs=canonical_retrieved,
            invalid_markers_count=0,
        )
        return (
            RetrievalResult(
                retrieved_refs=canonical_retrieved,
                generation_status="retrieval_only",
                latency_ms=latency_ms,
                retries=0,
                retrieved_scores=scored_chunks,
                answer=None,
                citations=[],
                citation_validity=cit_validity,
            ),
            None,
        )

    service = RAGService(
        embedder=embedder,
        retriever=retriever,
        generator=generator,
    )

    started = time.monotonic()
    try:
        result = await service.answer(
            tenant_id=EVAL_TENANT_ID_DETERMINISTIC,
            query=record["query"],
            top_k=top_k,
        )
    except Exception as exc:
        latency_ms = int((time.monotonic() - started) * 1000)
        logger.warning(
            "Question %s failed: %s",
            record["id"],
            type(exc).__name__,
            exc_info=False,
        )
        return (
            RetrievalResult(
                retrieved_refs=[],
                generation_status="failed",
                latency_ms=latency_ms,
                retries=0,
            ),
            traceback.format_exception_only(type(exc), exc)[-1].strip(),
        )

    latency_ms = int((time.monotonic() - started) * 1000)

    scored_chunks_det: list[tuple[str, int, float]] = []
    for passage, score in retriever.last_search_results:
        scored_chunks_det.append((passage.ref[0], passage.ref[1], score))

    canonical_citations: list[dict[str, Any]] = [
        {"document_id": c.ref[0], "chunk_index": c.ref[1]}
        for c in result.generation.citations
    ]
    cit_validity = evaluate_citation_validity(
        generation_status=result.generation.status,
        citations=canonical_citations,
        retrieved_refs=result.retrieved,
        invalid_markers_count=result.generation.invalid_citation_markers,
    )

    return (
        RetrievalResult(
            retrieved_refs=result.retrieved,
            generation_status=result.generation.status,  # type: ignore[arg-type]
            latency_ms=latency_ms,
            total_cost=result.total_cost or None,
            embedding_tokens=result.embedding_tokens,
            retries=0,
            retrieved_scores=scored_chunks_det,
            answer=result.generation.answer,
            citations=canonical_citations,
            citation_validity=cit_validity,
        ),
        None,
    )


async def _run_one_question_live(
    record: GoldenRecord,
    service: RAGService,
    tenant_id: int,
    top_k: int,
    uuid_to_doc_id: dict[str, str],
    *,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    max_retries: int = MAX_LIVE_RETRIES,
    base_backoff: float = BASE_BACKOFF_SECONDS,
    retrieval_only: bool = False,
) -> tuple[RetrievalResult, str | None]:
    """
    Run one question through the live pipeline with per-question failure isolation,
    exponential backoff with jitter on GenerationRateLimitError, and daily quota detection.

    Returns (RetrievalResult, error_message_or_None).
    """
    started = time.monotonic()
    retries = 0

    actual_retriever = getattr(service, "retriever", None)
    if actual_retriever is None:
        raise ValueError("Service must have a retriever attached")

    if retrieval_only:
        while True:
            try:
                vector = await asyncio.to_thread(
                    service.embedder.embed_query, record["query"]
                )
                if hasattr(actual_retriever, "search_with_scores"):
                    scored = actual_retriever.search_with_scores(
                        tenant_id=tenant_id,
                        vector=vector,
                        k=top_k,
                        fingerprint=service.embedder.fingerprint,
                    )
                else:
                    passages = actual_retriever.search(
                        tenant_id=tenant_id,
                        vector=vector,
                        k=top_k,
                        fingerprint=service.embedder.fingerprint,
                    )
                    scored = [(p, 1.0) for p in passages]
                latency_ms = int((time.monotonic() - started) * 1000)
                break
            except Exception as exc:
                is_daily, quota_msg = _is_daily_quota_error(exc)
                if is_daily:
                    latency_ms = int((time.monotonic() - started) * 1000)
                    return (
                        RetrievalResult(
                            retrieved_refs=[],
                            generation_status="failed",
                            latency_ms=latency_ms,
                            retries=retries,
                            is_daily_quota=True,
                            quota_details=quota_msg,
                        ),
                        quota_msg,
                    )

                from app.rag.generation import GenerationRateLimitError

                is_rate_limit = (
                    isinstance(exc, GenerationRateLimitError)
                    or "ratelimit" in type(exc).__name__.lower()
                )

                if is_rate_limit and retries < max_retries:
                    retries += 1
                    backoff = base_backoff * (2 ** (retries - 1))
                    provider_delay = _extract_retry_delay(exc)
                    delay = (
                        provider_delay
                        if (provider_delay is not None and provider_delay > 0)
                        else backoff
                    )
                    jitter = random.uniform(0.1, 1.0)
                    sleep_duration = delay + jitter
                    await sleep_fn(sleep_duration)
                    continue

                latency_ms = int((time.monotonic() - started) * 1000)
                return (
                    RetrievalResult(
                        retrieved_refs=[],
                        generation_status="failed",
                        latency_ms=latency_ms,
                        retries=retries,
                    ),
                    traceback.format_exception_only(type(exc), exc)[-1].strip(),
                )

        canonical_retrieved = []
        scored_chunks = []
        for passage, score in scored:
            canonical_doc_id = uuid_to_doc_id.get(passage.ref[0], passage.ref[0])
            canonical_retrieved.append((canonical_doc_id, passage.ref[1]))
            scored_chunks.append((canonical_doc_id, passage.ref[1], score))

        cit_validity = evaluate_citation_validity(
            generation_status="retrieval_only",
            citations=[],
            retrieved_refs=canonical_retrieved,
            invalid_markers_count=0,
        )

        return (
            RetrievalResult(
                retrieved_refs=canonical_retrieved,
                generation_status="retrieval_only",
                latency_ms=latency_ms,
                retries=retries,
                retrieved_scores=scored_chunks,
                answer=None,
                citations=[],
                citation_validity=cit_validity,
            ),
            None,
        )

    # Full answer mode
    while True:
        try:
            result = await service.answer(
                tenant_id=tenant_id,
                query=record["query"],
                top_k=top_k,
            )
            latency_ms = int((time.monotonic() - started) * 1000)
            break
        except Exception as exc:
            is_daily, quota_msg = _is_daily_quota_error(exc)
            if is_daily:
                latency_ms = int((time.monotonic() - started) * 1000)
                return (
                    RetrievalResult(
                        retrieved_refs=[],
                        generation_status="failed",
                        latency_ms=latency_ms,
                        retries=retries,
                        is_daily_quota=True,
                        quota_details=quota_msg,
                    ),
                    quota_msg,
                )

            from app.rag.generation import GenerationRateLimitError

            is_rate_limit = (
                isinstance(exc, GenerationRateLimitError)
                or "ratelimit" in type(exc).__name__.lower()
            )

            if is_rate_limit and retries < max_retries:
                retries += 1
                backoff = base_backoff * (2 ** (retries - 1))
                provider_delay = _extract_retry_delay(exc)
                if provider_delay is not None and provider_delay > 0:
                    delay = provider_delay
                else:
                    delay = backoff

                # Full jitter
                jitter = random.uniform(0.1, 1.0)
                sleep_duration = delay + jitter

                logger.warning(
                    "Question %s hit rate limit (retry %d/%d). Backing off for %.2fs...",
                    record["id"],
                    retries,
                    max_retries,
                    sleep_duration,
                )
                await sleep_fn(sleep_duration)
                continue

            latency_ms = int((time.monotonic() - started) * 1000)
            logger.warning(
                "Question %s failed during live execution (after %d retries): %s",
                record["id"],
                retries,
                type(exc).__name__,
                exc_info=False,
            )
            return (
                RetrievalResult(
                    retrieved_refs=[],
                    generation_status="failed",
                    latency_ms=latency_ms,
                    retries=retries,
                ),
                traceback.format_exception_only(type(exc), exc)[-1].strip(),
            )

    # Map retrieved database UUIDs back to canonical corpus document IDs
    canonical_retrieved = []
    scored_chunks = []
    last_scored = getattr(actual_retriever, "last_search_results", None)
    if isinstance(last_scored, list) and len(last_scored) > 0:
        for passage, score in last_scored:
            doc_ref, chunk_idx = passage.ref
            canonical_doc_id = uuid_to_doc_id.get(doc_ref, doc_ref)
            canonical_retrieved.append((canonical_doc_id, chunk_idx))
            scored_chunks.append((canonical_doc_id, chunk_idx, score))
    else:
        for doc_ref, chunk_idx in result.retrieved:
            canonical_doc_id = uuid_to_doc_id.get(doc_ref, doc_ref)
            canonical_retrieved.append((canonical_doc_id, chunk_idx))
            scored_chunks.append((canonical_doc_id, chunk_idx, 1.0))

    canonical_citations = []
    for c in result.generation.citations:
        doc_ref, chunk_idx = c.ref
        canonical_doc_id = uuid_to_doc_id.get(doc_ref, doc_ref)
        canonical_citations.append(
            {"document_id": canonical_doc_id, "chunk_index": chunk_idx}
        )

    cit_validity = evaluate_citation_validity(
        generation_status=result.generation.status,
        citations=canonical_citations,
        retrieved_refs=canonical_retrieved,
        invalid_markers_count=result.generation.invalid_citation_markers,
    )

    return (
        RetrievalResult(
            retrieved_refs=canonical_retrieved,
            generation_status=result.generation.status,  # type: ignore[arg-type]
            latency_ms=latency_ms,
            total_cost=result.total_cost or None,
            embedding_tokens=result.embedding_tokens,
            retries=retries,
            retrieved_scores=scored_chunks,
            answer=result.generation.answer,
            citations=canonical_citations,
            citation_validity=cit_validity,
        ),
        None,
    )


def _make_question_result(
    record: GoldenRecord,
    retrieval: RetrievalResult,
    error: str | None,
    top_k: int,
) -> QuestionResult:
    """Convert a pipeline result into a QuestionResult with ranking metrics."""
    answerable: bool = record["answerable"]
    failed = error is not None

    recall_at_1_val: float | None = None
    recall_at_3_val: float | None = None
    recall_at_5_val: float | None = None
    reciprocal_rank_val: float | None = None
    recall_at_k_val: float | None = None

    if not failed and answerable:
        expected_refs = [
            (ev["document_id"], ev["chunk_index"]) for ev in record["expected_evidence"]
        ]
        recall_at_1_val = recall_at_1(
            expected_refs=expected_refs,
            retrieved_refs=retrieval.retrieved_refs,
        )
        recall_at_3_val = recall_at_3(
            expected_refs=expected_refs,
            retrieved_refs=retrieval.retrieved_refs,
        )
        recall_at_5_val = recall_at_k(
            expected_refs=expected_refs,
            retrieved_refs=retrieval.retrieved_refs,
            k=5,
        )
        reciprocal_rank_val = reciprocal_rank(
            expected_refs=expected_refs,
            retrieved_refs=retrieval.retrieved_refs,
        )
        recall_at_k_val = recall_at_k(
            expected_refs=expected_refs,
            retrieved_refs=retrieval.retrieved_refs,
            k=top_k,
        )

    # Failed questions have status "failed", kept strictly distinct from abstentions ("no_context")
    status = "failed" if failed else retrieval.generation_status

    retrieved_chunks = [
        {"document_id": doc, "chunk_index": idx, "score": round(score, 6)}
        for doc, idx, score in retrieval.retrieved_scores
    ]

    return QuestionResult(
        question_id=record["id"],
        split=record["split"],  # type: ignore[arg-type]
        answerable=answerable,
        generation_status=status,
        recall_at_k=recall_at_k_val,
        recall_at_1=recall_at_1_val,
        recall_at_3=recall_at_3_val,
        recall_at_5=recall_at_5_val,
        reciprocal_rank=reciprocal_rank_val,
        latency_ms=retrieval.latency_ms,
        total_cost=retrieval.total_cost,
        embedding_tokens=retrieval.embedding_tokens,
        error=sanitize_secrets(error),
        failed=failed,
        retries=retrieval.retries,
        answer=sanitize_secrets(retrieval.answer),
        citations=retrieval.citations,
        retrieved_chunks=retrieved_chunks,
        citation_validity=retrieval.citation_validity,
    )


async def run_evaluation(
    records: list[GoldenRecord],
    mode: str,
    top_k: int,
    *,
    tenant_id: int | None = None,
    service: RAGService | None = None,
    uuid_to_doc_id: dict[str, str] | None = None,
    corpus: dict[str, list[str]] | None = None,
    request_delay: float = 0.0,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    retrieval_only: bool = False,
    checkpoint_path: Path | None = None,
    completed_results: dict[str, QuestionResult] | None = None,
    run_state: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> list[QuestionResult]:
    """
    Run evaluation over all *records* in the given *mode* ('deterministic' or 'live'),
    pacing questions with *request_delay*.
    """
    results: list[QuestionResult] = []
    consecutive_exhaustions = 0
    completed_map = completed_results or {}

    retriever: FakeCorpusRetriever | None = None
    generator: FakeGenerator | None = None

    if mode == "deterministic":
        eval_corpus = corpus or SYNTHETIC_CORPUS
        passages = build_passages_from_corpus(eval_corpus)
        retriever = FakeCorpusRetriever(passages=passages)
        generator = FakeGenerator()

    for i, record in enumerate(records):
        qid = record["id"]

        # Check if already completed (e.g. via --resume)
        if qid in completed_map:
            qr = completed_map[qid]
            results.append(qr)
            _log_question(qr)
            continue

        if len(results) > 0 and request_delay > 0:
            await sleep_fn(request_delay)

        if mode == "deterministic":
            assert retriever is not None
            assert generator is not None
            retrieval, error = await _run_one_question_deterministic(
                record=record,
                retriever=retriever,
                generator=generator,
                top_k=top_k,
                retrieval_only=retrieval_only,
            )
            is_daily_quota = False
            quota_details = None
        elif mode == "live":
            if service is None or uuid_to_doc_id is None or tenant_id is None:
                raise ValueError(
                    "Live mode requires an initialized RAGService, tenant_id, and uuid_to_doc_id map."
                )

            retrieval, error = await _run_one_question_live(
                record=record,
                service=service,
                tenant_id=tenant_id,
                top_k=top_k,
                uuid_to_doc_id=uuid_to_doc_id,
                sleep_fn=sleep_fn,
                retrieval_only=retrieval_only,
            )
            is_daily_quota = retrieval.is_daily_quota
            quota_details = retrieval.quota_details
        else:
            raise ValueError(f"Unknown evaluation mode: {mode!r}")

        qr = _make_question_result(
            record=record,
            retrieval=retrieval,
            error=error,
            top_k=top_k,
        )
        results.append(qr)
        _log_question(qr)

        # Streaming checkpoint append
        if checkpoint_path is not None:
            append_checkpoint_record(
                checkpoint_path,
                {
                    "type": "question",
                    **_question_result_to_dict(qr),
                },
            )

        # 1. Immediate stop on 429 Daily Quota
        if is_daily_quota:
            logger.error(
                "Stopping evaluation immediately: 429 daily quota reported on question %s: %s",
                qid,
                quota_details,
            )
            pending_ids = [r["id"] for r in records[i + 1 :]]
            if run_state is not None:
                run_state["stopped_early"] = True
                run_state["pending_ids"] = pending_ids
                run_state["stop_reason"] = "daily_quota"
                run_state["quota_error"] = {
                    "type": "daily_quota",
                    "question_id": qid,
                    "details": quota_details,
                }
            break

        # 2. Immediate stop on 2 consecutive retry exhaustions
        if qr.failed and qr.retries >= MAX_LIVE_RETRIES:
            consecutive_exhaustions += 1
            if consecutive_exhaustions >= MAX_CONSECUTIVE_RETRY_EXHAUSTIONS:
                logger.error(
                    "Stopping evaluation immediately: %d consecutive questions exhausted retries (last: %s).",
                    consecutive_exhaustions,
                    qid,
                )
                pending_ids = [r["id"] for r in records[i + 1 :]]
                if run_state is not None:
                    run_state["stopped_early"] = True
                    run_state["pending_ids"] = pending_ids
                    run_state["stop_reason"] = "consecutive_retries_exhausted"
                break
        else:
            if not qr.failed:
                consecutive_exhaustions = 0

    return results


def _log_question(qr: QuestionResult) -> None:
    status_parts = [
        f"id={qr.question_id}",
        f"split={qr.split}",
        f"answerable={qr.answerable}",
    ]
    if qr.failed:
        status_parts.append(f"status=FAILED (retries={qr.retries}, error={qr.error})")
    else:
        status_parts.append(f"status={qr.generation_status}")
        if qr.retries > 0:
            status_parts.append(f"retries={qr.retries}")
        if qr.recall_at_1 is not None:
            status_parts.append(f"r@1={qr.recall_at_1:.2f}")
        if qr.recall_at_3 is not None:
            status_parts.append(f"r@3={qr.recall_at_3:.2f}")
        if qr.recall_at_5 is not None:
            status_parts.append(f"r@5={qr.recall_at_5:.2f}")
        if qr.reciprocal_rank is not None:
            status_parts.append(f"rr={qr.reciprocal_rank:.2f}")

    if qr.latency_ms is not None:
        status_parts.append(f"latency={qr.latency_ms}ms")
    logger.info("  ".join(status_parts))


def _print_incomplete_warning(failed_or_pending_ids: list[str]) -> None:
    border = "!" * 72
    print(f"\n{border}")
    print("  [CRITICAL WARNING] EVALUATION RUN IS INCOMPLETE")
    print(
        f"  {len(failed_or_pending_ids)} question(s) failed or pending: {failed_or_pending_ids}"
    )
    print("  DO NOT USE THESE METRICS AS A BASELINE!")
    print(f"{border}\n")


# ---------------------------------------------------------------------------
# CLI & Execution
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evals.evaluate_rag",
        description="RAG evaluation runner for ai-cost-guardrail.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/datasets/rag_golden.jsonl"),
        help="Path to the golden JSONL dataset.",
    )
    parser.add_argument(
        "--mode",
        choices=["deterministic", "live"],
        default="deterministic",
        help=(
            "deterministic: fake embeddings/generation, no API calls. "
            "live: real Gemini API and PostgreSQL/pgvector."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "all"],
        default="all",
        help=(
            "Filter evaluation to a specific split ('dev', 'test', or 'all'). "
            "In live mode, only 'dev' is permitted until configuration is frozen."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of passages to retrieve per question (default: 5).",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=None,
        dest="request_delay",
        help=(
            "Pacing delay in seconds between questions. "
            f"Defaults to {DEFAULT_REQUEST_DELAY_DETERMINISTIC:.0f} in deterministic mode "
            f"and {DEFAULT_REQUEST_DELAY_LIVE:.0f} in live mode."
        ),
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        dest="retrieval_only",
        help="Embed query and retrieve top-k passages with similarity scores, skipping generation entirely.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        dest="resume",
        help="Resume from checkpoint or report. Refuses on config-fingerprint mismatch.",
    )
    parser.add_argument(
        "--only-ids",
        nargs="+",
        default=None,
        dest="only_ids",
        help="Evaluate only specified question IDs (space-separated or comma-separated).",
    )
    parser.add_argument(
        "--order",
        choices=["original", "unanswerable-first"],
        default="original",
        dest="order",
        help="Order of questions: 'original' or 'unanswerable-first'.",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        dest="skip_ingest",
        help="Skip corpus ingestion in live mode if database chunks match embedder fingerprint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        dest="checkpoint",
        help="Path for streaming checkpoint JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/results/latest.json"),
        help="Path to write the JSON report.",
    )
    parser.add_argument(
        "--tenant-id",
        type=int,
        default=None,
        dest="tenant_id",
        help="Tenant ID for live mode (defaults to dedicated evaluation tenant).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


async def _main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # 1. Enforce live-mode split guardrail: never evaluate test split in live mode
    if args.mode == "live" and args.split in ("test", "all"):
        logger.error(
            "Rule violation: live evaluation on the 'test' split is prohibited "
            "until configuration is frozen. Run live mode with --split dev."
        )
        return 1

    # 2. Determine pacing request delay
    if args.request_delay is not None:
        request_delay = float(args.request_delay)
    else:
        request_delay = (
            DEFAULT_REQUEST_DELAY_LIVE
            if args.mode == "live"
            else DEFAULT_REQUEST_DELAY_DETERMINISTIC
        )

    # 3. Load and validate dataset.
    logger.info("Loading dataset from %s", args.dataset)
    try:
        all_records = load_and_validate(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Dataset error: %s", exc)
        return 1

    # Filter by split if requested
    if args.split != "all":
        records = [r for r in all_records if r["split"] == args.split]
    else:
        records = all_records

    # Filter by --only-ids if requested
    if args.only_ids:
        target_ids: set[str] = set()
        for item in args.only_ids:
            for part in item.split(","):
                part_clean = part.strip()
                if part_clean:
                    target_ids.add(part_clean)
        records = [r for r in records if r["id"] in target_ids]
        logger.info(
            "Filtered to %d records using --only-ids: %s",
            len(records),
            sorted(target_ids),
        )

    # Order records if requested
    if args.order == "unanswerable-first":
        records = sorted(records, key=lambda r: 0 if not r["answerable"] else 1)
        logger.info("Ordered records unanswerable-first")

    logger.info(
        "Dataset loaded: %d records to evaluate (%d dev / %d test in full file)",
        len(records),
        sum(1 for r in all_records if r["split"] == "dev"),
        sum(1 for r in all_records if r["split"] == "test"),
    )

    # 4. Setup services and determine configuration headers
    service: RAGService | None = None
    uuid_to_doc_id: dict[str, str] | None = None
    eval_tenant_id: int = EVAL_TENANT_ID_DETERMINISTIC
    model_config: dict[str, Any] = {}

    dataset_sha = compute_file_sha256(args.dataset)
    git_hash = get_git_commit()

    if args.mode == "live":
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        from app.core.config import settings
        from app.rag.embeddings import GeminiEmbedder
        from app.rag.generation import (
            PROMPT_VERSION,
            GenerationService,
        )
        from app.rag.pipeline import RAGService

        logger.info("Initializing live mode services...")
        engine = create_engine(settings.database_url)
        db = Session(bind=engine)
        try:
            eval_tenant_id = ensure_eval_tenant(db, args.tenant_id)
            logger.info("Using dedicated evaluation tenant ID: %d", eval_tenant_id)

            embedder = GeminiEmbedder()
            generator = GenerationService()
            eval_retriever = EvalRetriever(db=db)
            service = RAGService(
                embedder=embedder,
                retriever=eval_retriever,
                generator=generator,
            )

            skip_done = False
            if args.skip_ingest:
                from app.services.models import Document, DocumentChunkRecord

                existing_count = (
                    db.query(DocumentChunkRecord)
                    .filter(
                        DocumentChunkRecord.tenant_id == eval_tenant_id,
                        DocumentChunkRecord.fingerprint == embedder.fingerprint,
                    )
                    .count()
                )
                if existing_count == CORPUS_CHUNK_COUNT:
                    logger.info(
                        "Found %d chunks matching embedder fingerprint %s; skipping ingestion",
                        existing_count,
                        embedder.fingerprint,
                    )
                    docs = (
                        db.query(Document)
                        .filter(Document.tenant_id == eval_tenant_id)
                        .all()
                    )
                    uuid_to_doc_id = {str(d.id): (d.source or "") for d in docs}
                    skip_done = True
                else:
                    logger.warning(
                        "--skip-ingest requested but found %d chunks (expected %d); proceeding with ingestion",
                        existing_count,
                        CORPUS_CHUNK_COUNT,
                    )

            if not skip_done:
                cleanup_evaluation_documents(db, eval_tenant_id)
                uuid_to_doc_id = ingest_evaluation_corpus(
                    db=db,
                    tenant_id=eval_tenant_id,
                    embedder=embedder,
                    corpus=SYNTHETIC_CORPUS,
                )
        except Exception as exc:
            db.close()
            engine.dispose()
            logger.error("Live mode setup failed: %s", exc)
            return 1

        generator_name = (
            generator.config.model
            if not args.retrieval_only
            else "none (retrieval-only)"
        )
        temp = generator.config.temperature if not args.retrieval_only else 0.0
        prompt_ver = PROMPT_VERSION if not args.retrieval_only else "none"

        model_config = {
            "embedder": embedder.config.model,
            "embedder_fingerprint": embedder.fingerprint,
            "generator": generator_name,
            "tenant_id": eval_tenant_id,
            "prompt_version": prompt_ver,
            "temperature": temp,
        }
    else:
        generator_name = (
            "FakeGenerator" if not args.retrieval_only else "none (retrieval-only)"
        )
        temp = 0.0
        prompt_ver = "eval-fake-v1" if not args.retrieval_only else "none"

        model_config = {
            "embedder": "FakeEmbedder(dimensions=8)",
            "generator": generator_name,
            "prompt_version": prompt_ver,
            "temperature": temp,
        }

    current_config_fingerprint = compute_config_fingerprint(
        dataset_sha256=dataset_sha,
        corpus_version=CORPUS_VERSION,
        prompt_version=prompt_ver,
        generator_model=generator_name,
        temperature=temp,
        top_k=args.top_k,
        mode=args.mode,
    )

    # 5. Handle --resume and Checkpoint setup
    completed_map: dict[str, QuestionResult] = {}
    checkpoint_path = args.checkpoint or (
        args.output.parent / f"{args.output.stem}.checkpoint.jsonl"
    )

    if args.resume:
        resume_path: Path
        if isinstance(args.resume, str):
            resume_path = Path(args.resume)
        else:
            resume_path = checkpoint_path
            if not resume_path.exists() and args.output.exists():
                resume_path = args.output

        if not resume_path.exists():
            logger.error("Cannot resume: resume file %s not found", resume_path)
            return 1

        chk_header, chk_questions = read_checkpoint(resume_path)
        if not chk_header or not chk_header.get("config_fingerprint"):
            logger.error(
                "Cannot resume: resume file %s is missing config fingerprint header",
                resume_path,
            )
            return 1

        if chk_header.get("config_fingerprint") != current_config_fingerprint:
            logger.error(
                "Refusing to resume: configuration fingerprint mismatch.\n"
                "  Checkpoint: %s\n"
                "  Current:    %s",
                chk_header.get("config_fingerprint"),
                current_config_fingerprint,
            )
            return 1

        logger.info(
            "Configuration fingerprint verified: %s", current_config_fingerprint
        )
        for q_dict in chk_questions:
            if not q_dict.get("failed") and q_dict.get("generation_status") != "failed":
                qr_obj = QuestionResult(
                    question_id=q_dict["id"],
                    split=q_dict["split"],
                    answerable=q_dict["answerable"],
                    generation_status=q_dict["generation_status"],
                    recall_at_1=q_dict.get("recall_at_1"),
                    recall_at_3=q_dict.get("recall_at_3"),
                    recall_at_5=q_dict.get("recall_at_5"),
                    reciprocal_rank=q_dict.get("reciprocal_rank"),
                    recall_at_k=q_dict.get("recall_at_k"),
                    latency_ms=q_dict.get("latency_ms"),
                    total_cost=Decimal(q_dict["total_cost"])
                    if q_dict.get("total_cost")
                    else None,
                    embedding_tokens=q_dict.get("embedding_tokens"),
                    error=q_dict.get("error"),
                    failed=False,
                    retries=q_dict.get("retries", 0),
                    answer=q_dict.get("answer"),
                    citations=q_dict.get("citations", []),
                    retrieved_chunks=q_dict.get("retrieved", []),
                    citation_validity=q_dict.get("citation_validity"),
                )
                completed_map[qr_obj.question_id] = qr_obj

        logger.info(
            "Resuming: %d questions already completed from %s",
            len(completed_map),
            resume_path,
        )

    # Initialize checkpoint file with header if creating fresh file
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
        append_checkpoint_record(
            checkpoint_path,
            {
                "type": "header",
                "config_fingerprint": current_config_fingerprint,
                "dataset_sha256": dataset_sha,
                "corpus_version": CORPUS_VERSION,
                "prompt_version": prompt_ver,
                "generator_model": generator_name,
                "temperature": temp,
                "top_k": args.top_k,
                "mode": args.mode,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            },
        )

    # 6. Run evaluation.
    logger.info(
        "Running evaluation in %s mode (top_k=%d, split=%s, retrieval_only=%s, request_delay=%.1fs)",
        args.mode,
        args.top_k,
        args.split,
        args.retrieval_only,
        request_delay,
    )

    run_state: dict[str, Any] = {}
    try:
        question_results = await run_evaluation(
            records=records,
            mode=args.mode,
            top_k=args.top_k,
            tenant_id=eval_tenant_id,
            service=service,
            uuid_to_doc_id=uuid_to_doc_id,
            request_delay=request_delay,
            retrieval_only=args.retrieval_only,
            checkpoint_path=checkpoint_path,
            completed_results=completed_map,
            run_state=run_state,
        )
    except Exception as exc:
        logger.error("Evaluation failed with unexpected error: %s", exc)
        return 1

    pending_ids = run_state.get("pending_ids", [])
    quota_error = run_state.get("quota_error")
    failed_question_ids = [qr.question_id for qr in question_results if qr.failed]
    complete = (len(failed_question_ids) == 0) and (len(pending_ids) == 0)

    # 7. Aggregate metrics.
    eval_splits = sorted(
        {
            r["split"]
            for r in records
            if any(qr.question_id == r["id"] for qr in question_results)
        }
    )
    split_metrics: dict[str, SplitMetrics] = {}
    for split in eval_splits:
        split_metrics[split] = compute_split_metrics(
            question_results,
            split=split,
            top_k=args.top_k,
            corpus_size=CORPUS_CHUNK_COUNT,
        )

    # 8. Print summary.
    _print_summary(
        split_metrics,
        mode=args.mode,
        failed_question_ids=failed_question_ids + pending_ids,
    )

    # 9. Write report.
    write_report(
        output_path=args.output,
        dataset_path=args.dataset,
        dataset_size=len(records),
        mode=args.mode,
        model_config=model_config,
        top_k=args.top_k,
        question_results=question_results,
        split_metrics=split_metrics,
        git_commit=git_hash,
        dataset_sha256=dataset_sha,
        corpus_version=CORPUS_VERSION,
        prompt_version=prompt_ver,
        generator_model=generator_name,
        temperature=temp,
        pending_question_ids=pending_ids,
        quota_error=quota_error,
        complete=complete,
    )
    logger.info("Report written to %s", args.output)

    # 10. Exit code: 2 if any questions failed or pending.
    if not complete:
        _print_incomplete_warning(failed_question_ids + pending_ids)
        return 2
    return 0


def _print_summary(
    split_metrics: dict[str, SplitMetrics],
    mode: str,
    failed_question_ids: list[str] | None = None,
) -> None:
    print("\n" + "=" * 68)
    print(f"  RAG EVALUATION SUMMARY  (mode={mode})")
    print("=" * 68)
    for split, m in split_metrics.items():
        print(f"\n  [{split.upper()}]")
        print(f"    Total questions   : {m.total}")
        print(f"    Answerable        : {m.answerable_count}")
        print(f"    Unanswerable      : {m.unanswerable_count}")
        print(f"    Failed            : {m.failed_count}")

        # Ranking metrics with analytic random-chance references
        def _fmt(val: float | None, ref: float | None) -> str:
            if val is None:
                return "N/A"
            ref_str = f"{ref:.4f}" if ref is not None else "N/A"
            return f"{val:.4f}  (random-chance reference: {ref_str})"

        print(
            f"    Recall@1 (macro)  : {_fmt(m.recall_at_1_mean, m.random_chance_recall_at_1)}"
        )
        print(
            f"    Recall@3 (macro)  : {_fmt(m.recall_at_3_mean, m.random_chance_recall_at_3)}"
        )
        print(
            f"    Recall@5 (macro)  : {_fmt(m.recall_at_5_mean, m.random_chance_recall_at_5)}"
        )
        if m.mrr_mean is not None:
            print(f"    MRR (macro)       : {m.mrr_mean:.4f}")
        else:
            print("    MRR (macro)       : N/A")

        print(
            f"    Correct abstentions / unanswerable  : "
            f"{m.correct_abstentions}/{m.unanswerable_count}"
        )
        print(
            f"    Unsupported answers / unanswerable  : "
            f"{m.unsupported_answers}/{m.unanswerable_count}"
        )
        print(
            f"    Unnecessary abstentions / answerable: "
            f"{m.unnecessary_abstentions}/{m.answerable_count}"
        )
        if m.latency_ms_mean is not None:
            print(
                f"    Latency mean/p50/p95 (ms): "
                f"{m.latency_ms_mean:.0f} / {m.latency_ms_p50:.0f} / "
                f"{m.latency_ms_p95:.0f}"
            )
    print()
    if mode == "deterministic":
        print(
            "  NOTE: deterministic mode validates the harness only.\n"
            "  Retrieval and generation quality numbers are NOT meaningful.\n"
        )
    print("=" * 68 + "\n")


def main() -> None:
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
