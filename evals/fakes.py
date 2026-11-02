"""
Fake retriever and fake generator for deterministic evaluation.

These classes implement the Retriever and Generator protocols defined in
``app.rag.pipeline`` without touching any database or external API.

FakeCorpusRetriever
-------------------
Holds an in-memory corpus keyed by ``(document_id, chunk_index)``.  Retrieval
is performed by cosine similarity between the query embedding and pre-computed
passage embeddings.  Because ``FakeEmbedder`` is deterministic (SHA-256 of
text), the retrieval results are also deterministic.

FakeGenerator
-------------
Returns a canned ``GenerationResult`` without calling any LLM.  Its
``generation_status`` is determined by whether any passages were supplied:

* Empty passages  → ``"no_context"``   (mirrors ``GenerationService`` behaviour)
* Non-empty       → ``"answered"``

This mirrors the exact short-circuit logic in ``GenerationService.generate()``.

Why this matters for abstention testing
----------------------------------------
The pipeline's only abstention mechanism is the empty-passages short-circuit
in ``GenerationService``.  When the retriever finds no matching passages for
an unanswerable question, the pipeline produces ``status="no_context"``.
In deterministic mode, the ``FakeCorpusRetriever`` deliberately returns an
empty list for unanswerable questions (which have no matching corpus chunks),
exercising this path without any LLM call.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.embeddings import FakeEmbedder
from app.rag.generation import (
    Citation,
    ContextPassage,
    GenerationResult,
    ProviderMeta,
)

PROMPT_VERSION = "eval-fake-v1"
FAKE_EMBEDDER_DIMS = 8  # matches FakeEmbedder default


# ---------------------------------------------------------------------------
# Corpus builder helpers
# ---------------------------------------------------------------------------


def build_passages_from_corpus(
    corpus: dict[str, list[str]],
) -> dict[tuple[str, int], ContextPassage]:
    """
    Convert the synthetic corpus dict into a flat mapping of
    (document_id, chunk_index) → ContextPassage.
    """
    passages: dict[tuple[str, int], ContextPassage] = {}
    for doc_id, chunks in corpus.items():
        for chunk_index, text in enumerate(chunks):
            key = (doc_id, chunk_index)
            passages[key] = ContextPassage(
                ref=key,
                text=text,
                source=doc_id,
                page=chunk_index,
            )
    return passages


# ---------------------------------------------------------------------------
# FakeCorpusRetriever
# ---------------------------------------------------------------------------


class FakeCorpusRetriever:
    """
    In-memory retriever using FakeEmbedder cosine similarity.

    This retriever satisfies the ``Retriever`` protocol from
    ``app.rag.pipeline`` without touching any database.  It is intended
    exclusively for deterministic evaluation mode.
    """

    def __init__(
        self,
        passages: dict[tuple[str, int], ContextPassage],
        embedder: FakeEmbedder | None = None,
    ) -> None:
        self._passages = passages
        self._embedder = embedder or FakeEmbedder(dimensions=FAKE_EMBEDDER_DIMS)
        self.last_search_results: list[tuple[ContextPassage, float]] = []

        # Pre-compute passage embeddings once.
        self._passage_embeddings: dict[tuple[str, int], list[float]] = {}
        for key, passage in self._passages.items():
            self._passage_embeddings[key] = self._embedder._embed_text(passage.text)

    def search_with_scores(
        self,
        *,
        tenant_id: int,
        vector: Sequence[float],
        k: int,
        fingerprint: str,
        max_distance: float | None = None,
    ) -> list[tuple[ContextPassage, float]]:
        """Return the top-k passages with similarity scores by cosine similarity to *vector*."""
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if not vector:
            raise ValueError("vector must not be empty")

        query_vec = list(vector)

        scored: list[tuple[float, tuple[str, int]]] = []
        for key, emb in self._passage_embeddings.items():
            sim = _cosine_similarity(query_vec, emb)
            scored.append((sim, key))

        # Sort descending by similarity (cosine distance = 1 - similarity).
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[tuple[ContextPassage, float]] = []
        for sim, key in scored[:k]:
            if max_distance is not None:
                distance = 1.0 - sim
                if distance > max_distance:
                    continue
            results.append((self._passages[key], sim))

        self.last_search_results = results
        return results

    def search(
        self,
        *,
        tenant_id: int,
        vector: Sequence[float],
        k: int,
        fingerprint: str,
        max_distance: float | None = None,
    ) -> list[ContextPassage]:
        """Return the top-k passages by cosine similarity to *vector*."""
        scored = self.search_with_scores(
            tenant_id=tenant_id,
            vector=vector,
            k=k,
            fingerprint=fingerprint,
            max_distance=max_distance,
        )
        return [p for p, _ in scored]


# ---------------------------------------------------------------------------
# FakeGenerator
# ---------------------------------------------------------------------------


@dataclass
class FakeGenerator:
    """
    Canned generator for deterministic evaluation.

    Mirrors the observable behaviour of ``GenerationService``:
    - Empty passages  → status="no_context"
    - Non-empty       → status="answered" with a template answer and citations
                        that reference the first passage.

    No LLM call is made.
    """

    answer_template: str = (
        "Deterministic fake answer for evaluation harness. "
        "No real language model was used. [1]"
    )
    latency_ms: int = 0

    def build_prompt(
        self,
        query: str,
        passages: Sequence[ContextPassage],
    ) -> str:
        parts = [f"[{i + 1}]\n{p.text.strip()}" for i, p in enumerate(passages)]
        evidence = "\n\n".join(parts)
        return f"EVIDENCE\n=======\n{evidence}\n\nQUESTION\n========\n{query.strip()}"

    async def count_prompt_tokens(self, prompt: str) -> int:
        return max(1, len(prompt.split()))

    async def generate(
        self,
        query: str,
        passages: Sequence[ContextPassage],
        *,
        prompt: str | None = None,
    ) -> GenerationResult:
        passages = list(passages)

        if not passages:
            return GenerationResult(
                status="no_context",
                answer=None,
                citations=[],
                passages_sent=[],
                finish_reason="other",
                invalid_citation_markers=0,
                prompt_version=PROMPT_VERSION,
                provider=None,
            )

        # For fake generation: cite the first passage.
        first = passages[0]
        citations = [
            Citation(
                ref=first.ref,
                source=first.source,
                page=first.page,
            )
        ]

        provider_meta = ProviderMeta(
            provider="fake",
            model="fake-generator",
            prompt_tokens=10,
            completion_tokens=20,
            latency_ms=self.latency_ms,
        )

        return GenerationResult(
            status="answered",
            answer=self.answer_template,
            citations=citations,
            passages_sent=[p.ref for p in passages],
            finish_reason="stop",
            invalid_citation_markers=0,
            prompt_version=PROMPT_VERSION,
            provider=provider_meta,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
