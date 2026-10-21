from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol

from app.rag.embeddings import EmbedResult
from app.rag.generation import (
    ContextPassage,
    GenerationProviderError,
    GenerationResult,
)

if TYPE_CHECKING:
    from app.rag.accounting import RAGBudget

MAX_QUERY_LENGTH = 4000
MIN_TOP_K = 1
MAX_TOP_K = 20


class Embedder(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def embed_query(self, query: str) -> list[float]: ...

    def embed_query_with_usage(self, query: str) -> EmbedResult: ...

    def count_tokens(self, text: str) -> int: ...


class Retriever(Protocol):
    def search(
        self,
        *,
        tenant_id: int,
        vector: Sequence[float],
        k: int,
        fingerprint: str,
        max_distance: float | None = None,
    ) -> list[ContextPassage]: ...


class Generator(Protocol):
    async def generate(
        self,
        query: str,
        passages: Sequence[ContextPassage],
    ) -> GenerationResult: ...


@dataclass(frozen=True)
class RAGAnswer:
    generation: GenerationResult
    retrieved: list[tuple[str, int]]
    timings_ms: dict[str, int]
    embedding_tokens: int | None
    total_cost: Decimal = Decimal(0)
    request_id: str | None = None


class RAGService:
    def __init__(
        self,
        embedder: Embedder,
        retriever: Retriever,
        generator: Generator,
    ) -> None:
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator

    async def answer(
        self,
        *,
        tenant_id: int,
        query: str,
        top_k: int = 5,
        budget: RAGBudget | None = None,
    ) -> RAGAnswer:
        self._validate_query(query)
        self._validate_top_k(top_k)

        try:
            if budget is not None:
                query_tokens = self.embedder.count_tokens(query)
                budget.reserve_embedding(query_tokens)

            embed_started = time.monotonic()
            if hasattr(self.embedder, "embed_query_with_usage"):
                embed_res = await asyncio.to_thread(
                    self.embedder.embed_query_with_usage,
                    query,
                )
                vector = embed_res.vector
                embedding_tokens = embed_res.tokens
            else:
                vector = await asyncio.to_thread(
                    self.embedder.embed_query,
                    query,
                )
                embedding_tokens = None
            embed_ms = int((time.monotonic() - embed_started) * 1000)

            if budget is not None:
                settle_tokens = (
                    embedding_tokens if embedding_tokens is not None else query_tokens
                )
                budget.settle_embedding(settle_tokens)

            retrieve_started = time.monotonic()
            passages = await asyncio.to_thread(
                self.retriever.search,
                tenant_id=tenant_id,
                vector=vector,
                k=top_k,
                fingerprint=self.embedder.fingerprint,
            )
            retrieve_ms = int((time.monotonic() - retrieve_started) * 1000)

            has_reserved_gen = False
            if budget is not None and passages:
                # Generation cost requires a real prompt-token count.
                # Never fabricate — if the count is unavailable, skip the
                # generation reservation entirely rather than guessing.
                if not hasattr(self.generator, "count_prompt_tokens"):
                    raise GenerationProviderError(
                        "GenerationService does not expose count_prompt_tokens; "
                        "cannot reserve generation cost without a real token count"
                    )
                prompt_tokens = await self.generator.count_prompt_tokens(
                    query, passages
                )
                model = getattr(
                    getattr(self.generator, "config", None),
                    "model",
                    "gemini-3-flash-preview",
                )
                max_output_tokens = getattr(
                    getattr(self.generator, "config", None),
                    "max_output_tokens",
                    2000,
                )
                budget.reserve_generation(
                    prompt_tokens=prompt_tokens,
                    max_output_tokens=max_output_tokens,
                    model=model,
                )
                has_reserved_gen = True

            generate_started = time.monotonic()
            generation = await self.generator.generate(
                query,
                passages,
            )
            generate_ms = int((time.monotonic() - generate_started) * 1000)

            if budget is not None and has_reserved_gen:
                if generation.status == "blocked" and generation.provider is None:
                    budget.release_all()
                elif generation.provider is not None:
                    budget.settle_generation(generation.provider)
                else:
                    budget.release_all()

            total_cost = budget.total_cost if budget is not None else Decimal(0)
            request_id = budget.request_id if budget is not None else None

            return RAGAnswer(
                generation=generation,
                retrieved=[passage.ref for passage in passages],
                timings_ms={
                    "embed": embed_ms,
                    "retrieve": retrieve_ms,
                    "generate": generate_ms,
                },
                embedding_tokens=embedding_tokens,
                total_cost=total_cost,
                request_id=request_id,
            )

        except Exception:
            if budget is not None:
                budget.release_all()
            raise

    @staticmethod
    def _validate_query(query: str) -> None:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query cannot be empty")

        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError(f"query cannot exceed {MAX_QUERY_LENGTH} characters")

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")

        if not MIN_TOP_K <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}")
