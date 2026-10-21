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

    def embed_query_with_usage(
        self,
        query: str,
        *,
        precomputed_tokens: int | None = None,
    ) -> EmbedResult: ...

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
    def build_prompt(
        self,
        query: str,
        passages: Sequence[ContextPassage],
    ) -> str: ...

    async def count_prompt_tokens(self, prompt: str, /) -> int: ...

    async def generate(
        self,
        query: str,
        passages: Sequence[ContextPassage],
        *,
        prompt: str | None = None,
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
            query_tokens: int | None = None
            embedding_tokens: int | None = None

            if budget is not None:
                # Count before the embedding request so the reservation
                # is committed before any embedding provider call.
                query_tokens = await asyncio.to_thread(
                    self.embedder.count_tokens,
                    query,
                )
                budget.reserve_embedding(query_tokens)

                embed_started = time.monotonic()
                embed_result = await asyncio.to_thread(
                    self.embedder.embed_query_with_usage,
                    query,
                    precomputed_tokens=query_tokens,
                )
                embed_ms = int((time.monotonic() - embed_started) * 1000)
                vector = embed_result.vector
                embedding_tokens = query_tokens
                budget.settle_embedding(query_tokens)
            else:
                embed_started = time.monotonic()
                if hasattr(self.embedder, "embed_query_with_usage"):
                    embed_result = await asyncio.to_thread(
                        self.embedder.embed_query_with_usage,
                        query,
                    )
                    vector = embed_result.vector
                    embedding_tokens = embed_result.tokens
                else:
                    vector = await asyncio.to_thread(
                        self.embedder.embed_query,
                        query,
                    )
                    embedding_tokens = None
                embed_ms = int((time.monotonic() - embed_started) * 1000)

            # Retrieve only after embedding has been settled.
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
            prompt: str | None = None

            # Generation cost depends on the prompt that includes
            # retrieved evidence, so reserve it only after retrieval.
            if budget is not None and passages:
                if not callable(getattr(self.generator, "count_prompt_tokens", None)):
                    raise GenerationProviderError(
                        "GenerationService does not expose count_prompt_tokens; "
                        "cannot reserve generation cost without a real token count"
                    )

                if not callable(getattr(self.generator, "build_prompt", None)):
                    raise GenerationProviderError(
                        "GenerationService does not expose build_prompt; "
                        "cannot construct prompt for budget accounting"
                    )

                prompt = self.generator.build_prompt(
                    query,
                    passages,
                )

                prompt_tokens = await self.generator.count_prompt_tokens(prompt)

                config = getattr(self.generator, "config", None)

                model = getattr(
                    config,
                    "model",
                    "gemini-3-flash-preview",
                )

                max_output_tokens = getattr(
                    config,
                    "max_output_tokens",
                    2000,
                )

                budget.reserve_generation(
                    prompt_tokens=prompt_tokens,
                    max_output_tokens=max_output_tokens,
                    model=model,
                )

                has_reserved_gen = True

            # Keep this call unconditional. For empty passages, the
            # generator should return "no_context" without an LLM call.
            generate_started = time.monotonic()

            if has_reserved_gen:
                generation = await self.generator.generate(
                    query,
                    passages,
                    prompt=prompt,
                )
            else:
                generation = await self.generator.generate(
                    query,
                    passages,
                )

            generate_ms = int((time.monotonic() - generate_started) * 1000)

            if budget is not None and has_reserved_gen:
                # Settle generation reservation using provider usage metadata.
                # Incomplete or missing usage fails closed: preserves the
                # reservation to protect tenant budget and raises GenerationProviderError.
                budget.settle_generation(generation.provider)

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
