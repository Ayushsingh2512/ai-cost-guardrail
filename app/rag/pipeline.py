from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.rag.generation import ContextPassage, GenerationResult


MAX_QUERY_LENGTH = 4000
MIN_TOP_K = 1
MAX_TOP_K = 20


class Embedder(Protocol):
    @property
    def fingerprint(self) -> str:
        ...

    def embed_query(self, query: str) -> list[float]:
        ...


class Retriever(Protocol):
    def search(
        self,
        *,
        tenant_id: int,
        vector: Sequence[float],
        k: int,
        fingerprint: str,
        max_distance: float | None = None,
    ) -> list[ContextPassage]:
        ...


class Generator(Protocol):
    async def generate(
        self,
        query: str,
        passages: Sequence[ContextPassage],
    ) -> GenerationResult:
        ...


@dataclass(frozen=True)
class RAGAnswer:
    generation: GenerationResult
    retrieved: list[tuple[str, int]]
    timings_ms: dict[str, int]
    embedding_tokens: int | None


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
    ) -> RAGAnswer:
        self._validate_query(query)
        self._validate_top_k(top_k)

        embed_started = time.monotonic()
        vector = await asyncio.to_thread(
            self.embedder.embed_query,
            query,
        )
        embed_ms = int((time.monotonic() - embed_started) * 1000)

        retrieve_started = time.monotonic()
        passages = await asyncio.to_thread(
            self.retriever.search,
            tenant_id=tenant_id,
            vector=vector,
            k=top_k,
            fingerprint=self.embedder.fingerprint,
        )
        retrieve_ms = int((time.monotonic() - retrieve_started) * 1000)

        generate_started = time.monotonic()
        generation = await self.generator.generate(
            query,
            passages,
        )
        generate_ms = int((time.monotonic() - generate_started) * 1000)

        return RAGAnswer(
            generation=generation,
            retrieved=[passage.ref for passage in passages],
            timings_ms={
                "embed": embed_ms,
                "retrieve": retrieve_ms,
                "generate": generate_ms,
            },
            embedding_tokens=None,
        )

    @staticmethod
    def _validate_query(query: str) -> None:
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        if not query.strip():
            raise ValueError("query cannot be empty")

        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError(
                f"query cannot exceed {MAX_QUERY_LENGTH} characters"
            )

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if not isinstance(top_k, int):
            raise ValueError("top_k must be an integer")

        if not MIN_TOP_K <= top_k <= MAX_TOP_K:
            raise ValueError(
                f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}"
            )