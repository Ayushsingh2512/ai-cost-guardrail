from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.rag.generation import (
    ContextPassage,
    GenerationResult,
)
from app.rag.pipeline import RAGService


FINGERPRINT = "fake/fake-embedding/8"


def make_generation_result(
    *,
    status: str = "answered",
) -> GenerationResult:
    return GenerationResult(
        status=status,
        answer="Grounded answer" if status == "answered" else None,
        citations=[],
        passages_sent=[],
        finish_reason="stop",
        invalid_citation_markers=0,
        prompt_version="test-v1",
        provider=None,
    )


@dataclass
class RecordingEmbedder:
    vector: list[float]
    calls: list[str]

    @property
    def fingerprint(self) -> str:
        return FINGERPRINT

    def embed_query(self, query: str) -> list[float]:
        self.calls.append(query)
        return self.vector


@dataclass
class RecordingRetriever:
    passages: list[ContextPassage]
    calls: list[dict]

    def search(
        self,
        *,
        tenant_id: int,
        vector,
        k: int,
        fingerprint: str,
        max_distance: float | None = None,
    ) -> list[ContextPassage]:
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "vector": list(vector),
                "k": k,
                "fingerprint": fingerprint,
                "max_distance": max_distance,
            }
        )
        return self.passages


@dataclass
class RecordingGenerator:
    result: GenerationResult
    calls: list[dict]

    async def generate(
        self,
        query: str,
        passages,
    ) -> GenerationResult:
        self.calls.append(
            {
                "query": query,
                "passages": list(passages),
            }
        )
        return self.result


def build_service():
    embedder = RecordingEmbedder(
        vector=[1.0, 2.0, 3.0],
        calls=[],
    )

    passages = [
        ContextPassage(
            ref=("doc-1", 0),
            text="First passage",
            source="test.pdf",
            page=1,
        ),
        ContextPassage(
            ref=("doc-1", 1),
            text="Second passage",
            source="test.pdf",
            page=2,
        ),
    ]

    retriever = RecordingRetriever(
        passages=passages,
        calls=[],
    )

    generator = RecordingGenerator(
        result=make_generation_result(),
        calls=[],
    )

    service = RAGService(
        embedder=embedder,
        retriever=retriever,
        generator=generator,
    )

    return service, embedder, retriever, generator


@pytest.mark.asyncio
async def test_pipeline_calls_collaborators_in_order():
    events: list[str] = []

    class Embedder:
        fingerprint = FINGERPRINT

        def embed_query(self, query: str) -> list[float]:
            events.append("embed")
            return [1.0, 2.0, 3.0]

    class Retriever:
        def search(
            self,
            *,
            tenant_id: int,
            vector,
            k: int,
            fingerprint: str,
            max_distance=None,
        ):
            events.append("retrieve")
            return [
                ContextPassage(
                    ref=("doc-1", 0),
                    text="Evidence",
                )
            ]

    class Generator:
        async def generate(self, query, passages):
            events.append("generate")
            return make_generation_result()

    service = RAGService(
        embedder=Embedder(),
        retriever=Retriever(),
        generator=Generator(),
    )

    await service.answer(
        tenant_id=1,
        query="What is the policy?",
        top_k=5,
    )

    assert events == ["embed", "retrieve", "generate"]


@pytest.mark.asyncio
async def test_pipeline_passes_embedding_vector_tenant_and_fingerprint():
    service, embedder, retriever, _ = build_service()

    await service.answer(
        tenant_id=42,
        query="What is the policy?",
        top_k=7,
    )

    assert retriever.calls == [
        {
            "tenant_id": 42,
            "vector": [1.0, 2.0, 3.0],
            "k": 7,
            "fingerprint": FINGERPRINT,
            "max_distance": None,
        }
    ]


@pytest.mark.asyncio
async def test_pipeline_passes_retrieved_passages_to_generator():
    service, _, retriever, generator = build_service()

    await service.answer(
        tenant_id=1,
        query="What is the policy?",
        top_k=5,
    )

    assert len(generator.calls) == 1
    assert generator.calls[0]["query"] == "What is the policy?"
    assert generator.calls[0]["passages"] == retriever.passages


@pytest.mark.asyncio
async def test_embedding_error_stops_pipeline():
    events: list[str] = []

    class Embedder:
        fingerprint = FINGERPRINT

        def embed_query(self, query: str) -> list[float]:
            events.append("embed")
            raise RuntimeError("embedding failed")

    class Retriever:
        def search(self, **kwargs):
            events.append("retrieve")
            return []

    class Generator:
        async def generate(self, query, passages):
            events.append("generate")
            return make_generation_result()

    service = RAGService(
        embedder=Embedder(),
        retriever=Retriever(),
        generator=Generator(),
    )

    with pytest.raises(RuntimeError, match="embedding failed"):
        await service.answer(
            tenant_id=1,
            query="question",
        )

    assert events == ["embed"]


@pytest.mark.asyncio
async def test_retrieval_error_stops_generation():
    events: list[str] = []

    class Embedder:
        fingerprint = FINGERPRINT

        def embed_query(self, query: str) -> list[float]:
            events.append("embed")
            return [1.0]

    class Retriever:
        def search(self, **kwargs):
            events.append("retrieve")
            raise RuntimeError("retrieval failed")

    class Generator:
        async def generate(self, query, passages):
            events.append("generate")
            return make_generation_result()

    service = RAGService(
        embedder=Embedder(),
        retriever=Retriever(),
        generator=Generator(),
    )

    with pytest.raises(RuntimeError, match="retrieval failed"):
        await service.answer(
            tenant_id=1,
            query="question",
        )

    assert events == ["embed", "retrieve"]


@pytest.mark.asyncio
async def test_empty_retrieval_reaches_generator():
    service, _, retriever, generator = build_service()
    retriever.passages = []

    generator.result = make_generation_result(
        status="no_context",
    )

    result = await service.answer(
        tenant_id=1,
        query="question",
    )

    assert result.generation.status == "no_context"
    assert result.retrieved == []
    assert len(generator.calls) == 1
    assert generator.calls[0]["passages"] == []


@pytest.mark.asyncio
async def test_invalid_query_is_rejected_before_collaborators():
    service, embedder, retriever, generator = build_service()

    with pytest.raises(ValueError, match="query cannot be empty"):
        await service.answer(
            tenant_id=1,
            query="   ",
        )

    assert embedder.calls == []
    assert retriever.calls == []
    assert generator.calls == []


@pytest.mark.asyncio
async def test_query_length_is_rejected_before_collaborators():
    service, embedder, retriever, generator = build_service()

    with pytest.raises(ValueError, match="cannot exceed"):
        await service.answer(
            tenant_id=1,
            query="x" * 4001,
        )

    assert embedder.calls == []
    assert retriever.calls == []
    assert generator.calls == []


@pytest.mark.asyncio
async def test_invalid_top_k_is_rejected_before_collaborators():
    service, embedder, retriever, generator = build_service()

    with pytest.raises(ValueError, match="between 1 and 20"):
        await service.answer(
            tenant_id=1,
            query="question",
            top_k=21,
        )

    assert embedder.calls == []
    assert retriever.calls == []
    assert generator.calls == []


@pytest.mark.asyncio
async def test_pipeline_returns_retrieved_refs_and_timings():
    service, _, _, _ = build_service()

    result = await service.answer(
        tenant_id=1,
        query="question",
        top_k=2,
    )

    assert result.retrieved == [
        ("doc-1", 0),
        ("doc-1", 1),
    ]

    assert set(result.timings_ms) == {
        "embed",
        "retrieve",
        "generate",
    }

    assert all(
        isinstance(value, int) and value >= 0
        for value in result.timings_ms.values()
    )

    assert result.embedding_tokens is None


@pytest.mark.asyncio
async def test_pipeline_preserves_generation_result():
    service, _, _, _ = build_service()

    result = await service.answer(
        tenant_id=1,
        query="question",
    )

    assert result.generation.status == "answered"
    assert result.generation.answer == "Grounded answer"
    assert result.generation.prompt_version == "test-v1"