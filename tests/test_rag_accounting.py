from decimal import Decimal
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.v1.rag import get_rag_service
from app.core.model_registry import DEFAULT_GENERATION_MODEL
from app.main import app
from app.rag.accounting import RAGBudget
from app.rag.embeddings import (
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    EmbedResult,
)
from app.rag.generation import (
    ContextPassage,
    GenerationProviderError,
    GenerationResult,
    GenerationTimeoutError,
    ProviderMeta,
)
from app.rag.pipeline import RAGAnswer, RAGService
from app.services.database import get_db
from app.services.models import Tenant, User
from app.services.usage import BudgetExceededError

client = TestClient(app)


class StubEmbedder:
    def __init__(self, vector=None, tokens=10):
        self.vector = vector or [0.1] * 768
        self.tokens = tokens
        self.calls = []

    @property
    def fingerprint(self) -> str:
        return "stub/embed/768"

    def embed_query(self, query: str) -> list[float]:
        self.calls.append(query)
        return self.vector

    def embed_query_with_usage(self, query: str) -> EmbedResult:
        self.calls.append(query)
        return EmbedResult(vector=self.vector, tokens=self.tokens)

    def count_tokens(self, text: str) -> int:
        return self.tokens


class StubRetriever:
    def __init__(self, passages=None):
        self.passages = (
            passages
            if passages is not None
            else [ContextPassage(ref=("doc-1", 1), text="Valid context passage text.")]
        )

    def search(self, **kwargs) -> list[ContextPassage]:
        return self.passages


class StubGenerator:
    def __init__(self, result=None, prompt_tokens=25):
        self.result = result
        self.prompt_tokens = prompt_tokens
        self.config = type(
            "Config",
            (),
            {
                "model": DEFAULT_GENERATION_MODEL,
                "max_output_tokens": 2000,
            },
        )()

    async def generate(self, query: str, passages) -> GenerationResult:
        if self.result is not None:
            return self.result
        return GenerationResult(
            status="answered",
            answer="Grounded answer based on evidence.",
            citations=[],
            passages_sent=[p.ref for p in passages],
            finish_reason="stop",
            invalid_citation_markers=0,
            prompt_version="test-v1",
            provider=ProviderMeta(
                provider="gemini",
                model=DEFAULT_GENERATION_MODEL,
                prompt_tokens=self.prompt_tokens,
                completion_tokens=15,
                latency_ms=50,
                thinking_tokens=5,
                total_tokens=45,
            ),
        )

    async def count_prompt_tokens(self, query: str, passages) -> int:
        return self.prompt_tokens


@pytest.fixture
def test_tenant(db):
    tenant = Tenant(
        name="RAG Accounting Tenant",
        monthly_budget=Decimal("50.000000"),
        current_spend=Decimal("0.000000"),
    )
    db.add(tenant)
    db.commit()
    db.refresh(tenant)
    return tenant


@pytest.fixture
def test_user(db, test_tenant):
    user = User(
        tenant_id=test_tenant.id,
        email="rag-accounting@example.com",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# RAGBudget Unit Tests
# ---------------------------------------------------------------------------


def test_rag_budget_lifecycle_embed_and_generate(db, test_tenant, test_user):
    budget = RAGBudget(
        db=db,
        tenant_id=test_tenant.id,
        user_id=test_user.id,
        request_id=str(uuid4()),
    )

    # 1. Reserve embedding
    budget.reserve_embedding(query_tokens=20)
    db.refresh(test_tenant)
    assert budget.embed_record is not None
    assert budget.embed_record.reserved_cost > Decimal(0)

    # 2. Settle embedding
    budget.settle_embedding(tokens=20)
    db.refresh(test_tenant)
    assert budget.embed_record.status == "completed"
    assert budget.embed_record.actual_cost > Decimal(0)
    actual_embed = budget.embed_record.actual_cost

    # 3. Reserve generation
    budget.reserve_generation(
        prompt_tokens=50,
        max_output_tokens=2000,
        model=DEFAULT_GENERATION_MODEL,
    )
    db.refresh(test_tenant)
    assert budget.generate_record is not None
    assert budget.generate_record.status == "reserved"

    # 4. Settle generation
    meta = ProviderMeta(
        provider="gemini",
        model=DEFAULT_GENERATION_MODEL,
        prompt_tokens=50,
        completion_tokens=20,
        latency_ms=100,
        thinking_tokens=10,
        total_tokens=80,
    )
    budget.settle_generation(meta)
    db.refresh(test_tenant)
    assert budget.generate_record.status == "completed"
    actual_gen = budget.generate_record.actual_cost
    assert actual_gen > Decimal(0)

    # 5. Total cost property
    assert budget.total_cost == actual_embed + actual_gen
    assert test_tenant.current_spend == budget.total_cost


def test_rag_budget_release_all_releases_reserved_only(db, test_tenant, test_user):
    budget = RAGBudget(
        db=db,
        tenant_id=test_tenant.id,
        user_id=test_user.id,
        request_id=str(uuid4()),
    )

    budget.reserve_embedding(query_tokens=15)
    budget.settle_embedding(tokens=15)
    actual_embed = budget.embed_record.actual_cost

    budget.reserve_generation(
        prompt_tokens=40,
        max_output_tokens=2000,
        model=DEFAULT_GENERATION_MODEL,
    )
    assert budget.generate_record.status == "reserved"

    # Release all: should only release the reserved generate record
    budget.release_all()
    db.refresh(test_tenant)

    assert budget.embed_record.status == "completed"
    assert budget.generate_record.status == "failed"
    assert budget.generate_record.actual_cost == Decimal(0)
    # Tenant current_spend should reflect only the settled embedding
    assert test_tenant.current_spend == actual_embed
    assert budget.total_cost == actual_embed


def test_rag_budget_exceeded_error_embedding(db, test_tenant, test_user):
    test_tenant.current_spend = test_tenant.monthly_budget
    db.commit()

    budget = RAGBudget(
        db=db,
        tenant_id=test_tenant.id,
        user_id=test_user.id,
        request_id=str(uuid4()),
    )

    with pytest.raises(BudgetExceededError):
        budget.reserve_embedding(query_tokens=100)

    assert budget.embed_record is None
    db.refresh(test_tenant)
    assert test_tenant.current_spend == test_tenant.monthly_budget


# ---------------------------------------------------------------------------
# Pipeline Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_with_budget_successful_settlement(db, test_tenant, test_user):
    embedder = StubEmbedder(tokens=12)
    retriever = StubRetriever()
    generator = StubGenerator()

    service = RAGService(
        embedder=embedder,
        retriever=retriever,
        generator=generator,
    )
    budget = RAGBudget(
        db=db,
        tenant_id=test_tenant.id,
        user_id=test_user.id,
        request_id=str(uuid4()),
    )

    answer = await service.answer(
        tenant_id=test_tenant.id,
        query="What is the refund policy?",
        budget=budget,
    )

    assert isinstance(answer, RAGAnswer)
    assert answer.embedding_tokens == 12
    assert answer.total_cost > Decimal(0)
    assert budget.embed_record.status == "completed"
    assert budget.generate_record.status == "completed"
    db.refresh(test_tenant)
    assert test_tenant.current_spend == answer.total_cost


@pytest.mark.asyncio
async def test_pipeline_with_budget_no_context_skips_generation_reservation(
    db, test_tenant, test_user
):
    embedder = StubEmbedder(tokens=10)
    retriever = StubRetriever(passages=[])  # Empty retrieval!
    generator = StubGenerator(
        result=GenerationResult(
            status="no_context",
            answer=None,
            citations=[],
            passages_sent=[],
            finish_reason="other",
            invalid_citation_markers=0,
            prompt_version="test-v1",
            provider=None,
        )
    )

    service = RAGService(
        embedder=embedder,
        retriever=retriever,
        generator=generator,
    )
    budget = RAGBudget(
        db=db,
        tenant_id=test_tenant.id,
        user_id=test_user.id,
        request_id=str(uuid4()),
    )

    answer = await service.answer(
        tenant_id=test_tenant.id,
        query="What is the refund policy?",
        budget=budget,
    )

    assert answer.generation.status == "no_context"
    assert answer.embedding_tokens == 10
    # No generate reservation was made
    assert budget.generate_record is None
    assert budget.embed_record.status == "completed"
    assert answer.total_cost == budget.embed_record.actual_cost
    db.refresh(test_tenant)
    assert test_tenant.current_spend == budget.embed_record.actual_cost


@pytest.mark.asyncio
async def test_pipeline_with_budget_blocked_with_meta_settles(
    db, test_tenant, test_user
):
    embedder = StubEmbedder(tokens=8)
    retriever = StubRetriever()
    generator = StubGenerator(
        result=GenerationResult(
            status="blocked",
            answer=None,
            citations=[],
            passages_sent=[("doc-1", 1)],
            finish_reason="safety",
            invalid_citation_markers=0,
            prompt_version="test-v1",
            provider=ProviderMeta(
                provider="gemini",
                model=DEFAULT_GENERATION_MODEL,
                prompt_tokens=20,
                completion_tokens=0,
                latency_ms=30,
                thinking_tokens=0,
                total_tokens=20,
            ),
        )
    )

    service = RAGService(
        embedder=embedder,
        retriever=retriever,
        generator=generator,
    )
    budget = RAGBudget(
        db=db,
        tenant_id=test_tenant.id,
        user_id=test_user.id,
        request_id=str(uuid4()),
    )

    answer = await service.answer(
        tenant_id=test_tenant.id,
        query="Blocked query",
        budget=budget,
    )

    assert answer.generation.status == "blocked"
    assert budget.generate_record.status == "completed"
    assert answer.total_cost == budget.total_cost


@pytest.mark.asyncio
async def test_pipeline_with_budget_blocked_without_meta_releases(
    db, test_tenant, test_user
):
    embedder = StubEmbedder(tokens=8)
    retriever = StubRetriever()
    generator = StubGenerator(
        result=GenerationResult(
            status="blocked",
            answer=None,
            citations=[],
            passages_sent=[("doc-1", 1)],
            finish_reason="safety",
            invalid_citation_markers=0,
            prompt_version="test-v1",
            provider=None,  # No provider metadata!
        )
    )

    service = RAGService(
        embedder=embedder,
        retriever=retriever,
        generator=generator,
    )
    budget = RAGBudget(
        db=db,
        tenant_id=test_tenant.id,
        user_id=test_user.id,
        request_id=str(uuid4()),
    )

    answer = await service.answer(
        tenant_id=test_tenant.id,
        query="Blocked query without meta",
        budget=budget,
    )

    assert answer.generation.status == "blocked"
    assert budget.embed_record.status == "completed"
    assert budget.generate_record.status == "failed"
    assert budget.generate_record.actual_cost == Decimal(0)
    assert answer.total_cost == budget.embed_record.actual_cost


@pytest.mark.asyncio
async def test_pipeline_with_budget_generation_error_releases_all(
    db, test_tenant, test_user
):
    embedder = StubEmbedder(tokens=8)
    retriever = StubRetriever()

    class FailingGenerator(StubGenerator):
        async def generate(self, query, passages):
            raise GenerationProviderError("LLM exploded")

    service = RAGService(
        embedder=embedder,
        retriever=retriever,
        generator=FailingGenerator(),
    )
    budget = RAGBudget(
        db=db,
        tenant_id=test_tenant.id,
        user_id=test_user.id,
        request_id=str(uuid4()),
    )

    with pytest.raises(GenerationProviderError):
        await service.answer(
            tenant_id=test_tenant.id,
            query="Query that will fail LLM",
            budget=budget,
        )

    assert budget.embed_record.status == "completed"
    assert budget.generate_record.status == "failed"
    db.refresh(test_tenant)
    assert test_tenant.current_spend == budget.embed_record.actual_cost


@pytest.mark.asyncio
async def test_pipeline_with_budget_missing_count_prompt_tokens_raises(
    db, test_tenant, test_user
):
    embedder = StubEmbedder(tokens=8)
    retriever = StubRetriever()

    class GeneratorWithoutCountTokens:
        async def generate(self, query, passages):
            return GenerationResult(
                status="answered",
                answer="Answer",
                citations=[],
                passages_sent=[p.ref for p in passages],
                finish_reason="stop",
                invalid_citation_markers=0,
                prompt_version="test-v1",
                provider=None,
            )

    service = RAGService(
        embedder=embedder,
        retriever=retriever,
        generator=GeneratorWithoutCountTokens(),
    )
    budget = RAGBudget(
        db=db,
        tenant_id=test_tenant.id,
        user_id=test_user.id,
        request_id=str(uuid4()),
    )

    with pytest.raises(
        GenerationProviderError,
        match="GenerationService does not expose count_prompt_tokens",
    ):
        await service.answer(
            tenant_id=test_tenant.id,
            query="Query with missing count_tokens",
            budget=budget,
        )

    assert budget.embed_record.status == "completed"
    assert budget.generate_record is None
    db.refresh(test_tenant)
    assert test_tenant.current_spend == budget.embed_record.actual_cost


# ---------------------------------------------------------------------------
# Route Error Mapping & Response Tests
# ---------------------------------------------------------------------------


def test_rag_route_budget_exceeded_returns_400(db, test_tenant, test_user):
    test_tenant.current_spend = test_tenant.monthly_budget
    db.commit()

    def override_test_db():
        yield db

    stub_service = RAGService(
        embedder=StubEmbedder(),
        retriever=StubRetriever(),
        generator=StubGenerator(),
    )

    app.dependency_overrides[get_db] = override_test_db
    app.dependency_overrides[get_rag_service] = lambda: stub_service

    try:
        with patch(
            "app.api.dependencies.verify_access_token",
            return_value={
                "tenant_id": str(test_tenant.id),
                "user_id": str(test_user.id),
            },
        ):
            response = client.post(
                "/api/v1/rag/query",
                headers={"Authorization": "Bearer token"},
                json={"query": "Valid query but budget full", "top_k": 5},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert "exceed budget" in response.json()["detail"].lower()


def test_rag_route_rate_limit_error_returns_429(db, test_tenant, test_user):
    def override_test_db():
        yield db

    class FailingService:
        async def answer(self, **kwargs):
            raise EmbeddingRateLimitError("Quota exceeded")

    app.dependency_overrides[get_db] = override_test_db
    app.dependency_overrides[get_rag_service] = lambda: FailingService()

    try:
        with patch(
            "app.api.dependencies.verify_access_token",
            return_value={
                "tenant_id": str(test_tenant.id),
                "user_id": str(test_user.id),
            },
        ):
            response = client.post(
                "/api/v1/rag/query",
                headers={"Authorization": "Bearer token"},
                json={"query": "Valid query", "top_k": 5},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 429


def test_rag_route_timeout_error_returns_504(db, test_tenant, test_user):
    def override_test_db():
        yield db

    class TimingOutService:
        async def answer(self, **kwargs):
            raise GenerationTimeoutError("Timed out")

    app.dependency_overrides[get_db] = override_test_db
    app.dependency_overrides[get_rag_service] = lambda: TimingOutService()

    try:
        with patch(
            "app.api.dependencies.verify_access_token",
            return_value={
                "tenant_id": str(test_tenant.id),
                "user_id": str(test_user.id),
            },
        ):
            response = client.post(
                "/api/v1/rag/query",
                headers={"Authorization": "Bearer token"},
                json={"query": "Valid query", "top_k": 5},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 504


def test_rag_route_provider_error_returns_502(db, test_tenant, test_user):
    def override_test_db():
        yield db

    class ProviderFailingService:
        async def answer(self, **kwargs):
            raise EmbeddingProviderError("Upstream failure")

    app.dependency_overrides[get_db] = override_test_db
    app.dependency_overrides[get_rag_service] = lambda: ProviderFailingService()

    try:
        with patch(
            "app.api.dependencies.verify_access_token",
            return_value={
                "tenant_id": str(test_tenant.id),
                "user_id": str(test_user.id),
            },
        ):
            response = client.post(
                "/api/v1/rag/query",
                headers={"Authorization": "Bearer token"},
                json={"query": "Valid query", "top_k": 5},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502


def test_rag_route_success_returns_request_id_and_total_cost(
    db, test_tenant, test_user
):
    def override_test_db():
        yield db

    stub_service = RAGService(
        embedder=StubEmbedder(),
        retriever=StubRetriever(),
        generator=StubGenerator(),
    )

    app.dependency_overrides[get_db] = override_test_db
    app.dependency_overrides[get_rag_service] = lambda: stub_service

    try:
        with patch(
            "app.api.dependencies.verify_access_token",
            return_value={
                "tenant_id": str(test_tenant.id),
                "user_id": str(test_user.id),
            },
        ):
            response = client.post(
                "/api/v1/rag/query",
                headers={"Authorization": "Bearer token"},
                json={"query": "What is the policy?", "top_k": 5},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "answered"
    assert "request_id" in data
    assert data["request_id"] is not None
    assert "total_cost" in data
    # In JSON, Pydantic encodes Decimal as string
    assert Decimal(str(data["total_cost"])) > Decimal(0)
