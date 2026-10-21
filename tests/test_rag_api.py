from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app.api.dependencies import get_redis_client
from app.api.v1.rag import get_rag_service
from app.core.model_registry import DEFAULT_GENERATION_MODEL
from app.main import app
from app.rag.embeddings import EmbedResult
from app.rag.generation import (
    Citation,
    ContextPassage,
    GenerationResult,
    GenerationTimeoutError,
    ProviderMeta,
)
from app.rag.pipeline import RAGService
from app.services.database import get_db
from app.services.guardrail import RateLimitExceeded
from app.services.models import Tenant, UsageRecord, User
from app.services.usage import usage_service

client = TestClient(app)


def test_rag_requires_authentication():
    response = client.post(
        "/api/v1/rag/query",
        json={
            "query": "What is the refund policy?",
            "top_k": 5,
        },
    )

    assert response.status_code == 401


def test_rag_rejects_invalid_token():
    response = client.post(
        "/api/v1/rag/query",
        headers={
            "Authorization": "Bearer definitely-not-a-valid-jwt",
        },
        json={
            "query": "What is the refund policy?",
            "top_k": 5,
        },
    )

    assert response.status_code == 401


def test_rag_rejects_token_without_tenant_id():
    with patch(
        "app.api.dependencies.verify_access_token",
        return_value={"user_id": "1"},
    ):
        response = client.post(
            "/api/v1/rag/query",
            headers={
                "Authorization": "Bearer valid-looking-token",
            },
            json={
                "query": "What is the refund policy?",
                "top_k": 5,
            },
        )

    assert response.status_code == 401


def test_rag_rejects_token_without_user_id():
    with patch(
        "app.api.dependencies.verify_access_token",
        return_value={"tenant_id": "1"},
    ):
        response = client.post(
            "/api/v1/rag/query",
            headers={
                "Authorization": "Bearer valid-looking-token",
            },
            json={
                "query": "What is the refund policy?",
                "top_k": 5,
            },
        )

    assert response.status_code == 401


def test_rag_returns_400_when_token_limit_guardrail_rejects():
    with (
        patch(
            "app.api.dependencies.verify_access_token",
            return_value={
                "tenant_id": "1",
                "user_id": "1",
            },
        ),
        patch(
            "app.api.dependencies.guardrail_service.check_token_limit",
            side_effect=ValueError("Token limit exceeded"),
        ),
    ):
        response = client.post(
            "/api/v1/rag/query",
            headers={
                "Authorization": "Bearer valid-looking-token",
            },
            json={
                "query": "What is the refund policy?",
                "top_k": 5,
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Token limit exceeded"


def test_rag_returns_429_when_rate_limit_exceeded():
    def fake_redis():
        return object()

    def fake_rate_limit(
        redis_client,
        tenant_id,
        limit=30,
        window_seconds=60,
    ):
        raise RateLimitExceeded("Rate limit exceeded")

    with (
        patch(
            "app.api.dependencies.verify_access_token",
            return_value={
                "tenant_id": "1",
                "user_id": "1",
            },
        ),
        patch(
            "app.api.dependencies.guardrail_service.check_rate_limit",
            side_effect=fake_rate_limit,
        ),
    ):
        app.dependency_overrides[get_redis_client] = fake_redis

        try:
            response = client.post(
                "/api/v1/rag/query",
                headers={
                    "Authorization": "Bearer valid-looking-token",
                },
                json={
                    "query": "What is the refund policy?",
                    "top_k": 5,
                },
            )
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 429


def test_rag_returns_503_when_rate_limiter_unavailable():
    def fake_redis():
        return object()

    def fake_rate_limit(
        redis_client,
        tenant_id,
        limit=30,
        window_seconds=60,
    ):
        raise RuntimeError("Rate limiting service is unavailable")

    with (
        patch(
            "app.api.dependencies.verify_access_token",
            return_value={
                "tenant_id": "1",
                "user_id": "1",
            },
        ),
        patch(
            "app.api.dependencies.guardrail_service.check_rate_limit",
            side_effect=fake_rate_limit,
        ),
    ):
        app.dependency_overrides[get_redis_client] = fake_redis

        try:
            response = client.post(
                "/api/v1/rag/query",
                headers={
                    "Authorization": "Bearer valid-looking-token",
                },
                json={
                    "query": "What is the refund policy?",
                    "top_k": 5,
                },
            )
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 503


def test_rag_rejects_invalid_top_k():
    with patch(
        "app.api.dependencies.verify_access_token",
        return_value={
            "tenant_id": "1",
            "user_id": "1",
        },
    ):
        response = client.post(
            "/api/v1/rag/query",
            headers={
                "Authorization": "Bearer valid-looking-token",
            },
            json={
                "query": "What is the refund policy?",
                "top_k": 21,
            },
        )

    assert response.status_code == 422


def test_rag_returns_404_when_tenant_not_found(db):
    def override_test_db():
        yield db

    app.dependency_overrides[get_db] = override_test_db

    try:
        with patch(
            "app.api.dependencies.verify_access_token",
            return_value={
                "tenant_id": "999999",
                "user_id": "1",
            },
        ):
            response = client.post(
                "/api/v1/rag/query",
                headers={
                    "Authorization": "Bearer valid-looking-token",
                },
                json={
                    "query": "What is the refund policy?",
                    "top_k": 5,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
    assert "Tenant 999999 not found" in response.json()["detail"]


def test_rag_successfully_maps_service_result(db):
    tenant = Tenant(
        name="RAG API Test Tenant",
        monthly_budget=100.0,
        current_spend=0.0,
    )
    db.add(tenant)
    db.flush()

    user = User(
        tenant_id=tenant.id,
        email="rag-api-test@example.com",
    )
    db.add(user)
    db.flush()

    def override_test_db():
        yield db

    fake_result = SimpleNamespace(
        generation=SimpleNamespace(
            status="answered",
            answer="The refund window is 30 days.",
            citations=[
                SimpleNamespace(
                    ref=("refund_policy.pdf", 3),
                    source="refund_policy.pdf",
                    page=3,
                ),
                SimpleNamespace(
                    ref=("terms.pdf", 7),
                    source="terms.pdf",
                    page=7,
                ),
            ],
        ),
        retrieved=[
            ("refund_policy.pdf", 3),
            ("terms.pdf", 7),
        ],
        timings_ms={
            "embedding": 12,
            "retrieval": 8,
            "generation": 140,
            "total": 160,
        },
        embedding_tokens=42,
    )

    class FakeRAGService:
        async def answer(self, *, tenant_id, query, top_k, budget):
            assert budget is not None, "route must pass RAGBudget to service"
            assert tenant_id == tenant.id
            assert query == "What is the refund policy?"
            assert top_k == 5
            return fake_result

    fake_service = FakeRAGService()

    def override_rag_service():
        return fake_service

    app.dependency_overrides[get_db] = override_test_db
    app.dependency_overrides[get_rag_service] = override_rag_service

    try:
        with patch(
            "app.api.dependencies.verify_access_token",
            return_value={
                "tenant_id": str(tenant.id),
                "user_id": str(user.id),
            },
        ):
            response = client.post(
                "/api/v1/rag/query",
                headers={
                    "Authorization": "Bearer valid-looking-token",
                },
                json={
                    "query": "What is the refund policy?",
                    "top_k": 5,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "answered"
    assert data["answer"] == "The refund window is 30 days."
    assert data["top_k"] == 5
    assert data["retrieved_count"] == 2
    assert data["embedding_tokens"] == 42

    assert data["timings_ms"] == {
        "embedding": 12,
        "retrieval": 8,
        "generation": 140,
        "total": 160,
    }

    assert data["citations"] == [
        {
            "ref": ["refund_policy.pdf", 3],
            "source": "refund_policy.pdf",
            "page": 3,
        },
        {
            "ref": ["terms.pdf", 7],
            "source": "terms.pdf",
            "page": 7,
        },
    ]


# ---------------------------------------------------------------------------
# API RAG Accounting & Ledger Integration Tests
# ---------------------------------------------------------------------------


class _ApiStubEmbedder:
    def __init__(self, tokens: int = 10) -> None:
        self.tokens = tokens
        self.calls: list[str] = []
        self.count_tokens_calls: list[str] = []

    @property
    def fingerprint(self) -> str:
        return "api-stub/embed/768"

    def embed_query(self, query: str) -> list[float]:
        self.calls.append(query)
        return [0.1] * 768

    def embed_query_with_usage(
        self,
        query: str,
        *,
        precomputed_tokens: int | None = None,
    ) -> EmbedResult:
        self.calls.append(query)
        tokens = (
            precomputed_tokens
            if precomputed_tokens is not None
            else self.count_tokens(query)
        )
        return EmbedResult(vector=[0.1] * 768, tokens=tokens)

    def count_tokens(self, text: str) -> int:
        self.count_tokens_calls.append(text)
        return self.tokens


class _ApiStubRetriever:
    def __init__(self) -> None:
        self.passages = [
            ContextPassage(
                ref=("doc.pdf", 1),
                text="Evidence passage",
                source="doc.pdf",
                page=1,
            )
        ]

    def search(self, **kwargs) -> list[ContextPassage]:
        return self.passages


class _ApiStubGenerator:
    def __init__(
        self,
        prompt_tokens: int = 30,
        completion_tokens: int = 20,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.config = type(
            "Config",
            (),
            {
                "model": DEFAULT_GENERATION_MODEL,
                "max_output_tokens": 2000,
            },
        )()

    def build_prompt(self, query: str, passages) -> str:
        return f"PROMPT: {query}"

    async def count_prompt_tokens(self, prompt: str) -> int:
        return self.prompt_tokens

    async def generate(
        self,
        query: str,
        passages,
        *,
        prompt: str | None = None,
    ) -> GenerationResult:
        return GenerationResult(
            status="answered",
            answer="Grounded answer.",
            citations=[Citation(ref=("doc.pdf", 1), source="doc.pdf", page=1)],
            passages_sent=[p.ref for p in passages],
            finish_reason="stop",
            invalid_citation_markers=0,
            prompt_version="v1",
            provider=ProviderMeta(
                provider="gemini",
                model=self.config.model,
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                latency_ms=80,
                thinking_tokens=10,
                total_tokens=self.prompt_tokens + 10 + self.completion_tokens,
            ),
        )


def test_rag_api_accounting_happy_path_records_usage_and_spend(db):
    tenant = Tenant(
        name=f"API Accounting Tenant {uuid4()}",
        monthly_budget=Decimal("50.000000"),
        current_spend=Decimal("0.000000"),
    )
    db.add(tenant)
    db.flush()

    user = User(
        tenant_id=tenant.id,
        email=f"api-accounting-{uuid4()}@example.com",
    )
    db.add(user)
    db.flush()

    service = RAGService(
        embedder=_ApiStubEmbedder(tokens=10),
        retriever=_ApiStubRetriever(),
        generator=_ApiStubGenerator(prompt_tokens=30, completion_tokens=20),
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        with patch(
            "app.api.dependencies.verify_access_token",
            return_value={
                "tenant_id": str(tenant.id),
                "user_id": str(user.id),
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
    assert isinstance(data["embedding_tokens"], int)
    assert data["embedding_tokens"] == 10

    db.refresh(tenant)
    embed_record = (
        db.query(UsageRecord)
        .filter(UsageRecord.tenant_id == tenant.id, UsageRecord.operation == "embed")
        .first()
    )
    generate_record = (
        db.query(UsageRecord)
        .filter(UsageRecord.tenant_id == tenant.id, UsageRecord.operation == "generate")
        .first()
    )

    assert embed_record is not None and embed_record.status == "completed"
    assert embed_record.input_tokens == 10
    assert embed_record.actual_cost > Decimal(0)

    assert generate_record is not None and generate_record.status == "completed"
    assert generate_record.input_tokens == 30
    assert generate_record.output_tokens == 20
    assert generate_record.actual_cost > Decimal(0)

    assert (
        tenant.current_spend == embed_record.actual_cost + generate_record.actual_cost
    )


def test_rag_api_accounting_timeout_maps_504_and_ledger(db):
    tenant = Tenant(
        name=f"API Timeout Tenant {uuid4()}",
        monthly_budget=Decimal("50.000000"),
        current_spend=Decimal("0.000000"),
    )
    db.add(tenant)
    db.flush()

    user = User(
        tenant_id=tenant.id,
        email=f"api-timeout-{uuid4()}@example.com",
    )
    db.add(user)
    db.flush()

    class _TimingOutGenerator(_ApiStubGenerator):
        async def generate(
            self,
            query: str,
            passages,
            *,
            prompt: str | None = None,
        ) -> GenerationResult:
            raise GenerationTimeoutError("Provider timed out after 30s")

    service = RAGService(
        embedder=_ApiStubEmbedder(tokens=10),
        retriever=_ApiStubRetriever(),
        generator=_TimingOutGenerator(),
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        with patch(
            "app.api.dependencies.verify_access_token",
            return_value={
                "tenant_id": str(tenant.id),
                "user_id": str(user.id),
            },
        ):
            response = client.post(
                "/api/v1/rag/query",
                headers={"Authorization": "Bearer token"},
                json={"query": "What is the policy?", "top_k": 5},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 504
    db.refresh(tenant)

    embed_record = (
        db.query(UsageRecord)
        .filter(UsageRecord.tenant_id == tenant.id, UsageRecord.operation == "embed")
        .first()
    )
    generate_record = (
        db.query(UsageRecord)
        .filter(UsageRecord.tenant_id == tenant.id, UsageRecord.operation == "generate")
        .first()
    )

    assert embed_record is not None and embed_record.status == "completed"
    assert embed_record.actual_cost > Decimal(0)

    assert generate_record is not None and generate_record.status == "failed"
    assert generate_record.actual_cost == Decimal("0.000000")

    assert tenant.current_spend == embed_record.actual_cost


def test_rag_api_accounting_budget_exceeded_returns_400_and_ledger(db):
    tenant = Tenant(
        name=f"API Exceeded Tenant {uuid4()}",
        monthly_budget=Decimal("0.000001"),
        current_spend=Decimal("0.000001"),
    )
    db.add(tenant)
    db.commit()
    db.refresh(tenant)

    user = User(
        tenant_id=tenant.id,
        email=f"api-exceeded-{uuid4()}@example.com",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    service = RAGService(
        embedder=_ApiStubEmbedder(tokens=10),
        retriever=_ApiStubRetriever(),
        generator=_ApiStubGenerator(),
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        with patch(
            "app.api.dependencies.verify_access_token",
            return_value={
                "tenant_id": str(tenant.id),
                "user_id": str(user.id),
            },
        ):
            response = client.post(
                "/api/v1/rag/query",
                headers={"Authorization": "Bearer token"},
                json={"query": "What is the policy?", "top_k": 5},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert "exceed budget" in response.json()["detail"].lower()

    tenant = db.query(Tenant).filter(Tenant.id == tenant.id).first()
    assert tenant is not None
    assert tenant.current_spend == Decimal("0.000001")

    reserved_records = (
        db.query(UsageRecord)
        .filter(UsageRecord.tenant_id == tenant.id, UsageRecord.status == "reserved")
        .all()
    )
    assert len(reserved_records) == 0


def test_rag_api_accounting_settlement_failure_returns_500_and_ledger(db):
    tenant = Tenant(
        name=f"API Settlement Fail Tenant {uuid4()}",
        monthly_budget=Decimal("50.000000"),
        current_spend=Decimal("0.000000"),
    )
    db.add(tenant)
    db.commit()
    db.refresh(tenant)

    user = User(
        tenant_id=tenant.id,
        email=f"api-settle-{uuid4()}@example.com",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    service = RAGService(
        embedder=_ApiStubEmbedder(tokens=10),
        retriever=_ApiStubRetriever(),
        generator=_ApiStubGenerator(),
    )

    real_settle_success = usage_service.settle_success
    call_count = 0

    def failing_settle(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("DB connection dropped during generation settlement")
        return real_settle_success(*args, **kwargs)

    client_no_raise = TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        with (
            patch(
                "app.api.dependencies.verify_access_token",
                return_value={
                    "tenant_id": str(tenant.id),
                    "user_id": str(user.id),
                },
            ),
            patch.object(usage_service, "settle_success", side_effect=failing_settle),
        ):
            response = client_no_raise.post(
                "/api/v1/rag/query",
                headers={"Authorization": "Bearer token"},
                json={"query": "What is the policy?", "top_k": 5},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    db.refresh(tenant)

    embed_record = (
        db.query(UsageRecord)
        .filter(UsageRecord.tenant_id == tenant.id, UsageRecord.operation == "embed")
        .first()
    )
    generate_record = (
        db.query(UsageRecord)
        .filter(UsageRecord.tenant_id == tenant.id, UsageRecord.operation == "generate")
        .first()
    )

    assert embed_record is not None and embed_record.status == "completed"
    assert generate_record is not None and generate_record.status == "failed"
    assert generate_record.actual_cost == Decimal("0.000000")

    reserved_records = (
        db.query(UsageRecord)
        .filter(UsageRecord.tenant_id == tenant.id, UsageRecord.status == "reserved")
        .all()
    )
    assert len(reserved_records) == 0
    assert tenant.current_spend == embed_record.actual_cost


def test_rag_route_missing_count_prompt_tokens_returns_500(db):
    tenant = Tenant(
        name=f"Config Error Tenant {uuid4()}",
        monthly_budget=Decimal("50.000000"),
        current_spend=Decimal("0.000000"),
    )
    db.add(tenant)
    db.flush()

    user = User(
        tenant_id=tenant.id,
        email=f"config-err-{uuid4()}@example.com",
    )
    db.add(user)
    db.flush()

    class _GeneratorWithoutCountTokens:
        async def generate(self, query: str, passages) -> GenerationResult:
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
        embedder=_ApiStubEmbedder(tokens=10),
        retriever=_ApiStubRetriever(),
        generator=_GeneratorWithoutCountTokens(),
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        with patch(
            "app.api.dependencies.verify_access_token",
            return_value={
                "tenant_id": str(tenant.id),
                "user_id": str(user.id),
            },
        ):
            response = client.post(
                "/api/v1/rag/query",
                headers={"Authorization": "Bearer token"},
                json={"query": "What is the policy?", "top_k": 5},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert "Server misconfiguration" in response.json()["detail"]
