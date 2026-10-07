from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.dependencies import get_redis_client
from app.api.v1.rag import get_rag_service
from app.main import app
from app.services.database import get_db
from app.services.guardrail import RateLimitExceeded
from app.services.models import Tenant, User


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
    with patch(
        "app.api.dependencies.verify_access_token",
        return_value={
            "tenant_id": "1",
            "user_id": "1",
        },
    ), patch(
        "app.api.dependencies.guardrail_service.check_token_limit",
        side_effect=ValueError("Token limit exceeded"),
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

    with patch(
        "app.api.dependencies.verify_access_token",
        return_value={
            "tenant_id": "1",
            "user_id": "1",
        },
    ), patch(
        "app.api.dependencies.guardrail_service.check_rate_limit",
        side_effect=fake_rate_limit,
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

    with patch(
        "app.api.dependencies.verify_access_token",
        return_value={
            "tenant_id": "1",
            "user_id": "1",
        },
    ), patch(
        "app.api.dependencies.guardrail_service.check_rate_limit",
        side_effect=fake_rate_limit,
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
        async def answer(self, *, tenant_id, query, top_k):
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