from fastapi.testclient import TestClient

from app.main import app

from unittest.mock import patch

from app.api.dependencies import get_redis_client

from app.services.guardrail import RateLimitExceeded

import httpx


client = TestClient(app)


def test_chat_requires_authentication():
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "Hello",
            "model": "gemini-3-flash-preview",
            "max_tokens": 100,
        },
    )

    assert response.status_code == 401
    
def test_chat_rejects_invalid_token():
    response = client.post(
        "/api/v1/chat",
        headers={
            "Authorization": "Bearer definitely-not-a-valid-jwt",
        },
        json={
            "message": "Hello",
            "model": "gemini-3-flash-preview",
            "max_tokens": 100,
        },
    )

    assert response.status_code == 401
    
def test_chat_rejects_token_without_tenant_id():
    with patch(
        "app.api.dependencies.verify_access_token",
        return_value={"user_id": "1"},
    ):
        response = client.post(
            "/api/v1/chat",
            headers={
                "Authorization": "Bearer valid-looking-token",
            },
            json={
                "message": "Hello",
                "model": "gemini-3-flash-preview",
                "max_tokens": 100,
            },
        )

    assert response.status_code == 401
    
def test_chat_rejects_token_without_user_id():
    with patch(
        "app.api.dependencies.verify_access_token",
        return_value={"tenant_id": "1"},
    ):
        response = client.post(
            "/api/v1/chat",
            headers={
                "Authorization": "Bearer valid-looking-token",
            },
            json={
                "message": "Hello",
                "model": "gemini-3-flash-preview",
                "max_tokens": 100,
            },
        )

    assert response.status_code == 401
def test_chat_rejects_excessive_token_limit():
    with patch(
        "app.api.dependencies.verify_access_token",
        return_value={
            "tenant_id": "1",
            "user_id": "1",
        },
    ):
        response = client.post(
            "/api/v1/chat",
            headers={
                "Authorization": "Bearer valid-looking-token",
            },
            json={
                "message": "Hello",
                "model": "gemini-3-flash-preview",
                "max_tokens": 2001,
            },
        )

    assert response.status_code == 400
def test_chat_rejects_unsupported_model():
    with patch(
        "app.api.dependencies.verify_access_token",
        return_value={
            "tenant_id": "1",
            "user_id": "1",
        },
    ):
        response = client.post(
            "/api/v1/chat",
            headers={
                "Authorization": "Bearer valid-looking-token",
            },
            json={
                "message": "Hello",
                "model": "some-nonexistent-model",
                "max_tokens": 100,
            },
        )

    assert response.status_code == 400
    
def test_chat_returns_429_when_rate_limit_exceeded():
    def fake_redis():
        return object()

    def fake_rate_limit(redis_client, tenant_id, limit=30, window_seconds=60):
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
                "/api/v1/chat",
                headers={
                    "Authorization": "Bearer valid-looking-token",
                },
                json={
                    "message": "Hello",
                    "model": "gemini-3-flash-preview",
                    "max_tokens": 100,
                },
            )
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 429
def test_chat_returns_503_when_rate_limiter_unavailable():
    def fake_redis():
        return object()

    def fake_rate_limit(redis_client, tenant_id, limit=30, window_seconds=60):
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
                "/api/v1/chat",
                headers={
                    "Authorization": "Bearer valid-looking-token",
                },
                json={
                    "message": "Hello",
                    "model": "gemini-3-flash-preview",
                    "max_tokens": 100,
                },
            )
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 503
    
def test_chat_returns_503_when_token_counting_fails(db):
    from app.api.dependencies import get_genai_client
    from app.services.database import get_db
    from app.services.models import Tenant, User

    class FakeModels:
        def __init__(self):
            self.generate_called = False

        async def count_tokens(self, **kwargs):
            raise httpx.ReadTimeout("Simulated token counting timeout")

        async def generate_content(self, **kwargs):
            self.generate_called = True
            raise AssertionError("generate_content must not be called")

    class FakeAio:
        def __init__(self):
            self.models = FakeModels()

    class FakeClient:
        def __init__(self):
            self.aio = FakeAio()

    fake_client = FakeClient()

    tenant = Tenant(
        name="Token Count Failure Test Tenant",
        monthly_budget=100.0,
        current_spend=0.0,
    )
    db.add(tenant)
    db.flush()

    user = User(
        tenant_id=tenant.id,
        email="token-count-failure@example.com",
    )
    db.add(user)
    db.flush()

    def override_test_db():
        yield db

    def override_genai_client():
        return fake_client

    app.dependency_overrides[get_db] = override_test_db
    app.dependency_overrides[get_genai_client] = override_genai_client

    try:
        with patch(
            "app.api.dependencies.verify_access_token",
            return_value={
                "tenant_id": str(tenant.id),
                "user_id": str(user.id),
            },
        ):
            response = client.post(
                "/api/v1/chat",
                headers={
                    "Authorization": "Bearer valid-looking-token",
                },
                json={
                    "message": "Hello",
                    "model": "gemini-3-flash-preview",
                    "max_tokens": 100,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert fake_client.aio.models.generate_called is False

    db.refresh(tenant)

    assert tenant.current_spend == 0