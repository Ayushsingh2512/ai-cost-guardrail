from fastapi.testclient import TestClient

from app.main import app

from unittest.mock import patch

from app.api.dependencies import get_redis_client

from app.services.guardrail import RateLimitExceeded
from app.schemas.chat import ChatRequest
import httpx

from app.api.dependencies import enforce_guardrails, get_genai_client
from app.services.database import get_db
from app.services.models import Tenant, User, UsageRecord
from app.services.circuit_breaker import (
    circuit_breaker,
    CircuitState,
)


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
            self.last_count_kwargs = None

        async def count_tokens(self, **kwargs):
            self.last_count_kwargs = kwargs
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
    assert fake_client.aio.models.last_count_kwargs["contents"] == "Hello"
    assert fake_client.aio.models.last_count_kwargs["model"] == "gemini-3-flash-preview"
    assert fake_client.aio.models.generate_called is False

    db.refresh(tenant)

    assert tenant.current_spend == 0
    
def test_chat_recovers_when_settlement_fails(db):
    class FakeUsageMetadata:
        prompt_token_count = 50
        candidates_token_count = 50
        total_token_count = 100
        thoughts_token_count = 0

    class FakeResponse:
        text = "Successful Gemini response"
        usage_metadata = FakeUsageMetadata()

    class FakeModels:
        async def count_tokens(self, **kwargs):
            class TokenCount:
                total_tokens = 50

            return TokenCount()

        async def generate_content(self, **kwargs):
            return FakeResponse()

    class FakeAio:
        def __init__(self):
            self.models = FakeModels()

    class FakeClient:
        def __init__(self):
            self.aio = FakeAio()

    fake_client = FakeClient()

    tenant = Tenant(
        name="Settlement Failure Test Tenant",
        monthly_budget=100.0,
        current_spend=0.0,
    )
    db.add(tenant)
    db.flush()

    user = User(
        tenant_id=tenant.id,
        email="settlement-failure@example.com",
    )
    db.add(user)
    db.flush()

    initial_spend = tenant.current_spend
    request_id = "settlement-failure-test"

    def override_test_db():
        yield db

    def override_genai_client():
        return fake_client

    def override_guardrails(request: ChatRequest) -> ChatRequest:
        return request

    app.dependency_overrides[get_db] = override_test_db
    app.dependency_overrides[get_genai_client] = override_genai_client
    app.dependency_overrides[enforce_guardrails] = override_guardrails

    try:
        with patch(
            "app.api.dependencies.verify_access_token",
            return_value={
                "tenant_id": str(tenant.id),
                "user_id": str(user.id),
            },
        ), patch(
            "app.api.v1.chat.uuid4",
            return_value=request_id,
        ), patch(
            "app.api.v1.chat.usage_service.settle_success",
            side_effect=RuntimeError("Simulated settlement failure"),
        ):
            response = client.post(
                "/api/v1/chat",
                headers={
                    "Authorization": "Bearer valid-looking-token",
                },
                json={
                    "message": "Test settlement failure",
                    "model": "gemini-3-flash-preview",
                    "max_tokens": 100,
                },
            )

    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "LLM succeeded but usage settlement failed"
    )

    db.refresh(tenant)

    assert tenant.current_spend == initial_spend

    usage = (
        db.query(UsageRecord)
        .filter(UsageRecord.request_id == request_id)
        .one()
    )

    assert usage.status == "failed"
    assert usage.actual_cost == 0
    
def test_chat_returns_504_when_llm_generation_times_out(db):
    class FakeUsageMetadata:
        prompt_token_count = 50
        candidates_token_count = 50
        total_token_count = 100
        thoughts_token_count = 0

    class FakeResponse:
        text = "Should never be returned"
        usage_metadata = FakeUsageMetadata()

    class FakeModels:
        async def count_tokens(self, **kwargs):
            class TokenCount:
                total_tokens = 50

            return TokenCount()

        async def generate_content(self, **kwargs):
            raise httpx.ReadTimeout(
                "Simulated LLM generation timeout"
            )

    class FakeAio:
        def __init__(self):
            self.models = FakeModels()

    class FakeClient:
        def __init__(self):
            self.aio = FakeAio()

    fake_client = FakeClient()

    tenant = Tenant(
        name="LLM Timeout Test Tenant",
        monthly_budget=100.0,
        current_spend=0.0,
    )
    db.add(tenant)
    db.flush()

    user = User(
        tenant_id=tenant.id,
        email="llm-timeout@example.com",
    )
    db.add(user)
    db.flush()

    initial_spend = tenant.current_spend

    def override_test_db():
        yield db

    def override_genai_client():
        return fake_client

    def override_guardrails(
        request: ChatRequest,
    ) -> ChatRequest:
        return request

    app.dependency_overrides[get_db] = override_test_db
    app.dependency_overrides[get_genai_client] = override_genai_client
    app.dependency_overrides[enforce_guardrails] = override_guardrails

    original_failure_count = circuit_breaker.failure_count
    original_state = circuit_breaker.state
    original_opened_at = circuit_breaker.opened_at
    original_probe_in_flight = circuit_breaker._probe_in_flight

    circuit_breaker.state = CircuitState.CLOSED
    circuit_breaker.failure_count = 0
    circuit_breaker.opened_at = None
    circuit_breaker._probe_in_flight = False

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
                    "message": "Test timeout",
                    "model": "gemini-3-flash-preview",
                    "max_tokens": 100,
                },
            )

    finally:
        app.dependency_overrides.clear()

        circuit_breaker.failure_count = original_failure_count
        circuit_breaker.state = original_state
        circuit_breaker.opened_at = original_opened_at
        circuit_breaker._probe_in_flight = original_probe_in_flight

    assert response.status_code == 504
    assert response.json()["detail"] == (
        "LLM provider request timed out"
    )

    db.refresh(tenant)

    # The reservation must be released.
    assert tenant.current_spend == initial_spend