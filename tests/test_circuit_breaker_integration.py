from fastapi.testclient import TestClient

from app.main import app
from app.api.dependencies import (
    get_genai_client,
    enforce_guardrails,
)
from app.services.database import get_db
from app.schemas.chat import ChatRequest
from app.services.circuit_breaker import (
    circuit_breaker,
    CircuitState,
)
from app.services.models import Tenant, User


# ---------------------------------------------------------
# Fake Gemini client
# ---------------------------------------------------------


class FakeResponse:
    def __init__(self):
        self.text = "Fake Gemini response"

        class UsageMetadata:
            prompt_token_count = 50
            candidates_token_count = 50
            total_token_count = 100

        self.usage_metadata = UsageMetadata()


class FakeModels:
    def __init__(self):
        self.calls = 0
        self.should_fail = True

    async def generate_content(self, **kwargs):
        self.calls += 1

        if self.should_fail:
            raise RuntimeError("Simulated Gemini failure")

        return FakeResponse()


class FakeAio:
    def __init__(self):
        self.models = FakeModels()


class FakeGeminiClient:
    def __init__(self):
        self.aio = FakeAio()


fake_client = FakeGeminiClient()


def override_genai_client():
    return fake_client


# ---------------------------------------------------------
# Override guardrails for this integration test
# ---------------------------------------------------------


def override_guardrails(request: ChatRequest):
    return request


# ---------------------------------------------------------
# Integration test
# ---------------------------------------------------------


def test_circuit_breaker_full_lifecycle(db):

    # -----------------------------------------------------
    # Create an isolated test tenant
    # -----------------------------------------------------

    tenant = Tenant(
        name="Circuit Breaker Test Tenant",
        monthly_budget=100.0,
        current_spend=0.0,
    )

    db.add(tenant)
    db.flush()

    user = User(
        tenant_id=tenant.id,
        email="circuit-breaker-test@example.com",
    )

    db.add(user)
    db.flush()

    # -----------------------------------------------------
    # Start from a clean circuit-breaker state
    # -----------------------------------------------------

    circuit_breaker.state = CircuitState.CLOSED
    circuit_breaker.failure_count = 0
    circuit_breaker.opened_at = None

    # Start fake Gemini in failure mode
    fake_client.aio.models.calls = 0
    fake_client.aio.models.should_fail = True

    # -----------------------------------------------------
    # Override dependencies
    # -----------------------------------------------------

    def override_test_db():
        yield db

    app.dependency_overrides[get_genai_client] = override_genai_client
    app.dependency_overrides[enforce_guardrails] = override_guardrails
    app.dependency_overrides[get_db] = override_test_db

    try:

        with TestClient(app) as client:

            # -------------------------------------------------
            # Get JWT for our isolated test user
            # -------------------------------------------------

            token_response = client.post(
                "/token",
                params={
                    "tenant_id": tenant.id,
                    "user_id": user.id,
                },
            )

            assert token_response.status_code == 200

            token = token_response.json()["access_token"]

            headers = {
                "Authorization": f"Bearer {token}"
            }

            payload = {
                "model": "gemini-3-flash-preview",
                "message": "Test circuit breaker",
                "max_tokens": 100,
            }

            # -------------------------------------------------
            # 1. Five upstream failures
            # -------------------------------------------------

            for _ in range(5):

                response = client.post(
                    "/api/v1/chat",
                    json=payload,
                    headers=headers,
                )

                assert response.status_code == 502

            # Five consecutive failures should open the circuit
            assert circuit_breaker.state == CircuitState.OPEN

            # Gemini should have been called exactly five times
            assert fake_client.aio.models.calls == 5

            # -------------------------------------------------
            # 2. Circuit is OPEN
            # -------------------------------------------------

            response = client.post(
                "/api/v1/chat",
                json=payload,
                headers=headers,
            )

            assert response.status_code == 503

            # Gemini should NOT have been called again
            assert fake_client.aio.models.calls == 5

            # -------------------------------------------------
            # 3. Move past recovery timeout
            # -------------------------------------------------

            circuit_breaker.opened_at -= 31

            assert circuit_breaker.allow_request() is True
            assert circuit_breaker.state == CircuitState.HALF_OPEN

            # -------------------------------------------------
            # 4. Gemini recovers
            # -------------------------------------------------

            fake_client.aio.models.should_fail = False

            response = client.post(
                "/api/v1/chat",
                json=payload,
                headers=headers,
            )

            assert response.status_code == 200

            # Successful probe should close the circuit
            assert circuit_breaker.state == CircuitState.CLOSED

            # Failure counter should reset
            assert circuit_breaker.failure_count == 0

    finally:

        # -----------------------------------------------------
        # Always clean up dependency overrides
        # -----------------------------------------------------

        app.dependency_overrides.clear()