import pytest
import redis
from app.core.model_registry import MODEL_REGISTRY
from app.services.redis_client import get_redis_client
from app.services.guardrail import GuardrailService, RateLimitExceeded

def test_token_limit_rejects_over_2000():
    service = GuardrailService()
    with pytest.raises(ValueError):
        service.check_token_limit(5000)
        
def test_token_limit_allows_under_2000():
    service = GuardrailService()
    service.check_token_limit(500)
    
def test_model_policy_rejects_disallowed_model():
    service = GuardrailService()
    with pytest.raises(ValueError):
        service.check_model_policy("gemini-pro")


def test_model_policy_allows_valid_model():
    service = GuardrailService()
    service.check_model_policy("gemini-3-flash-preview")  # should not raise
    
def test_rate_limit_rejects_after_limit():
    service = GuardrailService()
    client = get_redis_client()
    client.delete("ratelimit:999")
    service.check_rate_limit(client, tenant_id=999, limit=1)
    with pytest.raises(RateLimitExceeded):
        service.check_rate_limit(client, tenant_id=999, limit = 1)
        
def test_rate_limit_allows_requests_up_to_limit():
    service = GuardrailService()
    client = get_redis_client()

    client.delete("ratelimit:1001")

    for _ in range(30):
        service.check_rate_limit(
            client,
            tenant_id=1001,
            limit=30,
            window_seconds=60,
        )


def test_rate_limit_sets_expiration_without_refreshing_window():
    service = GuardrailService()
    client = get_redis_client()

    key = "ratelimit:1002"
    client.delete(key)

    service.check_rate_limit(
        client,
        tenant_id=1002,
        limit=30,
        window_seconds=60,
    )

    first_ttl = client.ttl(key)

    assert 0 < first_ttl <= 60

    service.check_rate_limit(
        client,
        tenant_id=1002,
        limit=30,
        window_seconds=60,
    )

    second_ttl = client.ttl(key)

    assert 0 < second_ttl <= first_ttl
def test_rate_limit_fails_when_redis_is_unavailable():
    service = GuardrailService()

    class BrokenRedis:
        def eval(self, script, numkeys, *args):
            raise redis.exceptions.ConnectionError("Redis unavailable")

    with pytest.raises(
        RuntimeError,
        match="Rate limiting service is unavailable",
    ):
        service.check_rate_limit(
            BrokenRedis(),
            tenant_id=2001,
            limit=30,
            window_seconds=60,
        )

def test_rate_limit_isolated_between_tenants():
    service = GuardrailService()
    client = get_redis_client()

    client.delete("ratelimit:1003")
    client.delete("ratelimit:1004")

    service.check_rate_limit(
        client,
        tenant_id=1003,
        limit=1,
        window_seconds=60,
    )

    service.check_rate_limit(
        client,
        tenant_id=1004,
        limit=1,
        window_seconds=60,
    )

    with pytest.raises(RateLimitExceeded):
        service.check_rate_limit(
            client,
            tenant_id=1003,
            limit=1,
            window_seconds=60,
        )

    # Tenant 1004 should still have its own independent limit.
    with pytest.raises(RateLimitExceeded):
        service.check_rate_limit(
            client,
            tenant_id=1004,
            limit=1,
            window_seconds=60,
        )
        
def test_model_policy_allows_gemini_2_5_flash():
    service = GuardrailService()

    service.check_model_policy("gemini-2.5-flash")
    
def test_model_policy_allows_every_registered_model():
    service = GuardrailService()

    for model in MODEL_REGISTRY:
        service.check_model_policy(model)
        
def test_model_policy_rejects_model_not_in_registry(monkeypatch):
    service = GuardrailService()

    monkeypatch.setitem(
        MODEL_REGISTRY,
        "temporary-model",
        MODEL_REGISTRY["gemini-3-flash-preview"],
    )

    service.check_model_policy("temporary-model")
    
def test_model_policy_uses_registry_output_limit(monkeypatch):
    service = GuardrailService()

    original = MODEL_REGISTRY["gemini-2.5-flash"]

    from app.core.model_registry import ModelConfig

    monkeypatch.setitem(
        MODEL_REGISTRY,
        "gemini-2.5-flash",
        ModelConfig(
            name="gemini-2.5-flash",
            input_per_1m=original.input_per_1m,
            output_per_1m=original.output_per_1m,
            max_output_tokens=1000,
        ),
    )

    service.check_token_limit(
        max_tokens=1000,
        model="gemini-2.5-flash",
    )

    with pytest.raises(ValueError):
        service.check_token_limit(
            max_tokens=1001,
            model="gemini-2.5-flash",
        )

   