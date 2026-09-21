import pytest
import redis
from app.services.database import SessionLocal
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


def test_rate_limit_sets_expiration():
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

    ttl = client.ttl(key)

    assert 0 < ttl <= 60
def test_rate_limit_fails_when_redis_is_unavailable():
    service = GuardrailService()

    class BrokenRedis:
        def incr(self, key):
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


def test_budget_rejects_over_daily_limit():
    db = SessionLocal()
    try:
        service = GuardrailService()
        with pytest.raises(ValueError):
            service.check_and_reserve_budget(db, tenant_id=1, max_tokens=2_000_000_000_000)
    finally:
        db.close()
   