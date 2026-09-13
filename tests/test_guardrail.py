import pytest
from app.services.guardrail import GuardrailService
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


def test_budget_rejects_over_daily_limit():
    db = SessionLocal()
    try:
        service = GuardrailService()
        with pytest.raises(ValueError):
            service.check_and_reserve_budget(db, tenant_id=1, max_tokens=2_000_000_000_000)
    finally:
        db.close()
   