import pytest
from app.services.guardrail import GuardrailService
from app.services.database import SessionLocal

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


def test_budget_rejects_over_daily_limit():
    db = SessionLocal()
    try:
        service = GuardrailService()
        with pytest.raises(ValueError):
            service.check_and_reserve_budget(db, tenant_id=1, max_tokens=2_000_000_000_000)
    finally:
        db.close()
   