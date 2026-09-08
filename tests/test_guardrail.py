import pytest
from app.services.guardrail import GuardrailService

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
    service = GuardrailService()
    # DAILY_BUDGET is 1.0, COST_PER_1000_TOKENS is 0.001
    # max_tokens that would cost more than $1.00: need max_tokens/1000 * 0.001 > 1.0
    # that requires max_tokens > 1,000,000 — but check_and_reserve_budget doesn't cap tokens itself,
    # so instead: call it enough times to exceed budget, or reserve a huge amount once
    with pytest.raises(ValueError):
        service.check_and_reserve_budget("test_user", 2_000_000_000)