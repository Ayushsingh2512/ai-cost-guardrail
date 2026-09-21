from sqlalchemy.orm import Session
from app.services.models import Tenant


class RateLimitExceeded(Exception):
    pass


class GuardrailService:
    ALLOWED_MODELS = ["gemini-3-flash-preview", "gemini-2.5-flash"]
    DAILY_BUDGET = 1.0
    COST_PER_1000_TOKENS = 0.001

    def __init__(self):
        self.user_spend: dict[str, float] = {}

    def check_token_limit(self, max_tokens: int) -> None:
        if max_tokens > 2000:
            raise ValueError(f"max_tokens ({max_tokens}) exceeds allowed limit of 2000")

    def check_model_policy(self, model: str) -> None:
        if model not in self.ALLOWED_MODELS:
            raise ValueError(f"Model '{model}' is not allowed. Allowed models: {self.ALLOWED_MODELS}")

    def estimate_cost(self, max_tokens: int) -> float:
        return (max_tokens / 1000) * self.COST_PER_1000_TOKENS

    def check_and_reserve_budget(self, db: Session, tenant_id: int, max_tokens: int) -> float:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if tenant is None:
            raise ValueError(f"Tenant {tenant_id} not found")
        cost = self.estimate_cost(max_tokens)
        if tenant.current_spend + cost > tenant.monthly_budget:
            raise ValueError(
                f"Request would exceed budget. Spent so far: ${tenant.current_spend:.4f},"
                f"Limit: ${tenant.monthly_budget:.4f}"
            )
        tenant.current_spend += cost
        db.commit()
        return cost
    def check_rate_limit(self, redis_client, tenant_id: int, limit: int = 30, window_seconds: int = 60) -> None:
        key = f"ratelimit:{tenant_id}"
        current_count = redis_client.incr(key)
        if current_count == 1:
            redis_client.expire(key, window_seconds)
        if current_count > limit:
            raise RateLimitExceeded(f"Rate limit exceeded: {limit} requests per { window_seconds}s")
       


guardrail_service = GuardrailService()  