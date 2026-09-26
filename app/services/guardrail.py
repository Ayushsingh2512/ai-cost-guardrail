import redis

class RateLimitExceeded(Exception):
    pass


class GuardrailService:
    ALLOWED_MODELS = ["gemini-3-flash-preview", "gemini-2.5-flash"]
    DAILY_BUDGET = 1.0

    def __init__(self):
        self.user_spend: dict[str, float] = {}

    def check_token_limit(self, max_tokens: int) -> None:
        if max_tokens > 2000:
            raise ValueError(f"max_tokens ({max_tokens}) exceeds allowed limit of 2000")

    def check_model_policy(self, model: str) -> None:
        if model not in self.ALLOWED_MODELS:
            raise ValueError(f"Model '{model}' is not allowed. Allowed models: {self.ALLOWED_MODELS}")

    def check_rate_limit(
        self,
        redis_client,
        tenant_id: int,
        limit: int = 30,
        window_seconds: int = 60,
    ) -> None:
        key = f"ratelimit:{tenant_id}"

        try:
            current_count = redis_client.incr(key)

            if current_count == 1:
                redis_client.expire(key, window_seconds)

        except redis.exceptions.RedisError as e:
            raise RuntimeError(
                "Rate limiting service is unavailable"
            ) from e

        if current_count > limit:
            raise RateLimitExceeded(
                f"Rate limit exceeded: {limit} requests per {window_seconds}s"
            )


guardrail_service = GuardrailService()  