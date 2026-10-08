import redis

from app.core.model_registry import MODEL_REGISTRY


class RateLimitExceeded(Exception):
    pass


class GuardrailService:
    def check_token_limit(
        self,
        max_tokens: int,
        model: str | None = None,
    ) -> None:
        """
        Validate the requested output-token limit.

        When a model is supplied, use that model's configured limit.
        The optional model argument keeps the service backwards-compatible
        with existing direct unit tests.
        """
        if max_tokens < 0:
            raise ValueError(
                f"max_tokens ({max_tokens}) cannot be negative"
            )

        if model is None:
            allowed_limit = 2000
        else:
            model_config = MODEL_REGISTRY.get(model)

            if model_config is None:
                raise ValueError(
                    f"Model '{model}' is not allowed. "
                    f"Allowed models: {list(MODEL_REGISTRY)}"
                )

            allowed_limit = model_config.max_output_tokens

        if max_tokens > allowed_limit:
            raise ValueError(
                f"max_tokens ({max_tokens}) exceeds allowed limit "
                f"of {allowed_limit} for model '{model}'"
                if model is not None
                else f"max_tokens ({max_tokens}) exceeds "
                     f"allowed limit of {allowed_limit}"
            )

    def check_model_policy(self, model: str) -> None:
        if model not in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{model}' is not allowed. "
                f"Allowed models: {list(MODEL_REGISTRY)}"
            )

    def check_rate_limit(
        self,
        redis_client,
        tenant_id: int,
        limit: int = 30,
        window_seconds: int = 60,
    ) -> None:
        key = f"ratelimit:{tenant_id}"

        try:
            script = """
            local count = redis.call("INCR", KEYS[1])

            if count == 1 then
                redis.call("EXPIRE", KEYS[1], ARGV[1])
            end

            return count
            """

            current_count = redis_client.eval(
                script,
                1,
                key,
                window_seconds,
            )

        except redis.exceptions.RedisError as exc:
            raise RuntimeError(
                "Rate limiting service is unavailable"
            ) from exc

        if current_count > limit:
            raise RateLimitExceeded(
                f"Rate limit exceeded: "
                f"{limit} requests per {window_seconds}s"
            )


guardrail_service = GuardrailService()