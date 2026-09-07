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

    def check_and_reserve_budget(self, user_id: str, max_tokens: int) -> float:
        cost = self.estimate_cost(max_tokens)
        current_spend = self.user_spend.get(user_id, 0.0)
        if current_spend + cost > self.DAILY_BUDGET:
            raise ValueError(
                f"Request would exceed daily budget. Spent so far: ${current_spend:.4f}, Limit: ${self.DAILY_BUDGET:.4f}"
            )
        self.user_spend[user_id] = current_spend + cost
        return cost


guardrail_service = GuardrailService()  