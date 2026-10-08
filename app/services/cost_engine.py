from decimal import Decimal, ROUND_HALF_UP

from app.core.model_registry import MODEL_REGISTRY


class CostEngine:
    MONEY_QUANTUM = Decimal("0.000001")

    EMBEDDING_PRICING = {
        "gemini-embedding-001": {
            "input_per_1m": Decimal("0.15"),
        },
    }

    def estimate_reservation(
        self,
        model: str,
        input_tokens: int,
        max_output_tokens: int,
    ) -> Decimal:
        """Estimate maximum generation cost for a request."""

        pricing = self._get_pricing(model)

        input_cost = (
            Decimal(input_tokens)
            / Decimal("1000000")
            * pricing.input_per_1m
        )

        output_cost = (
            Decimal(max_output_tokens)
            / Decimal("1000000")
            * pricing.output_per_1m
        )

        return self._quantize_cost(input_cost + output_cost)

    def calculate_actual_cost(
        self,
        model: str,
        input_tokens: int,
        thinking_tokens: int,
        output_tokens: int,
    ) -> Decimal:
        """Calculate actual generation cost from provider usage."""

        pricing = self._get_pricing(model)

        input_cost = (
            Decimal(input_tokens)
            / Decimal("1000000")
            * pricing.input_per_1m
        )

        billable_output_tokens = (
            thinking_tokens + output_tokens
        )

        output_cost = (
            Decimal(billable_output_tokens)
            / Decimal("1000000")
            * pricing.output_per_1m
        )

        return self._quantize_cost(input_cost + output_cost)

    def estimate_embedding_cost(
        self,
        model: str,
        input_tokens: int,
    ) -> Decimal:
        """Calculate estimated embedding cost."""

        pricing = self._get_embedding_pricing(model)

        raw_cost = (
            Decimal(input_tokens)
            / Decimal("1000000")
            * pricing["input_per_1m"]
        )

        return self._quantize_cost(raw_cost)

    def calculate_embedding_cost(
        self,
        model: str,
        input_tokens: int,
    ) -> Decimal:
        """Calculate actual embedding cost."""

        return self.estimate_embedding_cost(
            model=model,
            input_tokens=input_tokens,
        )

    def _quantize_cost(self, amount: Decimal) -> Decimal:
        """Quantize monetary values to NUMERIC(12,6) precision."""

        return amount.quantize(
            self.MONEY_QUANTUM,
            rounding=ROUND_HALF_UP,
        )

    def _get_pricing(self, model: str):
        try:
            return MODEL_REGISTRY[model]
        except KeyError as exc:
            raise ValueError(
                f"No pricing configuration exists for model '{model}'"
            ) from exc

    def _get_embedding_pricing(
        self,
        model: str,
    ) -> dict[str, Decimal]:
        try:
            return self.EMBEDDING_PRICING[model]
        except KeyError as exc:
            raise ValueError(
                "No embedding pricing configuration exists "
                f"for model '{model}'"
            ) from exc


cost_engine = CostEngine()