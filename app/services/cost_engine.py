from decimal import Decimal


class CostEngine:
    MODEL_PRICING = {
        "gemini-3-flash-preview": {
            "input_per_1m": Decimal("0.50"),
            "output_per_1m": Decimal("3.00"),
        },
    }

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
            * pricing["input_per_1m"]
        )

        output_cost = (
            Decimal(max_output_tokens)
            / Decimal("1000000")
            * pricing["output_per_1m"]
        )

        return input_cost + output_cost

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
            * pricing["input_per_1m"]
        )

        billable_output_tokens = thinking_tokens + output_tokens

        output_cost = (
            Decimal(billable_output_tokens)
            / Decimal("1000000")
            * pricing["output_per_1m"]
        )

        return input_cost + output_cost

    def estimate_embedding_cost(
        self,
        model: str,
        input_tokens: int,
    ) -> Decimal:
        """Calculate embedding cost from estimated input tokens."""

        pricing = self._get_embedding_pricing(model)

        return (
            Decimal(input_tokens)
            / Decimal("1000000")
            * pricing["input_per_1m"]
        )

    def calculate_embedding_cost(
        self,
        model: str,
        input_tokens: int,
    ) -> Decimal:
        """Calculate actual embedding cost from input tokens."""

        return self.estimate_embedding_cost(
            model=model,
            input_tokens=input_tokens,
        )

    def _get_pricing(self, model: str) -> dict[str, Decimal]:
        try:
            return self.MODEL_PRICING[model]
        except KeyError:
            raise ValueError(
                f"No pricing configuration exists for model '{model}'"
            )

    def _get_embedding_pricing(self, model: str) -> dict[str, Decimal]:
        try:
            return self.EMBEDDING_PRICING[model]
        except KeyError:
            raise ValueError(
                f"No embedding pricing configuration exists for model '{model}'"
            )


cost_engine = CostEngine()