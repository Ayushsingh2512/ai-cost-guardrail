from decimal import Decimal


class CostEngine:
    MODEL_PRICING = {
        "gemini-3-flash-preview": {
            "input_per_1m": Decimal("0.50"),
            "output_per_1m": Decimal("3.00"),
        },
    }

    def estimate_reservation(
        self,
        model: str,
        input_tokens: int,
        max_output_tokens: int,
    ) -> Decimal:
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
        output_tokens: int,
    ) -> Decimal:
        pricing = self._get_pricing(model)

        input_cost = (
            Decimal(input_tokens)
            / Decimal("1000000")
            * pricing["input_per_1m"]
        )

        output_cost = (
            Decimal(output_tokens)
            / Decimal("1000000")
            * pricing["output_per_1m"]
        )

        return input_cost + output_cost

    def _get_pricing(self, model: str) -> dict[str, Decimal]:
        try:
            return self.MODEL_PRICING[model]
        except KeyError:
            raise ValueError(
                f"No pricing configuration exists for model '{model}'"
            )


cost_engine = CostEngine()