from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ModelConfig:
    name: str
    input_per_1m: Decimal
    output_per_1m: Decimal
    max_output_tokens: int = 2000


DEFAULT_GENERATION_MODEL = "gemini-3-flash-preview"


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gemini-3-flash-preview": ModelConfig(
        name="gemini-3-flash-preview",
        input_per_1m=Decimal("0.50"),
        output_per_1m=Decimal("3.00"),
        max_output_tokens=2000,
    ),
    "gemini-2.5-flash": ModelConfig(
        name="gemini-2.5-flash",
        input_per_1m=Decimal("0.30"),
        output_per_1m=Decimal("2.50"),
        max_output_tokens=2000,
    ),
}