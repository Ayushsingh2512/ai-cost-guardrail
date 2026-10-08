from decimal import Decimal

from app.core.model_registry import (
    DEFAULT_GENERATION_MODEL,
    MODEL_REGISTRY,
)


def test_registry_contains_supported_generation_models():
    assert "gemini-3-flash-preview" in MODEL_REGISTRY
    assert "gemini-2.5-flash" in MODEL_REGISTRY


def test_default_generation_model_is_registered():
    assert DEFAULT_GENERATION_MODEL in MODEL_REGISTRY


def test_model_registry_contains_pricing():
    flash_3 = MODEL_REGISTRY["gemini-3-flash-preview"]
    flash_25 = MODEL_REGISTRY["gemini-2.5-flash"]

    assert flash_3.input_per_1m == Decimal("0.50")
    assert flash_3.output_per_1m == Decimal("3.00")

    assert flash_25.input_per_1m == Decimal("0.30")
    assert flash_25.output_per_1m == Decimal("2.50")


def test_model_registry_defines_output_token_limits():
    for model in MODEL_REGISTRY.values():
        assert model.max_output_tokens > 0