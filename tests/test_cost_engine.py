from decimal import Decimal

import pytest

from app.services.cost_engine import cost_engine


def test_estimate_reservation():
    result = cost_engine.estimate_reservation(
        model="gemini-3-flash-preview",
        input_tokens=500,
        max_output_tokens=1000,
    )

    assert result == Decimal("0.00325")
def test_estimate_embedding_cost():
    result = cost_engine.estimate_embedding_cost(
        model="gemini-embedding-001",
        input_tokens=1000,
    )

    assert result == Decimal("0.000150")


def test_estimate_embedding_cost_zero_tokens():
    result = cost_engine.estimate_embedding_cost(
        model="gemini-embedding-001",
        input_tokens=0,
    )

    assert result == Decimal("0")


def test_estimate_embedding_cost_unknown_model():
    with pytest.raises(
        ValueError,
        match="No embedding pricing configuration exists",
    ):
        cost_engine.estimate_embedding_cost(
            model="unknown-embedding-model",
            input_tokens=1000,
        )


def test_calculate_embedding_cost():
    result = cost_engine.calculate_embedding_cost(
        model="gemini-embedding-001",
        input_tokens=2000,
    )

    assert result == Decimal("0.000300")

def test_reservation_accounts_for_large_input():
    result = cost_engine.estimate_reservation(
        model="gemini-3-flash-preview",
        input_tokens=5000,
        max_output_tokens=1,
    )

    expected = (
        Decimal("0.0025")  # 5000 input tokens
        + Decimal("0.000003")  # 1 output token
    )

    assert result == expected


def test_calculate_actual_cost():
    result = cost_engine.calculate_actual_cost(
        model="gemini-3-flash-preview",
        input_tokens=500,
        thinking_tokens=0,
        output_tokens=1000,
    )

    assert result == Decimal("0.00325")


def test_calculate_actual_cost_includes_thinking_tokens():
    result = cost_engine.calculate_actual_cost(
        model="gemini-3-flash-preview",
        input_tokens=500,
        thinking_tokens=200,
        output_tokens=1000,
    )

    assert result == Decimal("0.00385")


def test_zero_tokens_cost_nothing():
    result = cost_engine.calculate_actual_cost(
        model="gemini-3-flash-preview",
        input_tokens=0,
        thinking_tokens=0,
        output_tokens=0,
    )

    assert result == Decimal("0")


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="No pricing configuration"):
        cost_engine.estimate_reservation(
            model="unknown-model",
            input_tokens=500,
            max_output_tokens=1000,
        )