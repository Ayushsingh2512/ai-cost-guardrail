import time

from app.services.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
)


def test_starts_closed():
    breaker = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=30,
    )

    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_opens_after_failure_threshold():
    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=30,
    )

    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state == CircuitState.CLOSED

    breaker.record_failure()

    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_request() is False


def test_open_circuit_rejects_requests():
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=30,
    )

    breaker.record_failure()

    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_request() is False


def test_moves_to_half_open_after_recovery_timeout():
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=0.1,
    )

    breaker.record_failure()

    assert breaker.state == CircuitState.OPEN

    time.sleep(0.15)

    assert breaker.allow_request() is True
    assert breaker.state == CircuitState.HALF_OPEN


def test_success_closes_circuit():
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=0.1,
    )

    breaker.record_failure()

    assert breaker.state == CircuitState.OPEN

    time.sleep(0.15)

    assert breaker.allow_request() is True
    assert breaker.state == CircuitState.HALF_OPEN

    breaker.record_success()

    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0
    assert breaker.allow_request() is True


def test_failure_in_half_open_reopens_circuit():
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=0.1,
    )

    breaker.record_failure()

    time.sleep(0.15)

    assert breaker.allow_request() is True
    assert breaker.state == CircuitState.HALF_OPEN

    breaker.record_failure()

    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_request() is False