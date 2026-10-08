import time
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.opened_at: float | None = None

        # Only one recovery probe may be in flight in HALF_OPEN.
        self._probe_in_flight = False

    def allow_request(self) -> bool:
        """
        Return True when an upstream request is allowed.

        CLOSED:
            All requests are allowed.

        OPEN:
            Requests are rejected until recovery_timeout has elapsed.
            After the timeout, the circuit transitions to HALF_OPEN and
            exactly one recovery probe is allowed.

        HALF_OPEN:
            Only the recovery probe is allowed. Additional requests are
            rejected until the probe succeeds or fails.
        """
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if (
                self.opened_at is not None
                and time.monotonic() - self.opened_at >= self.recovery_timeout
            ):
                self.state = CircuitState.HALF_OPEN
                self._probe_in_flight = True
                return True

            return False

        # HALF_OPEN
        if self._probe_in_flight:
            return False

        self._probe_in_flight = True
        return True

    def record_success(self) -> None:
        """
        Record a successful upstream call.

        A successful HALF_OPEN probe closes the circuit.
        """
        self.failure_count = 0
        self.opened_at = None
        self._probe_in_flight = False
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """
        Record an upstream failure.

        When the failure threshold is reached, the circuit opens.
        A failed HALF_OPEN probe therefore reopens the circuit.
        """
        self.failure_count += 1

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = time.monotonic()
            self._probe_in_flight = False


circuit_breaker = CircuitBreaker()