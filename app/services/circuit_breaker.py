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

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if (
                self.opened_at is not None
                and time.monotonic() - self.opened_at
                >= self.recovery_timeout
            ):
                self.state = CircuitState.HALF_OPEN
                return True

            return False

        # HALF_OPEN
        return True

    def record_success(self) -> None:
        self.failure_count = 0
        self.opened_at = None
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self.failure_count += 1

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = time.monotonic()
            
circuit_breaker = CircuitBreaker()