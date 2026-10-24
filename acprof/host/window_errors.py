"""Fatal measurement failures shared by the request adapter and window executor."""
from __future__ import annotations


class EnergyAbort(RuntimeError):
    """Raised when energy measurement prerequisites are not stable enough."""

class MIPSAbort(RuntimeError):
    """Raised when perf MIPS profiling cannot continue."""

class RequestTimeoutAbort(RuntimeError):
    """Raised when the current resource case cannot finish inference in time."""

    def __init__(
        self,
        message: str,
        *,
        input_scale: float | None = None,
        request_id: str = "",
        timeout_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.input_scale = input_scale
        self.request_id = request_id
        self.timeout_s = timeout_s
