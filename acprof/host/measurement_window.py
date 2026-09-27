"""Monitor ownership and ordered shutdown, without hardware or workload imports."""
from __future__ import annotations

from contextlib import ExitStack
from typing import Any


class MonitorCleanupError(RuntimeError):
    """A window cannot be followed by another window after cleanup failed."""


class MonitorGroup:
    """Keep the existing sampling order and attempt every stop/close on failure."""

    START_ORDER = ("gpu", "cpu", "resource", "mips")
    STOP_ORDER = ("mips", "resource", "gpu", "cpu")

    def __init__(self, *, close: bool = True):
        self.monitors: dict[str, Any] = {}
        self.results: dict[str, Any] = {}
        self.failures: list[tuple[str, BaseException]] = []
        self.started: set[str] = set()
        self._closers = ExitStack()
        self._owns_monitors = close

    def add(self, name: str, monitor: Any) -> None:
        if monitor is None:
            return
        if name not in self.START_ORDER or name in self.monitors:
            raise ValueError(f"invalid or duplicate monitor: {name}")
        self.monitors[name] = monitor
        if self._owns_monitors:
            self._closers.callback(self._attempt, f"{name}.close", monitor.close)

    def _attempt(self, operation, callback, *args):
        try:
            return callback(*args)
        except BaseException as error:
            self.failures.append((operation, error))
            return None

    def start(self) -> None:
        for name in self.START_ORDER:
            if name in self.monitors:
                # A start implementation may acquire resources before raising.
                self.started.add(name)
                self.monitors[name].start()

    def finish(self, repeat_count: int, latency_app_s: float) -> None:
        try:
            for name in self.STOP_ORDER:
                if name in self.started:
                    self.started.remove(name)
                    args = (repeat_count, latency_app_s) if name == "mips" else ()
                    self.results[name] = self._attempt(f"{name}.stop", self.monitors[name].stop, *args)
        finally:
            self._closers.close()

    @property
    def error(self) -> str:
        return "; ".join(f"{operation}: {type(error).__name__}: {error}"
                         for operation, error in self.failures)

    def raise_if_failed(self) -> None:
        if self.failures:
            # Preserve cancellation after all other monitors have been cleaned up.
            for _, error in self.failures:
                if not isinstance(error, Exception):
                    raise error
            raise MonitorCleanupError(f"monitor cleanup failed: {self.error}") from self.failures[0][1]
