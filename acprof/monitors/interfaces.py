"""Structural monitor contracts; ownership and prerequisite discovery are separate."""
from __future__ import annotations

from typing import Protocol, TypeVar

Result = TypeVar("Result", covariant=True)


class MonitorLifecycle(Protocol):
    def start(self) -> None: ...
    def close(self) -> None: ...


class PreparedMonitor(Protocol):
    def prepare(self) -> None: ...


class SampleMonitor(MonitorLifecycle, Protocol[Result]):
    def stop(self) -> Result: ...


class InstructionMonitor(MonitorLifecycle, Protocol[Result]):
    def stop(self, repeat_in_window: int, latency_app_s: float) -> Result: ...
