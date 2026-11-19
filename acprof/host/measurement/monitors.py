"""Monitor ownership and ordered shutdown, without hardware or workload imports."""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, TypeVar, cast

from acprof.monitors.interfaces import (
    CPUStopResult,
    GPUStopResult,
    MonitorLifecycle,
    ResourceStopResult,
)

if TYPE_CHECKING:
    from acprof.host.measurement.boundary import WindowBoundaryDiagnostics
    from acprof.monitors.perf_mips import MIPSResult

Result = TypeVar("Result")


class MonitorCleanupError(RuntimeError):
    """A window cannot be followed by another window after cleanup failed."""


class MonitorGroup:
    """Keep the existing sampling order and attempt every stop/close on failure."""

    START_ORDER = ("gpu", "cpu", "resource", "mips")
    STOP_ORDER = ("mips", "resource", "gpu", "cpu")

    def __init__(self, *, close: bool = True, optional: bool = False, diagnostics: WindowBoundaryDiagnostics | None = None):
        self.optional = optional
        self.disabled: set[str] = set()
        self.phases: dict[str, list[str]] = {}
        self.diagnostics = diagnostics
        self.monitors: dict[str, MonitorLifecycle] = {}
        self.results: dict[str, CPUStopResult | GPUStopResult | ResourceStopResult | MIPSResult | None] = {}
        self.failures: list[tuple[str, BaseException]] = []
        self.started: set[str] = set()
        self._closers = ExitStack()
        self._owns_monitors = close
        self._prepared = False

    def add(self, name: str, monitor: MonitorLifecycle | None) -> None:
        if monitor is None:
            return
        if name not in self.START_ORDER or name in self.monitors:
            raise ValueError(f"invalid or duplicate monitor: {name}")
        self.monitors[name] = monitor
        self._prepared = False
        if self._owns_monitors:
            self._closers.callback(self._attempt, f"{name}.close", monitor.close)

    def _attempt(self, operation: str, callback: Callable[..., Result], *args: Any) -> Result | None:
        name, phase = operation.split('.', 1)
        self.phases.setdefault(name, []).append(phase)
        try:
            if self.diagnostics is not None:
                return self.diagnostics.observe(operation, callback, *args)
            return callback(*args)
        except BaseException as error:
            self.failures.append((operation, error))
            return None

    def invoke(self, name, phase, callback, *args, **kwargs):
        """An optional acquisition failure never takes ownership away from close()."""
        self.phases.setdefault(name, []).append(phase)
        try:
            return callback(*args, **kwargs)
        except Exception as exc:
            if not self.optional:
                raise
            self.disabled.add(name)
            self.failures.append((f"{name}.{phase}", exc))
            return None

    def construct(self, name, factory, *args, **kwargs):
        def build():
            if factory is None:
                raise ModuleNotFoundError(f"{name} collector is unavailable")
            return factory(*args, **kwargs)
        monitor = self.invoke(name, "construct", build)
        self.add(name, monitor)
        return monitor

    def prepare(self) -> None:
        """Resolve every prerequisite before the first collector starts sampling."""
        if self._prepared:
            return
        for name in self.START_ORDER:
            prepare = getattr(self.monitors.get(name), "prepare", None)
            if prepare is not None and name not in self.disabled:
                self.invoke(name, "prepare", prepare)
        self._prepared = True

    def start(self) -> None:
        self.prepare()
        for name in self.START_ORDER:
            if name in self.monitors and name not in self.disabled:
                # A start implementation may acquire resources before raising.
                self.started.add(name)
                if self.diagnostics is None:
                    self.invoke(name, "start", self.monitors[name].start)
                else:
                    self.invoke(name, "start", self.diagnostics.observe, f"{name}.start", self.monitors[name].start)

    def finish(self, repeat_count: int, latency_app_s: float) -> None:
        try:
            for name in self.STOP_ORDER:
                if name in self.started:
                    self.started.remove(name)
                    args = (repeat_count, latency_app_s) if name == "mips" else ()
                    result = self._attempt(f"{name}.stop", getattr(self.monitors[name], "stop"), *args)
                    if name not in self.disabled:
                        self.results[name] = result
        finally:
            self._prepared = False
            try:
                if self.diagnostics is not None:
                    self.diagnostics.capture_boundaries(self.monitors)
            finally:
                try:
                    self._closers.close()
                finally:
                    if self.diagnostics is not None:
                        self.diagnostics.finish(self.failures)

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
            if self.optional and not any(op.endswith(".close") for op, _ in self.failures):
                return
            raise MonitorCleanupError(f"monitor cleanup failed: {self.error}") from self.failures[0][1]


def run_matched_control_window(monitors: MonitorGroup, *, idle_seconds: float,
                               trace: bool = False) -> None:
    """Borrow collectors for a blank window; their owner retains close responsibility."""
    if not math.isfinite(idle_seconds) or idle_seconds < 0:
        raise ValueError("idle_seconds must be finite and nonnegative")
    group = MonitorGroup(close=False, optional=monitors.optional)
    # Idle baseline measures energy only. Resource and perf monitors belong
    # exclusively to request windows; attaching them here contaminates control.
    for name in ("gpu", "cpu"):
        if name in monitors.monitors:
            group.add(name, monitors.monitors[name])
    primary_error = None
    try:
        group.start()
        if idle_seconds > 0:
            time.sleep(idle_seconds)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        group.finish(1, max(idle_seconds, 1e-9))
        if primary_error is None or isinstance(primary_error, Exception):
            group.raise_if_failed()
    for name in ("gpu", "cpu"):
        values = cast(CPUStopResult | GPUStopResult | None, group.results.get(name))
        if values is not None and name not in group.disabled:
            if monitors.optional and values[-2]:
                group.failures.append((f"{name}.baseline", RuntimeError(str(values[-2]))))
            else:
                group.invoke(name, "baseline", getattr(group.monitors[name], "apply_control_baseline"),
                             values[0], values[-1], trace=trace)
    for operation, failure in group.failures:
        name, phase = operation.split(".", 1)
        monitors.failures.append((f"{name}.control_{phase}", failure))
        # A failed control start may leave unsafe sampler state. A rejected
        # baseline alone still permits an independent gross workload reading.
        if phase != "baseline":
            monitors.disabled.add(name)
    for name, phases in group.phases.items():
        monitors.phases.setdefault(name, []).extend("control_" + phase for phase in phases)
