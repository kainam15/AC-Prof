"""Opt-in timing evidence; never sample hardware or publish inside a window."""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

_Result = TypeVar("_Result")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _error(error: BaseException | None) -> str | None:
    return f"{type(error).__name__}: {error}" if error is not None else None


class WindowBoundaryDiagnostics:
    """One request window, finalized by MonitorGroup after all cleanup attempts.

    Operation timings include the wrapped call only. Monitor timestamps describe
    their existing logical boundaries, not the exact instant of a hardware read.
    Every method is opt-in; the recorder never writes files or reads hardware.
    """

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter):
        self._clock = clock
        self._origin: float | None = None
        self._operations: list[tuple[str, float, float, BaseException | None]] = []
        self._request_start: float | None = None
        self._request_end: float | None = None
        self._request_error: BaseException | None = None
        self._requests_closed = False
        self._completed_requests = 0
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._diagnostic_errors: list[str] = []
        self._cleanup_errors: list[tuple[str, BaseException]] = []
        self._finished = False

    def _timestamp(self) -> float:
        timestamp = self._clock()
        if self._origin is None:
            self._origin = timestamp
        return timestamp

    def observe(self, operation: str, callback: Callable[..., _Result], *args: Any) -> _Result:
        """Time a start, stop or close while preserving its exact return/exception."""
        if self._finished:
            raise RuntimeError("diagnostic window is already finished")
        before = self._timestamp()
        failure = None
        try:
            return callback(*args)
        except BaseException as error:
            failure = error
            raise
        finally:
            self._operations.append((operation, before, self._clock(), failure))

    def requests_started(self) -> None:
        if self._finished or self._request_start is not None or self._requests_closed:
            raise RuntimeError("request window cannot be started twice")
        self._request_start = self._timestamp()

    def requests_finished(self, completed_requests: int, error: BaseException | None = None) -> None:
        if self._finished or self._requests_closed:
            raise RuntimeError("request window is already finished")
        if isinstance(completed_requests, bool) or not isinstance(completed_requests, int) or completed_requests < 0:
            raise ValueError("completed_requests must be a nonnegative integer")
        # A preparation/start failure must not invent an observed request window.
        if self._request_start is not None:
            self._request_end = self._timestamp()
        self._completed_requests = completed_requests
        self._request_error = error
        self._requests_closed = True

    def capture_boundaries(self, monitors: Mapping[str, Any]) -> None:
        """Copy existing timestamps after all stops, before close can retry a stop."""
        if self._finished:
            return
        starts = {operation: error for operation, _, _, error in self._operations if operation.endswith(".start")}
        for name, monitor in monitors.items():
            snapshot: dict[str, Any] = {"reason": "monitor_not_started"}
            if f"{name}.start" in starts:
                if starts[f"{name}.start"] is not None:
                    snapshot = {"reason": "monitor_start_failed; timestamps may belong to an earlier control window"}
                else:
                    try:
                        read = getattr(monitor, "sampling_boundary_snapshot", None)
                        value = read() if callable(read) else None
                        if value is None:
                            snapshot = {"reason": "monitor_does_not_expose_recorded_boundaries"}
                        elif not isinstance(value, dict):
                            raise TypeError("sampling boundary snapshot must be a dictionary")
                        else:
                            snapshot = dict(value)
                    except Exception as error:
                        detail = f"{name}.sampling_boundary_snapshot: {_error(error)}"
                        self._diagnostic_errors.append(detail)
                        snapshot = {"reason": detail}
            self._snapshots[name] = snapshot

    def finish(self, failures: list[tuple[str, BaseException]]) -> None:
        """Freeze once; repeated MonitorGroup.finish calls preserve the report."""
        if not self._finished:
            self._cleanup_errors = list(failures)
            self._finished = True

    def report(self) -> dict[str, Any]:
        if not self._finished:
            raise RuntimeError("MonitorGroup.finish must complete before a diagnostic report is read")
        origin = self._origin

        def relative(value: Any) -> float | None:
            number = _number(value)
            return number - origin if number is not None and origin is not None else None

        request_start, request_end = self._request_start, self._request_end
        monitors = {}
        for name, snapshot in self._snapshots.items():
            start = _number(snapshot.get("start_monotonic_s"))
            end = _number(snapshot.get("end_monotonic_s"))
            valid_duration = start is not None and end is not None and end >= start
            monitors[name] = {
                "boundary_status": "recorded" if valid_duration else "partial" if start is not None or end is not None else "unknown",
                "start_s": relative(start), "end_s": relative(end),
                "duration_s": end - start if start is not None and end is not None and end >= start else None,
                "lead_before_requests_s": request_start - start if start is not None and request_start is not None else None,
                "tail_after_requests_s": end - request_end if end is not None and request_end is not None else None,
                "semantics": snapshot.get("semantics", "unknown"),
                "counter_read_instants": "unknown",
                "reason": snapshot.get("reason", ""),
            }
        errors = [f"{operation}: {_error(error)}" for operation, error in self._cleanup_errors]
        errors.extend(self._diagnostic_errors)
        request_observed = request_start is not None and request_end is not None
        return {
            "schema_version": 1, "kind": "measurement_window_boundary_diagnostic",
            "clock": "time.perf_counter", "unit": "seconds", "origin": "first_recorded_diagnostic_event",
            "successful": request_observed and self._request_error is None and not errors,
            "request_window": {
                "status": "recorded" if request_observed else "not_started" if request_start is None else "incomplete",
                "start_s": relative(request_start), "end_s": relative(request_end),
                "duration_s": request_end - request_start if request_start is not None and request_end is not None else None,
                "completed_requests": self._completed_requests, "error": _error(self._request_error),
            },
            "operations": [{"operation": operation, "start_s": relative(before), "end_s": relative(after),
                            "duration_s": after - before, "error": _error(error)}
                           for operation, before, after, error in self._operations],
            "monitors": monitors, "errors": errors,
            "limitations": [
                "Opt-in diagnostic timing and recording add overhead; this is not a formal profiling result.",
                "Monitor boundaries are existing logical timestamps, not exact counter-read instants or a shared cutoff.",
                "All boundary snapshots are copied after stop attempts; reports are available only after close attempts.",
                "Matched-control windows are not recorded by this request-window diagnostic.",
                "Shutdown skew does not by itself establish a significant energy bias; native validation is required.",
            ],
        }
