"""Observe shutdown skew without changing collection order or sampled values."""
import json
import math
from unittest.mock import patch

import pytest

from acprof.host.measurement_window import (
    MonitorCleanupError,
    MonitorGroup,
    run_matched_control_window,
)
from acprof.host.window_boundary_diagnostics import WindowBoundaryDiagnostics


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Monitor:
    def __init__(self, name, clock, events, *, delay=0.0, failure=None, close_failure=None):
        self.name, self.clock, self.events = name, clock, events
        self.delay, self.failure, self.close_failure = delay, failure, close_failure
        self.start_at = self.end_at = None
        self.result = (object(), "", [(0, 1), (1, 1)])
        self.snapshot_calls = 0

    def start(self):
        self.events.append(self.name + ".start")
        self.start_at = self.clock()
        self.end_at = None

    def stop(self, *_args):
        self.events.append(self.name + ".stop")
        self.end_at = self.clock()
        self.clock.now += self.delay
        if self.failure is not None:
            raise self.failure
        return self.result

    def close(self):
        self.events.append(self.name + ".close")
        if self.close_failure is not None:
            raise self.close_failure

    def sampling_boundary_snapshot(self):
        self.snapshot_calls += 1
        return {"start_monotonic_s": self.start_at, "end_monotonic_s": self.end_at,
                "semantics": "monitor_recorded_timestamps", "counter_read_instants": "unknown"}


def test_delayed_stop_exposes_other_monitor_tail_without_changing_results():
    clock, events = Clock(), []
    recorder = WindowBoundaryDiagnostics(clock=clock)
    group = MonitorGroup(diagnostics=recorder)
    delays = {"mips": 0.4, "resource": 0.2, "gpu": 0.1, "cpu": 0.0}
    monitors = {name: Monitor(name, clock, events, delay=delays[name]) for name in group.START_ORDER}
    for name, monitor in monitors.items():
        group.add(name, monitor)
    group.start()
    recorder.requests_started()
    clock.now += 1.0
    recorder.requests_finished(1)
    with pytest.raises(RuntimeError, match="finish"):
        recorder.report()
    group.finish(1, 1.0)
    report = recorder.report()
    assert events[:4] == [name + ".start" for name in group.START_ORDER]
    assert events[4:8] == [name + ".stop" for name in group.STOP_ORDER]
    assert report["request_window"]["duration_s"] == 1.0
    assert report["monitors"]["gpu"]["tail_after_requests_s"] == pytest.approx(0.6)
    assert report["monitors"]["cpu"]["tail_after_requests_s"] == pytest.approx(0.7)
    assert next(item for item in report["operations"] if item["operation"] == "mips.stop")["duration_s"] == pytest.approx(0.4)
    assert all(group.results[name] is monitors[name].result for name in monitors)
    assert all(monitor.snapshot_calls == 1 for monitor in monitors.values())
    assert report["kind"] == "measurement_window_boundary_diagnostic"
    assert report["successful"] is True
    json.dumps(report, allow_nan=False)


def test_disabled_diagnostics_does_not_read_clock_or_boundary_snapshots():
    clock, events = Clock(), []
    monitor = Monitor("cpu", clock, events)
    group = MonitorGroup()
    group.add("cpu", monitor)
    with patch("acprof.host.measurement_window.time.perf_counter", side_effect=AssertionError("extra clock read")):
        group.start()
        group.finish(1, 1.0)
    assert monitor.snapshot_calls == 0
    assert group.results["cpu"] is monitor.result


@pytest.mark.parametrize("error", [RuntimeError("stop failed"), KeyboardInterrupt()])
def test_stop_failure_keeps_other_boundaries_and_original_error(error):
    clock, events = Clock(), []
    recorder = WindowBoundaryDiagnostics(clock=clock)
    group = MonitorGroup(diagnostics=recorder)
    group.add("mips", Monitor("mips", clock, events, delay=0.4, failure=error))
    group.add("cpu", Monitor("cpu", clock, events))
    group.start()
    recorder.requests_started()
    clock.now += 1
    recorder.requests_finished(1)
    group.finish(1, 1.0)
    expected = KeyboardInterrupt if isinstance(error, KeyboardInterrupt) else MonitorCleanupError
    with pytest.raises(expected):
        group.raise_if_failed()
    report = recorder.report()
    assert report["successful"] is False
    assert report["monitors"]["cpu"]["tail_after_requests_s"] == pytest.approx(0.4)
    assert "cpu.close" in events
    assert any("mips.stop" in item for item in report["errors"])


def test_request_failure_and_close_failure_are_both_preserved():
    clock, events = Clock(), []
    recorder = WindowBoundaryDiagnostics(clock=clock)
    group = MonitorGroup(diagnostics=recorder)
    group.add("cpu", Monitor("cpu", clock, events, close_failure=RuntimeError("close failed")))
    group.start()
    recorder.requests_started()
    clock.now += 2.0
    recorder.requests_finished(0, error=TimeoutError("request failed"))
    group.finish(0, math.nan)
    report = recorder.report()
    assert report["request_window"]["completed_requests"] == 0
    assert report["request_window"]["error"] == "TimeoutError: request failed"
    assert report["successful"] is False
    assert any("cpu.close" in item for item in report["errors"])


def test_failed_preparation_does_not_invent_request_or_sampling_boundaries():
    clock, events = Clock(), []
    recorder = WindowBoundaryDiagnostics(clock=clock)
    group = MonitorGroup(diagnostics=recorder)
    monitor = Monitor("cpu", clock, events)
    def fail():
        raise RuntimeError("prepare failed")
    monitor.prepare = fail
    group.add("cpu", monitor)
    with pytest.raises(RuntimeError, match="prepare failed"):
        try:
            group.start()
        finally:
            recorder.requests_finished(0, error=RuntimeError("prepare failed"))
            group.finish(0, math.nan)
    report = recorder.report()
    assert report["request_window"]["start_s"] is None
    assert report["request_window"]["end_s"] is None
    assert report["monitors"]["cpu"]["boundary_status"] == "unknown"
    assert report["successful"] is False


def test_control_window_and_repeated_finish_do_not_change_request_report():
    clock, events = Clock(), []
    recorder = WindowBoundaryDiagnostics(clock=clock)
    group = MonitorGroup(diagnostics=recorder)
    group.add("resource", Monitor("resource", clock, events))
    run_matched_control_window(group, idle_seconds=0)
    group.start()
    recorder.requests_started()
    clock.now += 1
    recorder.requests_finished(1)
    group.finish(1, 1.0)
    report = recorder.report()
    assert [item["operation"] for item in report["operations"]] == ["resource.start", "resource.stop", "resource.close"]
    group.finish(1, 1.0)
    assert recorder.report() == report


def test_unknown_or_failed_snapshot_is_reported_without_masking_cleanup():
    clock, events = Clock(), []
    recorder = WindowBoundaryDiagnostics(clock=clock)
    group = MonitorGroup(diagnostics=recorder)
    monitor = Monitor("cpu", clock, events)
    def fail():
        raise RuntimeError("snapshot failed")
    monitor.sampling_boundary_snapshot = fail
    group.add("cpu", monitor)
    group.start()
    recorder.requests_started()
    clock.now += 1
    recorder.requests_finished(1)
    group.finish(1, 1.0)
    group.raise_if_failed()
    report = recorder.report()
    assert report["successful"] is False
    assert report["monitors"]["cpu"]["boundary_status"] == "unknown"
    assert any("snapshot failed" in item for item in report["errors"])
    assert events[-1] == "cpu.close"


def test_close_retry_does_not_replace_the_stop_result_boundary():
    clock, events = Clock(), []
    recorder = WindowBoundaryDiagnostics(clock=clock)
    group = MonitorGroup(diagnostics=recorder)
    monitor = Monitor("cpu", clock, events)
    def close_again():
        clock.now += 7.0
        monitor.end_at = clock()
    monitor.close = close_again
    group.add("cpu", monitor)
    group.start()
    recorder.requests_started()
    clock.now += 1
    recorder.requests_finished(1)
    group.finish(1, 1.0)
    assert recorder.report()["monitors"]["cpu"]["tail_after_requests_s"] == 0
    assert monitor.end_at == 108.0


def test_partial_start_failure_cannot_reuse_a_control_window_timestamp():
    clock, events = Clock(), []
    recorder = WindowBoundaryDiagnostics(clock=clock)
    group = MonitorGroup(diagnostics=recorder)
    monitor = Monitor("cpu", clock, events)
    monitor.start_at, monitor.end_at = 1.0, 2.0
    def fail():
        raise RuntimeError("start failed before timestamp reset")
    monitor.start = fail
    group.add("cpu", monitor)
    with pytest.raises(RuntimeError, match="start failed"):
        group.start()
    recorder.requests_finished(0, error=RuntimeError("start failed"))
    group.finish(0, math.nan)
    assert recorder.report()["monitors"]["cpu"]["boundary_status"] == "unknown"
    assert monitor.snapshot_calls == 0


@pytest.mark.parametrize("module,class_name,has_end", [
    ("energy_cpu", "CPUEnergyMonitor", True),
    ("energy_nvml", "GPUEnergyMonitor", True),
    ("resource_usage", "ResourceUsageMonitor", True),
    ("perf_mips", "PerfMIPSMonitor", False),
])
def test_real_boundary_accessors_read_only_existing_timestamps(module, class_name, has_end):
    import importlib
    cls = getattr(importlib.import_module("acprof.monitors." + module), class_name)
    monitor = cls.__new__(cls)
    monitor._t_start, monitor._t_end = 12.0, 13.0
    before = dict(monitor.__dict__)
    snapshot = monitor.sampling_boundary_snapshot()
    assert snapshot["start_monotonic_s"] == 12.0
    assert snapshot["end_monotonic_s"] == (13.0 if has_end else None)
    assert snapshot["semantics"] == "monitor_recorded_timestamps"
    assert monitor.__dict__ == before


def test_nonfinite_snapshot_is_unknown_and_json_safe():
    clock, events = Clock(), []
    recorder = WindowBoundaryDiagnostics(clock=clock)
    group = MonitorGroup(diagnostics=recorder)
    monitor = Monitor("cpu", clock, events)
    monitor.sampling_boundary_snapshot = lambda: {"start_monotonic_s": math.nan, "end_monotonic_s": None}
    group.add("cpu", monitor)
    group.start()
    recorder.requests_started()
    clock.now += 1
    recorder.requests_finished(1)
    group.finish(1, 1.0)
    report = recorder.report()
    assert report["monitors"]["cpu"]["boundary_status"] == "unknown"
    json.dumps(report, allow_nan=False)
