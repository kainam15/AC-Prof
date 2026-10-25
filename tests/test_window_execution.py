"""Observe the public window boundary with real journals and replaceable collectors."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from acprof.host.measurement.config import ClientConfig
from acprof.host.measurement.errors import RequestTimeoutAbort
from acprof.host.measurement.execution import WindowExecutor
from acprof.host.measurement.monitors import MonitorCleanupError
from acprof.host.measurement.types import MonitorDependencies, WindowSpec
from acprof.host.profiling.execution_plan import ExecutionProfileResolver
from acprof.monitors import resource_metrics


class Collector:
    def __init__(self, events, *, fault=None, warning=""):
        self.events = events
        self.fault = fault
        self.warning = warning
        self.stop_calls = 0
        self.close_calls = 0

    def prepare(self):
        self.events.append("prepare")
        if self.fault == "prepare":
            raise RuntimeError("prepare failed")

    def start(self):
        self.events.append("start")
        if self.fault == "start":
            raise RuntimeError("start failed")

    def stop(self):
        self.stop_calls += 1
        self.events.append("stop")
        if self.fault == "stop":
            raise RuntimeError("stop failed")
        if self.fault == "malformed_result":
            return (None, "missing sample list")
        result = resource_metrics._nan_result()
        result.container_cpu_util_avg_pct = 25.0
        result.container_mem_usage_avg_bytes = 1024.0
        return result, self.warning, []

    def close(self):
        self.close_calls += 1
        self.events.append("close")
        if self.fault == "close":
            raise RuntimeError("close failed")


class Journal:
    """Check every filesystem write occurs after collector shutdown."""

    def __init__(self, stream, events, *, fail=False):
        self.stream = stream
        self.events = events
        self.fail = fail

    def write(self, value):
        assert "close" in self.events
        self.events.append("journal")
        if isinstance(self.fail, BaseException):
            raise self.fail
        if self.fail:
            raise OSError("journal failed")
        return self.stream.write(value)

    def flush(self):
        return self.stream.flush()

    def fileno(self):
        return self.stream.fileno()


def executor(collector, request, *, config=None, dependencies=None):
    config = config or ClientConfig(
        case_name="case", cpu_cores="2", mem_cap_gb="4",
        batch_size=2, repeat_in_window=2, idle_seconds=0.0,
        idle_cooldown_seconds=0.0, use_mips=False,
    )
    return WindowExecutor(
        config,
        dependencies or MonitorDependencies(
            resource=lambda **kw: collector,
        ),
        one_request=request,
        cold_start_metrics=lambda: {"cold_start_first_predict_app_s": "0.100000"},
        environment_class="native_linux",
        compute_profile_plan={"_disabled": True},
        execution_profiles=ExecutionProfileResolver({"_disabled": True}),
        use_energy=config.gpu_mode == "on",
        runtime_failures=[],
    )


def spec():
    return WindowSpec(
        {"input_scale": 10.0, "scale_label": "scale10", "payload": {"text": "fixed"}},
        request_limit=2, warmup=0, repeat_idx=0,
    )


def responses(events, *, error=None):
    def request(scale, req_id, payload_override):
        assert events[-1] in {"start", "request"}
        events.append("request")
        if req_id.endswith(":1") and error is not None:
            raise error
        return {
            "latency_app_s": 0.1 if req_id.endswith(":0") else 0.3,
            "effective_input_scale": scale,
            "request_payload_bytes": 32,
            "output_length": 4,
            "output_token_count": 5,
            "task_param": '{"text":"fixed"}',
        }
    return request


def test_completed_window_returns_row_and_writes_requests_after_shutdown(tmp_path):
    events = []
    collector = Collector(events)
    window = executor(collector, responses(events))
    path = tmp_path / "requests.jsonl"
    with path.open("w") as stream:
        result = window.execute(spec(), Journal(stream, events))
    assert events == ["prepare", "start", "request", "request", "stop", "close", "journal"]
    assert collector.stop_calls == collector.close_calls == 1
    assert result.sniff_group_id == "case_scale10_r0"
    assert result.idle_diag_record is None
    assert result.row["status"] == "ok"
    assert result.row["repeat_in_window"] == "2"
    assert float(result.row["latency_app_s"]) == pytest.approx(0.2)
    assert float(result.row["throughput_samples_per_s"]) == pytest.approx(10.0)
    assert result.row["gpu_energy_eff_j"] == "nan"
    record = json.loads(path.read_text())
    assert record["latency_app_s"] == [0.1, 0.3]
    assert record["input_scale"] == 10.0
    assert record["status"] == "ok"


def test_failed_window_retains_completed_requests_without_success_metrics(tmp_path):
    events = []
    collector = Collector(events)
    window = executor(collector, responses(events, error=RuntimeError("inference failed")))
    path = tmp_path / "requests.jsonl"
    with path.open("w") as stream:
        result = window.execute(spec(), Journal(stream, events))
    assert collector.stop_calls == collector.close_calls == 1
    assert result.row["status"] == "error"
    assert "inference failed" in result.row["error"]
    assert result.row["repeat_in_window"] == "1"
    assert result.row["latency_app_s"] == "nan"
    assert result.row["throughput_samples_per_s"] == "nan"
    record = json.loads(path.read_text())
    assert record["latency_app_s"] == [0.1]
    assert record["failed_request_id"] == "case_scale10_r0:1"
    assert "inference failed" in record["error"]


@pytest.mark.parametrize("primary", ["timeout", "cancel", "ordinary", "none"])
@pytest.mark.parametrize("journal_fails", [False, True])
def test_primary_abort_and_cleanup_priorities_survive_journal_failure(tmp_path, primary, journal_fails):
    events = []
    collector = Collector(events, fault="stop")
    errors = {
        "timeout": RequestTimeoutAbort("request timed out", request_id="case_scale10_r0:1"),
        "cancel": KeyboardInterrupt(),
        "ordinary": RuntimeError("inference failed"),
        "none": None,
    }
    error = errors[primary]
    window = executor(collector, responses(events, error=error))
    expected = type(error) if primary in {"timeout", "cancel"} else MonitorCleanupError
    with (tmp_path / "requests.jsonl").open("w") as stream, pytest.raises(expected) as raised:
        window.execute(spec(), Journal(stream, events, fail=journal_fails))
    if primary in {"timeout", "cancel"}:
        assert raised.value is error
    else:
        assert "stop failed" in str(raised.value)
    assert collector.stop_calls == collector.close_calls == 1


@pytest.mark.parametrize("fault, expected_stops", [("prepare", 0), ("start", 1)])
def test_startup_failure_closes_owned_collectors_without_sending_requests(tmp_path, fault, expected_stops):
    events = []
    collector = Collector(events, fault=fault)
    calls = []

    def request(*args, **kwargs):
        calls.append(args)
        raise AssertionError("no request may precede collector startup")

    window = executor(collector, request)
    path = tmp_path / "requests.jsonl"
    with path.open("w") as stream:
        result = window.execute(spec(), Journal(stream, events))
    assert not calls
    assert collector.stop_calls == expected_stops
    assert collector.close_calls == 1
    assert result.row["status"] == "error"
    assert f"{fault} failed" in result.row["error"]
    assert json.loads(path.read_text())["latency_app_s"] == []


def test_later_constructor_failure_closes_already_owned_collector(tmp_path):
    events = []
    gpu = SimpleNamespace(close=lambda: events.append("close"))

    def fail_cpu(**kwargs):
        raise RuntimeError("CPU collector construction failed")

    dependencies = MonitorDependencies(
        gpu=lambda **kwargs: gpu,
        cpu=fail_cpu,
    )
    window = executor(None, lambda *args, **kwargs: pytest.fail("request was sent"),
                      config=ClientConfig(case_name="case", gpu_mode="on", repeat_in_window=2),
                      dependencies=dependencies)
    path = tmp_path / "requests.jsonl"
    with path.open("w") as stream:
        result = window.execute(spec(), Journal(stream, events))
    assert events == ["close", "journal"]
    assert result.row["status"] == "error"
    assert "construction failed" in result.row["error"]


@pytest.mark.parametrize("mode, expected", [("full", "warn"), ("basic", "error")])
def test_collector_unavailability_keeps_mode_specific_result_status(tmp_path, mode, expected):
    events = []
    collector = Collector(events, warning="resource samples unavailable")
    config = ClientConfig(
        case_name="case", profiling_mode=mode, repeat_in_window=2,
        idle_seconds=0, idle_cooldown_seconds=0,
    )
    window = executor(collector, responses(events), config=config)
    with (tmp_path / "requests.jsonl").open("w") as stream:
        result = window.execute(spec(), Journal(stream, events))
    assert result.row["status"] == expected
    assert "resource samples unavailable" in result.row["error"]
    assert float(result.row["latency_app_s"]) == pytest.approx(0.2)
    assert result.row["gpu_energy_eff_j"] == "nan"


def test_journal_failure_becomes_error_row_after_successful_cleanup(tmp_path):
    events = []
    collector = Collector(events)
    window = executor(collector, responses(events))
    with (tmp_path / "requests.jsonl").open("w") as stream:
        result = window.execute(spec(), Journal(stream, events, fail=True))
    assert collector.stop_calls == collector.close_calls == 1
    assert result.row["status"] == "error"
    assert "journal failed" in result.row["error"]
    assert result.row["repeat_in_window"] == "2"
    assert result.row["latency_app_s"] == "nan"

def test_reduction_failure_keeps_pre_reduction_scale_and_partial_latency(tmp_path):
    events = []
    collector = Collector(events, warning="resource samples unavailable")
    config = ClientConfig(
        case_name="case", profiling_mode="basic", repeat_in_window=2,
        idle_seconds=0, idle_cooldown_seconds=0,
    )
    request = responses(events)

    def effective_request(scale, req_id, payload_override):
        response = request(scale, req_id, payload_override)
        response["effective_input_scale"] = 20.0
        return response

    window = executor(collector, effective_request, config=config)
    path = tmp_path / "requests.jsonl"
    with path.open("w") as stream:
        result = window.execute(spec(), Journal(stream, events))
    assert result.row["status"] == "error"
    assert float(result.row["latency_app_s"]) == pytest.approx(0.2)
    assert result.row["input_scale"] == "10.0"
    assert result.row["throughput_samples_per_s"] == "nan"
    assert json.loads(path.read_text())["input_scale"] == 20.0

@pytest.mark.parametrize("primary", [RequestTimeoutAbort("timed out"), KeyboardInterrupt()])
def test_primary_abort_survives_invalid_collector_result(tmp_path, primary):
    events = []
    collector = Collector(events, fault="malformed_result")
    window = executor(collector, responses(events, error=primary))
    with (tmp_path / "requests.jsonl").open("w") as stream, pytest.raises(type(primary)) as caught:
        window.execute(spec(), Journal(stream, events))
    assert caught.value is primary
    assert collector.stop_calls == collector.close_calls == 1
    assert "journal" not in events


def test_journal_cancellation_survives_recorded_cleanup_failure(tmp_path):
    events = []
    collector = Collector(events, fault="stop")
    window = executor(collector, responses(events))
    cancellation = KeyboardInterrupt()
    with (tmp_path / "requests.jsonl").open("w") as stream, pytest.raises(KeyboardInterrupt) as caught:
        window.execute(spec(), Journal(stream, events, fail=cancellation))
    assert caught.value is cancellation
    assert collector.stop_calls == collector.close_calls == 1
