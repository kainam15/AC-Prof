"""开销对照必须按同一轮配对，不能混用不完整实验。"""
from unittest.mock import Mock, patch

import pytest

from acprof.host.measurement_window import MonitorGroup


def group(*monitors, perf=None):
    result = MonitorGroup()
    for name, monitor in zip(("cpu", "resource", "gpu"), monitors):
        result.add(name, monitor)
    result.add("mips", perf)
    return result


def test_known_paired_increase_and_zero_change():
    from scripts.measure_overhead import summarize_overhead
    rows = []
    for index, latency in enumerate((1, 2, 4)):
        rows.extend(({"round": index, "scenario": "none", "latency_app_s": latency},
                     {"round": index, "scenario": "monitors-20", "latency_app_s": latency * 1.25}))
    result = summarize_overhead(rows, seed=7)
    assert (result[0]["paired_mean_change_pct"]) == (25)
    assert ((result[0]["ci_low_pct"], result[0]["ci_high_pct"])) == ((25, 25))
    assert (result[0]["paired_mean_change_s"]) == (7 / 12) or round(abs((result[0]["paired_mean_change_s"]) - (7 / 12)), 7) == 0
    assert (result[0]["latency_mean_s"]) == (35 / 12) or round(abs((result[0]["latency_mean_s"]) - (35 / 12)), 7) == 0
    assert (result[0]["latency_stdev_s"]) > (0)

@pytest.mark.parametrize('status,body', ((202, {'status': 'pending'}), (200, {'error': 'background failure'})))
def test_failed_or_unfinished_response_is_not_a_latency_sample(status, body):
    from scripts.measure_overhead import measure_window
    response = Mock(status_code=status)
    response.json.return_value = body
    with patch("requests.post", return_value=response):
        with pytest.raises(RuntimeError, match="completed"):
            measure_window("http://example.invalid", {}, count=1, monitors=group(), token="test")

def test_source_thread_settings_are_restored_without_inheriting_unrecorded_overrides():
    import os

    from acprof.host.execution_conditions import source_runtime_environment
    with patch.dict(os.environ, {"ACPROF_RUNTIME_THREADS": "8", "ACPROF_ONNX_INTRA_OP_THREADS": "6",
                                 "ACPROF_GPU_DEVICE": "GPU-caller"}):
        with source_runtime_environment({"ACPROF_RUNTIME_THREADS": "1"}):
            assert (os.environ["ACPROF_RUNTIME_THREADS"]) == ("1")
            assert ("ACPROF_ONNX_INTRA_OP_THREADS") not in (os.environ)
            assert (os.environ.get("ACPROF_GPU_DEVICE")) is None
        assert (os.environ["ACPROF_RUNTIME_THREADS"]) == ("8")
        assert (os.environ["ACPROF_ONNX_INTRA_OP_THREADS"]) == ("6")
        assert (os.environ["ACPROF_GPU_DEVICE"]) == ("GPU-caller")

def test_missing_baseline_duplicate_and_bad_values_are_rejected():
    from scripts.measure_overhead import summarize_overhead
    samples = [
        [{"round": 0, "scenario": "monitors-20", "latency_app_s": 2}],
        [{"round": 0, "scenario": "none", "latency_app_s": 1}] * 2,
        [{"round": 0, "scenario": "none", "latency_app_s": 0}],
    ]
    for rows in samples:
        with pytest.raises(ValueError):
            summarize_overhead(rows)

def test_request_failure_stops_and_closes_every_started_monitor():
    from scripts.measure_overhead import measure_window
    monitors = [Mock(), Mock()]
    for monitor in monitors:
        monitor.stop.return_value = (None, "", [1, 2])
    with patch("requests.post", side_effect=RuntimeError("request failed")):
        with pytest.raises(RuntimeError, match="request failed"):
            measure_window("http://example.invalid", {}, count=1, monitors=group(*monitors), token="test")
    for monitor in monitors:
        monitor.stop.assert_called_once()
        monitor.close.assert_called_once()

def test_perf_failure_still_stops_sampling_threads():
    from scripts.measure_overhead import measure_window
    monitor, perf = Mock(), Mock()
    monitor.stop.return_value = (None, "", [1, 2])
    perf.stop.side_effect = RuntimeError("instructions unavailable")
    response = Mock(status_code=200)
    response.json.return_value = {"workload_contract": {"input": {"actual_scale": 5}}}
    with patch("requests.post", return_value=response) as post:
        with pytest.raises(RuntimeError, match="instructions unavailable"):
            measure_window("http://example.invalid", {}, count=1, monitors=group(monitor, perf=perf),
                           token="test", timeout=17)
    assert (post.call_args.kwargs["timeout"]) == (17)
    monitor.stop.assert_called_once()
    monitor.close.assert_called_once()
    perf.close.assert_called_once()

def test_idle_failure_closes_collectors_before_any_request():
    from scripts.measure_overhead import measure_window
    monitors = [Mock(), Mock()]
    perf = Mock()
    with patch("requests.post") as post:
        with pytest.raises(RuntimeError, match="idle failure"):
            measure_window("http://example.invalid", {}, count=1, monitors=group(*monitors, perf=perf),
                           token="test",
                           control_window=Mock(side_effect=RuntimeError("idle failure")))
    post.assert_not_called()
    for monitor in [*monitors, perf]:
        monitor.close.assert_called_once()

def test_truncated_capture_cannot_pass_full_comparison():
    import json
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    import scripts.measure_overhead as overhead
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'capture.pcap'
        result = SimpleNamespace(returncode=0, stdout=json.dumps({
            'requests': {'round-0-full:4': {'latency_s': 0.1}}}), stderr='')
        with patch('subprocess.run', return_value=result):
            with pytest.raises(RuntimeError, match='coverage'):
                overhead.validate_capture(['parser'], path, token='round-0-full', count=2)
        assert not (path.with_suffix('.packets.json').exists())

def test_monitor_stop_failure_does_not_leave_other_threads_running():
    from scripts.measure_overhead import measure_window
    monitors = [Mock(), Mock()]
    monitors[0].stop.return_value = (None, "", [1, 2])
    monitors[1].stop.side_effect = RuntimeError("stop failed")
    response = Mock(status_code=200)
    response.json.return_value = {}
    with patch("requests.post", return_value=response):
        with pytest.raises(RuntimeError, match="stop failed"):
            measure_window("http://example.invalid", {}, count=1, monitors=group(*monitors), token="test")
    for monitor in monitors:
        monitor.stop.assert_called_once()
        monitor.close.assert_called_once()
