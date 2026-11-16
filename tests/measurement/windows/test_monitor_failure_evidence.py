"""Cleanup failures remain visible without replacing the original run outcome."""

import io
import subprocess
from unittest.mock import Mock

import pytest

from acprof.monitors import energy_nvml, perf_mips, resource_readers, resource_usage


@pytest.mark.parametrize("kind", ["energy", "resource"])
def test_nvml_shutdown_failure_propagates_until_close_succeeds(monkeypatch, kind):
    module = energy_nvml if kind == "energy" else resource_usage
    nvml = Mock()
    nvml.nvmlDeviceGetName.return_value = "fixture GPU"
    monkeypatch.setattr(module, "pynvml", nvml)
    if kind == "resource":
        monkeypatch.setattr(resource_readers, "resolve_container_metric_readers",
                            Mock(side_effect=RuntimeError("no fixture cgroup")))
        monkeypatch.setattr(resource_readers, "prepare_cpu_frequency_reader",
                            lambda *_args: lambda: (None, None))
        monitor = resource_usage.ResourceUsageMonitor(use_gpu=True)
    else:
        monitor = energy_nvml.GPUEnergyMonitor()
    failure = RuntimeError("driver shutdown denied")
    nvml.nvmlShutdown.side_effect = failure

    with pytest.raises(RuntimeError, match="driver shutdown denied") as caught:
        monitor.close()

    assert caught.value is failure
    assert monitor._closed is False
    assert monitor._gpu_initialized is True
    nvml.nvmlShutdown.assert_called_once_with()

    nvml.nvmlShutdown.side_effect = None
    monitor.close()
    monitor.close()

    assert monitor._closed is True
    assert monitor._gpu_initialized is False
    assert nvml.nvmlShutdown.call_count == 2


def test_perf_timeout_after_kill_propagates_and_releases_output_streams():
    monitor = perf_mips.PerfMIPSMonitor("test-container")
    process = Mock(stdout=io.StringIO(), stderr=io.StringIO())
    process.poll.return_value = None
    failure = subprocess.TimeoutExpired("perf", 5)
    process.communicate.side_effect = failure
    monitor._proc = process

    with pytest.raises(subprocess.TimeoutExpired) as caught:
        monitor.close()

    assert caught.value is failure
    assert "Cleanup failed for perf process" in caught.value.__notes__
    process.kill.assert_called_once_with()
    process.communicate.assert_called_once_with(timeout=5.0)
    assert process.stdout.closed
    assert process.stderr.closed

    process.poll.return_value = 0
    monitor.close()
    monitor.close()
    process.kill.assert_called_once_with()


def test_cpu_directory_fallback_explains_invalid_online_map(tmp_path, caplog):
    (tmp_path / "online").write_text("invalid-cpu-map", encoding="utf-8")
    (tmp_path / "cpu0").mkdir()
    (tmp_path / "cpu2").mkdir()

    assert resource_readers._discover_cpu_ids(str(tmp_path)) == [0, 2]
    assert "CPU online map unavailable" in caplog.text
    assert "ValueError" in caplog.text
