"""Cleanup failures remain visible without replacing the original run outcome."""

import subprocess
from unittest.mock import Mock

import pytest

from acprof.monitors import energy_nvml, perf_mips, resource_readers, resource_usage


@pytest.mark.parametrize("kind", ["energy", "resource"])
def test_nvml_shutdown_failure_is_reported_once_and_close_remains_idempotent(
    monkeypatch, caplog, kind,
):
    module = energy_nvml if kind == "energy" else resource_usage
    monitor_type = module.GPUEnergyMonitor if kind == "energy" else module.ResourceUsageMonitor
    monitor = object.__new__(monitor_type)
    monitor._closed = False
    monitor._thread = None
    if kind == "resource":
        monitor._gpu_initialized = True
    shutdown = Mock(side_effect=RuntimeError("driver shutdown denied"))
    monkeypatch.setattr(module.pynvml, "nvmlShutdown", shutdown)

    monitor.close()
    monitor.close()

    assert monitor._closed is True
    shutdown.assert_called_once_with()
    assert "driver shutdown denied" in caplog.text
    assert "RuntimeError" in caplog.text


def test_perf_timeout_after_kill_is_reported(caplog):
    monitor = perf_mips.PerfMIPSMonitor("test-container")
    process = Mock()
    process.poll.return_value = None
    process.communicate.side_effect = subprocess.TimeoutExpired("perf", 5)
    monitor._proc = process

    monitor.close()

    process.kill.assert_called_once_with()
    assert "Perf cleanup failed after kill" in caplog.text
    assert "TimeoutExpired" in caplog.text


def test_cpu_directory_fallback_explains_invalid_online_map(tmp_path, caplog):
    (tmp_path / "online").write_text("invalid-cpu-map", encoding="utf-8")
    (tmp_path / "cpu0").mkdir()
    (tmp_path / "cpu2").mkdir()

    assert resource_readers._discover_cpu_ids(str(tmp_path)) == [0, 2]
    assert "CPU online map unavailable" in caplog.text
    assert "ValueError" in caplog.text
