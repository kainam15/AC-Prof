import math
from unittest.mock import patch

import pytest

from acprof.monitors import energy_nvml


class FakeThread:
    instances = []

    def __init__(self, target, daemon=False):
        self.target = target
        self.daemon = daemon
        self.started = False
        self.joined = False
        FakeThread.instances.append(self)

    def start(self):
        self.started = True

    def join(self, timeout=None):
        self.joined = True

    def is_alive(self):
        return self.started and not self.joined



def test_cumulative_energy_is_primary_and_power_still_supplies_peak():
    with patch.object(energy_nvml.pynvml, "nvmlInit"), patch.object(
        energy_nvml.pynvml, "nvmlDeviceGetHandleByIndex", return_value="handle"
    ), patch.object(energy_nvml.pynvml, "nvmlDeviceGetName", return_value="GPU"), patch.object(
        energy_nvml.pynvml, "nvmlDeviceGetTotalEnergyConsumption", side_effect=[100_000, 140_000]
    ), patch.object(energy_nvml.pynvml, "nvmlDeviceGetPowerUsage", side_effect=[20_000, 40_000]), patch.object(
        energy_nvml.time, "perf_counter", side_effect=[0.0, 2.0]
    ), patch.object(energy_nvml.threading, "Thread", FakeThread):
        monitor = energy_nvml.GPUEnergyMonitor()
        monitor.idle_power_w = 10.0
        monitor.start()
        result, _, _, samples = monitor.stop()
    assert (result.energy_total_j) == (40.0)
    assert (result.avg_power_total_w) == (20.0)
    assert (result.energy_eff_j) == (20.0)
    assert (result.peak_power_total_w) == (40.0)
    assert (result.energy_source) == ("nvml_total_energy")
    assert (result.counter_start_mj) == (100_000)
    assert (result.counter_end_mj) == (140_000)
    assert (result.measurement_duration_s) == (2.0)
    assert (len(samples)) == (2)

@pytest.mark.parametrize('readings_case', range(2), ids=['[100000, 90000]', "[RuntimeError('not supported')]"])
def test_reset_or_unsupported_counter_uses_power_integral_with_reason(readings_case):
    readings = tuple(([100000, 90000], [RuntimeError('not supported')]))[readings_case]
    with patch.object(energy_nvml.pynvml, "nvmlInit"), patch.object(
        energy_nvml.pynvml, "nvmlDeviceGetHandleByIndex", return_value="handle"
    ), patch.object(energy_nvml.pynvml, "nvmlDeviceGetName", return_value="GPU"), patch.object(
        energy_nvml.pynvml, "nvmlDeviceGetTotalEnergyConsumption", side_effect=readings
    ), patch.object(energy_nvml.pynvml, "nvmlDeviceGetPowerUsage", side_effect=[20_000, 40_000]), patch.object(
        energy_nvml.time, "perf_counter", side_effect=[0.0, 2.0]
    ), patch.object(energy_nvml.threading, "Thread", FakeThread):
        monitor = energy_nvml.GPUEnergyMonitor()
        monitor.start()
        result, _, _, _ = monitor.stop()
    assert (result.energy_total_j) == (60.0)
    assert (getattr(result, "energy_source", None)) == ("power_integration")
    assert (result.energy_fallback_reason)

def test_cumulative_energy_survives_unsupported_power_sampling():
    with patch.object(energy_nvml.pynvml, "nvmlInit"), patch.object(
        energy_nvml.pynvml, "nvmlDeviceGetHandleByIndex", return_value="handle"
    ), patch.object(energy_nvml.pynvml, "nvmlDeviceGetName", return_value="GPU"), patch.object(
        energy_nvml.pynvml, "nvmlDeviceGetTotalEnergyConsumption", side_effect=[10_000, 20_000]
    ), patch.object(energy_nvml.pynvml, "nvmlDeviceGetPowerUsage", side_effect=RuntimeError("power unsupported")), patch.object(
        energy_nvml.time, "perf_counter", side_effect=[0.0, 2.0]
    ), patch.object(energy_nvml.threading, "Thread", FakeThread):
        monitor = energy_nvml.GPUEnergyMonitor()
        monitor.start()
        result, _, error, samples = monitor.stop()
    assert (result.energy_total_j) == (10.0)
    assert (result.avg_power_total_w) == (5.0)
    assert (result.energy_source) == ("nvml_total_energy")
    assert (math.isnan(result.peak_power_total_w))
    assert (samples) == ([])
    assert ("power unsupported") in (error)

def test_apply_control_baseline_uses_integrated_average_and_records_method() -> None:
    with patch("acprof.monitors.energy_nvml.pynvml.nvmlInit"), patch(
        "acprof.monitors.energy_nvml.pynvml.nvmlDeviceGetHandleByIndex",
        return_value="handle",
    ), patch(
        "acprof.monitors.energy_nvml.pynvml.nvmlDeviceGetName",
        return_value=b"Test GPU",
    ):
        monitor = energy_nvml.GPUEnergyMonitor(sample_hz=10.0, idle_seconds=2.0)

    samples = [(0.0, 10.0), (1.0, 30.0), (3.0, 30.0)]
    result = energy_nvml._result_from_samples(samples, idle_power_w=float("nan"))
    idle_power_w = monitor.apply_control_baseline(result, samples, trace=True)

    assert (idle_power_w) == (80.0 / 3.0) or round(abs((idle_power_w) - (80.0 / 3.0)), 7) == 0
    assert (monitor.idle_trace["gpu_idle_baseline_method"]) == ("matched_control_time_weighted_mean")
    assert (monitor.idle_trace["gpu_idle_trace_schema"]) == ("nvml_gpu_control_v1")


def test_start_stop_samples_and_calculates_energy() -> None:
    FakeThread.instances = []
    with patch("acprof.monitors.energy_nvml.pynvml.nvmlInit"), patch(
        "acprof.monitors.energy_nvml.pynvml.nvmlDeviceGetHandleByIndex",
        return_value="handle",
    ), patch(
        "acprof.monitors.energy_nvml.pynvml.nvmlDeviceGetName",
        return_value="Test GPU",
    ), patch(
        "acprof.monitors.energy_nvml.pynvml.nvmlDeviceGetPowerUsage",
        side_effect=[20000, 40000],
    ), patch(
        "acprof.monitors.energy_nvml.time.perf_counter",
        side_effect=[0.0, 2.0],
    ), patch("acprof.monitors.energy_nvml.threading.Thread", FakeThread):
        monitor = energy_nvml.GPUEnergyMonitor(sample_hz=10.0, idle_seconds=0.0)
        monitor.idle_power_w = 10.0

        monitor.start()
        assert (FakeThread.instances[0].started)

        result, gpu_name, err, samples = monitor.stop()

    assert (FakeThread.instances[0].joined)
    assert (gpu_name) == ("Test GPU")
    assert (err) == ("")
    assert (samples) == ([(0.0, 20.0), (2.0, 40.0)])
    assert (result.energy_iters) == (2)
    assert (result.idle_power_w) == (10.0)
    assert (result.avg_power_total_w) == (30.0)
    assert (result.peak_power_total_w) == (40.0)
    assert (result.energy_total_j) == (60.0)
    assert (result.avg_power_eff_w) == (20.0)
    assert (result.peak_power_eff_w) == (30.0)
    assert (result.energy_eff_j) == (40.0)

def test_nvml_init_failure_returns_error_result() -> None:
    with patch("acprof.monitors.energy_nvml.pynvml.nvmlInit", side_effect=RuntimeError("nvml boom")):
        monitor = energy_nvml.GPUEnergyMonitor(sample_hz=10.0, idle_seconds=0.0)
        monitor.start()
        result, gpu_name, err, samples = monitor.stop()

    assert (gpu_name) == ("unknown")
    assert ("nvml boom") in (err)
    assert (samples) == ([])
    assert (result.energy_iters) == (0)
    assert (math.isnan(result.energy_total_j))
