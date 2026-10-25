"""Raw evidence uses independently derived numeric oracles and no live hardware."""
from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest

from acprof.telemetry import (
    capture_samples,
    raw_capability,
    read_window_evidence,
    recompute_window,
    window_evidence,
    write_window_evidence,
)


def evidence(collectors):
    return window_evidence(
        identity={"run_id": "run", "case_id": "case", "case_attempt_id": "a1", "window_id": "w0"},
        context={"completed_requests": 2}, collectors=collectors,
    )


def cpu_collector():
    return {
        "status": "recorded", "retention": {"complete": True},
        "samples": [
            {"timestamp": 0.0, "energy_uj": [0], "host_active_s": 0.0, "container_cpu_s": 0.0},
            {"timestamp": 1.0, "energy_uj": [2_000_000], "host_active_s": 1.0, "container_cpu_s": 0.5},
            {"timestamp": 2.0, "energy_uj": [5_000_000], "host_active_s": 3.0, "container_cpu_s": 1.5},
        ],
        "context": {"domains": [{"name": "package0", "energy_path": "/not/read", "max_range_uj": 10_000_000}],
                    "idle_power_w": 1.0, "min_power_interval_s": 0.025, "dram_domains": [],
                    "dram_idle_power_w": None, "dram_status": "not_requested", "dram_error": ""},
    }


def test_replay_cpu_energy_and_container_share_from_known_counter_deltas():
    result = recompute_window(evidence({"cpu": cpu_collector()}))["cpu"]
    assert result["cpu_energy_total_j"] == 5.0
    assert result["cpu_avg_power_total_w"] == 2.5
    assert result["cpu_peak_power_total_w"] == 3.0
    assert result["cpu_energy_eff_j"] == 3.0
    assert result["vcpu_cpu_share"] == 0.5
    assert result["vcpu_cpu_time_s"] == 1.5
    assert result["vcpu_energy_total_j"] == 2.5
    assert result["vcpu_energy_eff_j"] == 1.5
    assert result["dram"]["status"] == "not_requested"
    assert result["dram"]["window_energy_j"] is None


def test_replay_rapl_wrap_uses_recorded_hardware_range():
    cpu = cpu_collector()
    cpu["samples"] = [cpu["samples"][0], cpu["samples"][-1]]
    cpu["samples"][0]["energy_uj"] = [9_000_000]
    cpu["samples"][-1]["energy_uj"] = [1_000_000]
    result = recompute_window(evidence({"cpu": cpu}))["cpu"]
    assert result["cpu_energy_total_j"] == 2.0
    assert result["cpu_avg_power_total_w"] == 1.0


@pytest.mark.parametrize(("source", "expected_energy", "expected_average", "expected_effective"), [
    ("power_integration", 80.0, 80.0 / 3.0, 50.0),
    ("nvml_total_energy", 75.0, 25.0, 45.0),
])
def test_replay_gpu_preserves_chosen_counter_or_trapezoidal_source(source, expected_energy,
                                                                 expected_average, expected_effective):
    gpu = {
        "status": "recorded", "retention": {"complete": True},
        "samples": [{"timestamp": 0.0, "power_w": 10.0}, {"timestamp": 1.0, "power_w": 30.0},
                    {"timestamp": 3.0, "power_w": 30.0}],
        "context": {"idle_power_w": 10.0, "counter_start_mj": 100_000, "counter_end_mj": 175_000,
                    "measurement_duration_s": 3.0, "energy_source": source, "energy_fallback_reason": ""},
    }
    result = recompute_window(evidence({"gpu": gpu}))["gpu"]
    assert result["energy_total_j"] == expected_energy
    assert result["avg_power_total_w"] == pytest.approx(expected_average)
    assert result["energy_eff_j"] == expected_effective
    assert result["peak_power_total_w"] == 30.0
    assert result["peak_power_eff_w"] == 20.0
    assert result["energy_source"] == source


def test_gpu_counter_only_replay_retains_unknown_peaks_and_uses_recorded_idle_baseline():
    gpu = {
        "status": "recorded", "retention": {"complete": True}, "samples": [],
        "context": {"idle_power_w": 1.0, "counter_start_mj": 10_000, "counter_end_mj": 20_000,
                    "measurement_duration_s": 2.0, "energy_source": "nvml_total_energy", "energy_fallback_reason": ""},
    }
    result = recompute_window(evidence({"gpu": gpu}))["gpu"]
    assert result["energy_total_j"] == 10.0
    assert result["avg_power_total_w"] == 5.0
    assert result["energy_eff_j"] == 8.0
    assert result["avg_power_eff_w"] == 4.0
    assert result["peak_power_total_w"] is None
    assert result["peak_power_eff_w"] is None
    assert result["energy_iters"] == 0
    assert gpu["samples"] == []


def test_replay_resource_usage_and_boundary_counters():
    sample = {"gpu_util_pct": None, "gpu_mem_used_bytes": None, "gpu_mem_total_bytes": None}
    resource = {
        "status": "recorded", "retention": {"complete": True},
        "samples": [{**sample, "timestamp": 0.0, "container_cpu_s": 1.0, "container_mem_usage_bytes": 100},
                    {**sample, "timestamp": 2.0, "container_cpu_s": 3.0, "container_mem_usage_bytes": 300}],
        "context": {"cpu_cores": 2.0, "mem_limit_bytes": 1000, "min_cpu_interval_s": 0.025,
                    "swap_limit_bytes": 4000, "io_start": [100, 30], "io_end": [190, 50],
                    "window_duration_s": 2.0,
                    "counters_start": {"cpu_throttle": {"nr_periods": 10, "nr_throttled": 2,
                                                           "throttled_usec": 100_000}},
                    "counters_end": {"cpu_throttle": {"nr_periods": 14, "nr_throttled": 3,
                                                         "throttled_usec": 300_000}}},
    }
    result = recompute_window(evidence({"resource": resource}))["resource"]
    assert result["container_cpu_util_avg_pct"] == 50.0
    assert result["container_mem_usage_avg_bytes"] == 200.0
    assert result["container_mem_usage_peak_bytes"] == 300.0
    assert result["container_mem_util_avg_pct"] == 20.0
    assert result["container_io_read_bytes"] == 90.0
    assert result["container_io_write_bytes"] == 20.0
    assert result["container_cpu_throttled_period_ratio_pct"] == 25.0
    assert result["container_cpu_throttled_time_s"] == 0.2
    assert result["gpu_util_avg_pct"] is None


def test_retention_keeps_boundaries_and_refuses_inexact_recomputation():
    samples = list(range(10))
    captured = capture_samples(samples, 4)
    assert samples == list(range(10))  # The original reduction input is untouched.
    assert captured["samples"] == [0, 3, 6, 9]
    assert captured["retention"]["omitted_samples"] == 6
    assert captured["retention"]["complete"] is False
    cpu = cpu_collector()
    cpu["retention"] = captured["retention"]
    document = evidence({"cpu": cpu})
    assert raw_capability(document)["can_recompute"] is False
    with pytest.raises(ValueError, match="truncated cpu"):
        recompute_window(document)
    assert raw_capability(None) == {"status": "unavailable", "reason": "legacy_or_missing_raw_evidence",
                                    "can_recompute": False}


def test_evidence_hash_immutability_and_finite_json(tmp_path):
    document = evidence({"cpu": cpu_collector(), "missing": {"status": "unavailable", "value": float("nan")}})
    path = tmp_path / "window.json"
    reference = write_window_evidence(path, document)
    assert json.loads(path.read_text())["collectors"]["missing"]["value"] is None
    assert read_window_evidence(path, expected_sha256=reference["sha256"]) == document
    assert write_window_evidence(path, document) == reference
    changed = copy.deepcopy(document)
    changed["context"]["completed_requests"] = 3
    with pytest.raises(ValueError, match="replace different"):
        write_window_evidence(path, changed)
    path.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_window_evidence(path, expected_sha256=reference["sha256"])


def test_gpu_replay_module_does_not_load_nvml_or_probe_hardware():
    subprocess.run([sys.executable, "-c", "import sys; import acprof.telemetry; "
                    "import acprof.monitors.gpu_metrics; assert 'pynvml' not in sys.modules"], check=True)
