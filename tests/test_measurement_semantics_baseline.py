"""Deterministic, hardware-free reference values for core measurement units.

Expected values are derived from the synthetic input counters, not calculated by
calling the implementation under test. These are numerical *contracts*, not
native-host accuracy or repeatability evidence.
"""
from __future__ import annotations

import math

import pytest

from acprof.monitors import energy_cpu, resource_metrics


def test_resource_reference_baseline_preserves_units_and_denominators():
    # Two CPU cores; elapsed=2 s; consumed CPU time=1 s => 25% average.
    samples = [
        resource_metrics.ResourceUsageSample(0.0, 0.0, 100, None, None, None),
        resource_metrics.ResourceUsageSample(1.0, 0.2, 200, None, None, None),
        resource_metrics.ResourceUsageSample(2.0, 1.0, 300, None, None, None),
    ]
    result = resource_metrics.reduce_resource_samples(
        samples, cpu_cores=2.0, mem_limit_bytes=400.0, min_cpu_interval_s=0.05,
    )
    assert result.resource_usage_iters == 3
    assert result.container_cpu_util_avg_pct == pytest.approx(25.0)
    assert result.container_cpu_util_peak_pct == pytest.approx(40.0)
    assert result.container_mem_usage_avg_bytes == pytest.approx(200.0)
    assert result.container_mem_usage_peak_bytes == pytest.approx(300.0)
    assert result.container_mem_util_avg_pct == pytest.approx(50.0)
    assert result.container_mem_util_peak_pct == pytest.approx(75.0)
    assert math.isnan(result.gpu_util_avg_pct)  # unavailable is not 0% utilization


def test_rapl_wrap_and_control_reference_baseline_preserve_joules():
    # RAPL range=1 J. First 1 s interval wraps: 0.9 -> 0.1 J = 0.2 J;
    # next interval 0.1 -> 0.5 J = 0.4 J. Idle baseline=0.1 W.
    domain = energy_cpu.RaplDomain("intel-rapl:0", "unused", 1_000_000)
    samples = [
        energy_cpu.CPUSample(0.0, [900_000], 0.0, 0.0),
        energy_cpu.CPUSample(1.0, [100_000], 2.0, 0.5),
        energy_cpu.CPUSample(2.0, [500_000], 4.0, 1.0),
    ]
    result = energy_cpu.reduce_cpu_samples(samples, idle_power_w=0.1, domains=[domain])
    assert result.cpu_energy_total_j == pytest.approx(0.6)
    assert result.cpu_avg_power_total_w == pytest.approx(0.3)
    assert result.cpu_peak_power_total_w == pytest.approx(0.4)
    assert result.cpu_energy_eff_j == pytest.approx(0.4)
    assert result.vcpu_cpu_share == pytest.approx(0.25)
    assert result.vcpu_energy_total_j == pytest.approx(0.15)
    assert result.vcpu_energy_eff_j == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        (100.0, 100.0, 0.0),
        (100.0, 120.0, 20.0),
        (100.0, 99.0, None),   # counter reset must not look like zero usage
        (100.0, math.nan, None),
        (100.0, math.inf, None),
        (-math.inf, 120.0, None),
        (-1.0, 120.0, None),  # cumulative counters cannot start below zero
    ],
)
def test_counter_baseline_distinguishes_zero_reset_and_invalid(before, after, expected):
    actual = resource_metrics.counter_delta({"count": before}, {"count": after}, "count")
    if expected is None:
        assert math.isnan(actual)
    else:
        assert actual == expected


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_invalid_snapshot_cannot_be_promoted_to_measured_value(invalid):
    assert math.isnan(resource_metrics._snapshot_value({"peak": invalid}, "peak"))
