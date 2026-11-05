"""Unavailable profiler rates remain distinct from valid zero work."""
import math

import pytest

from acprof.host.measurement.config import ClientConfig
from acprof.host.measurement.metrics import WindowMetrics
from acprof.host.measurement.requests import WindowRequestProgress
from acprof.host.measurement.types import WindowObservations
from acprof.host.profiling.compute_plan import (
    NCU_TOTAL_MFLOP_FIELD,
    TORCH_LOGICAL_MFLOP_FIELD,
    compute_mflops,
)
from acprof.host.profiling.execution_plan import ExecutionProfileResolver


@pytest.mark.parametrize(("work", "latency"), [
    (120, math.inf), (120, "inf"), (120, "1e999"),
    (120, -math.inf), (120, math.nan), (120, 0), (120, -1),
    (120, None), (120, ""), (120, "invalid"),
    (math.inf, 0.5), (-math.inf, 0.5), ("inf", 0.5),
    (math.nan, 0.5), (None, 0.5), ("invalid", 0.5),
    (1e308, 1e-308),
])
def test_compute_rate_requires_finite_inputs_and_result(work: object, latency: object) -> None:
    assert math.isnan(compute_mflops(work, latency))


@pytest.mark.parametrize(("work", "latency", "expected"), [
    (0, 0.4, 0), (120, 0.4, 300), ("120", "0.5", 240), (1e-200, 1e-200, 1),
])
def test_compute_rate_preserves_valid_numeric_values(work: object, latency: object, expected: float) -> None:
    assert compute_mflops(work, latency) == expected


@pytest.mark.parametrize("latency", [math.inf, -math.inf, math.nan])
def test_live_row_does_not_turn_nonfinite_latency_into_zero_compute_rate(latency: float) -> None:
    metrics = WindowMetrics(ClientConfig(gpu_mode="on"), {"input_scale": 8, "payload": {}}, use_energy=False)
    metrics.latency_app_s = latency
    plan = {"profiles": {"gpu": {
        "torch_profiler_eager": {"entries": [{"input_scale": 8, TORCH_LOGICAL_MFLOP_FIELD: 120}]},
        "ncu": {"entries": [{"input_scale": 8, NCU_TOTAL_MFLOP_FIELD: 60}]},
    }}}
    row = metrics.build_row(
        WindowRequestProgress(count=1), WindowObservations(), warmup=0, repeat_idx=0,
        environment_class="native_linux", cold_start={}, compute_profile_plan=plan,
        execution_profiles=ExecutionProfileResolver({}), cpu_idle_history=[], gpu_idle_history=[],
    )
    assert row[TORCH_LOGICAL_MFLOP_FIELD] == "120.000000"
    assert row[NCU_TOTAL_MFLOP_FIELD] == "60.000000"
    for field in (
        "model_logical_mflops_app_torch_profiler_eager", "model_logical_mflops_packet_torch_profiler_eager",
        "gpu_executed_mflops_app_ncu", "gpu_executed_mflops_packet_ncu",
    ):
        assert row[field] == "nan", field
