"""固定历史 CSV 顺序，并核对跨消费者和关键口径。"""
import hashlib
import json

import pytest

from acprof.config import CSV_FIELDS


def test_historical_field_order_is_unchanged():
    additions = {'environment_class', 'result_origin', 'workload_contract', 'gpu_device_uuid', 'gpu_energy_source',
                 'gpu_energy_fallback_reason', 'gpu_idle_energy_source',
                 'latency_tail_ratio', 'latency_app_tail_ratio',
                 'cpu_cycles_per_request', 'cpu_ref_cycles_per_request', 'cpu_ipc', 'cpu_perf_running_pct'}
    historical_fields = [name for name in CSV_FIELDS if name not in additions and not name.startswith('dram_')]
    assert (hashlib.sha256(json.dumps(historical_fields).encode()).hexdigest()) == ("1422b14ebaa48586573923cea2d33f615e6dc180d099cdf773678f453e3268f3")

def test_workload_contract_is_additive_text_not_a_numeric_metric():
    from acprof.metric_registry import NUMERIC_FIELDS
    assert ('workload_contract') in (CSV_FIELDS)
    assert ('workload_contract') not in (NUMERIC_FIELDS)

def test_consumers_share_registry_without_changing_tool_completeness():
    from acprof.host.posthoc.context import TOOL_FIELDS, TOOL_METRIC_FIELDS
    from acprof.metric_registry import CSV_FIELDS as registry_fields, tool_fields
    assert (CSV_FIELDS) is (registry_fields)
    for tool in TOOL_FIELDS:
        assert (TOOL_FIELDS[tool]) == (tool_fields(tool))
        assert (TOOL_METRIC_FIELDS[tool]) == (tool_fields(tool, numeric_only=True))
    assert (tool_fields("torch", numeric_only=True)) == (("model_logical_mflop_per_request_torch_profiler_eager",))

def test_units_windows_and_text_are_explicit():
    from acprof.metric_registry import METRICS, NUMERIC_FIELDS
    assert (METRICS["gpu_energy_total_j"].unit) == ("J/request")
    assert (METRICS["dram_window_energy_j"].unit) == ("J/window")
    assert (METRICS["dram_energy_per_request_j"].unit) == ("J/request")
    assert ("dram_energy_status") not in (NUMERIC_FIELDS)
    assert ("result_origin") not in (NUMERIC_FIELDS)
    assert (METRICS["container_mem_peak_cgroup_bytes"].window) == ("cgroup_lifetime")
    assert (METRICS["cpu_heap_peak_bytes_massif"].window) == ("profiler_process_lifetime")
    assert ("gpu_pstate") not in (NUMERIC_FIELDS)
    assert ("gpu_idle_measured_at") not in (NUMERIC_FIELDS)

@pytest.mark.parametrize('name,direction', [('latency_app_p95_s', 'lower'), ('throughput_samples_per_s', 'higher'), ('cpu_ipc', 'higher'), ('cpu_cycles_per_request', 'lower'), ('cpu_ref_cycles_per_request', 'lower'), ('cold_start_s', 'lower'), ('container_mem_usage_peak_bytes', 'lower'), ('gpu_mem_used_peak_bytes', 'lower'), ('container_cpu_util_peak_pct', 'neutral'), ('gpu_util_peak_pct', 'neutral')])
def test_visualization_semantics_distinguish_objectives_from_utilization(name, direction):
    from acprof.metric_registry import METRICS
    assert (getattr(METRICS[name], "direction", None)) == (direction)
    assert (getattr(METRICS[name], "label", ""))
