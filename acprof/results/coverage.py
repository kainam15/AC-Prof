"""Static producer contracts, separate from live capability and hardware evidence."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from acprof.results.metrics import METRICS, Metric


GPU_QUERIES = {
    'utilization': ('gpu_util_avg_pct', 'gpu_util_peak_pct'),
    'memory': ('gpu_mem_used_avg_bytes', 'gpu_mem_used_peak_bytes', 'gpu_mem_util_avg_pct', 'gpu_mem_util_peak_pct'),
    'sm_clock': ('gpu_sm_clock_mhz',), 'memory_clock': ('gpu_memory_clock_mhz',),
    'pstate': ('gpu_pstate',), 'temperature': ('gpu_temp_c',),
}


PMU_EVENT_FIELDS = {
    "cycles": ("cpu_cycles_per_request", "cpu_ipc"),
    "ref-cycles": ("cpu_ref_cycles_per_request",),
    "cache-references": ("cpu_cache_references_per_request", "cpu_cache_miss_rate_pct"),
    "cache-misses": ("cpu_cache_misses_per_request", "cpu_cache_miss_rate_pct"),
    "dTLB-loads": ("cpu_dtlb_loads_per_request", "cpu_dtlb_load_miss_rate_pct"),
    "dTLB-load-misses": ("cpu_dtlb_load_misses_per_request", "cpu_dtlb_load_miss_rate_pct"),
}


@dataclass(frozen=True)
class CollectorContract:
    scope: str
    cost: str
    prerequisites: str
    producer: str
    runtimes: tuple[str, ...] = ("docker", "process")
    derivation: str = "observed"


_ROW = "acprof/host/measurement/metrics.py"
_RESOURCE = "acprof/monitors/resource_usage.py"
_PROCESS = "acprof/monitors/process_usage.py"
_PROTOCOL = CollectorContract("protocol_identity", "none", "valid run identity", _ROW,
                              derivation="recorded")
SOURCE_CONTRACTS = {
    **dict.fromkeys(("protocol", "platform_identity", "execution_runtime",
                    "immutable_runtime_artifact", "verified_runtime_limits",
                    "collector_contract", "allocation_evidence", "selected_device_identity"), _PROTOCOL),
    "psutil": CollectorContract("observed_process_tree", "low", "target procfs access", _PROCESS,
                                derivation="recorded"),
    "psutil_observed_process_tree": CollectorContract(
        "observed_process_tree", "low", "procfs access; stable PID + start time", _PROCESS),
    "procfs_observed_process_tree": CollectorContract("observed_process_tree", "low", "Linux procfs; stable process membership", _PROCESS),
    "resource_quality": CollectorContract("per_metric", "none", "finalized collector evidence", _ROW, derivation="recorded"),
    "service_cgroup": CollectorContract("owned_service_cgroup", "low", "verified owned cgroup; membership; readable metric files", "acprof/monitors/process_cgroup.py", ("process",)),
    "managed_process_to_ready": CollectorContract("managed_service_lifetime", "none", "Process readiness", _ROW, ("process",)),
    "nvml_selected_device": CollectorContract("selected_gpu_device", "medium", "GPU UUID; NVML query support and permission", _RESOURCE),
    "pcap": CollectorContract("correlated_http_connection_frames", "high", "native Linux; capture permission; complete request matching", "acprof/packet/merge_packet_latency.py"),
    "input_plan": CollectorContract("materialized_request", "none", "frozen input plan", "acprof/host/input_plan/__init__.py", derivation="recorded"),
    "client_http": CollectorContract("client_http_requests", "none", "successful requests; tokenizer/geometry when applicable", _ROW),
    "derived": CollectorContract("source_dependent", "none", "finite source fields and positive denominator", "acprof/host/measurement/collector_metrics.py", derivation="derived"),
    "torch": CollectorContract("independent_model_forward", "high", "explicit profiler; supported backend", "acprof/host/profiling/compute_plan.py", ("docker",)),
    "ncu": CollectorContract("independent_gpu_kernels", "high", "explicit profiler; GPU counter permission", "acprof/host/profiling/compute_plan.py", ("docker",)),
    "massif": CollectorContract("independent_process_heap", "high", "explicit profiler; CPU; Valgrind", "acprof/host/profiling/execution_plan.py", ("docker",)),
    "nsys": CollectorContract("independent_cuda_timeline", "high", "explicit profiler; NVIDIA GPU; Nsys", "acprof/host/profiling/execution_plan.py", ("docker",)),
    "rapl_package": CollectorContract("host_cpu_packages", "medium", "native Linux; readable RAPL package domains", "acprof/monitors/energy_cpu.py"),
    "rapl_dram": CollectorContract("host_dram_domains", "medium", "native Linux; DRAM requested; complete readable domains", "acprof/monitors/energy_cpu.py"),
    "rapl_cgroup_attribution": CollectorContract("cgroup_cpu_time_share_of_host_packages", "medium", "native Linux; RAPL + cgroup + host CPU counters", "acprof/monitors/energy_cpu.py", ("docker",), "estimated"),
    "cgroup": CollectorContract("container_cgroup", "low", "local Docker; cgroup v2; readable metric files", _RESOURCE, ("docker",)),
    "host_cpufreq": CollectorContract("host_visible_cpus", "low", "readable cpufreq or cpuinfo; unavailable on WSL2", _RESOURCE, ("docker",)),
    "perf_stat": CollectorContract("container_root_pid_pmu_with_inheritance", "medium", "native Linux; perf; PID permission; thread coverage; valid running ratio", "acprof/monitors/perf_mips.py"),
    "docker_and_server": CollectorContract("container_service_lifetime", "none", "Docker readiness timestamps", _ROW, ("docker",)),
    "handler": CollectorContract("actual_request", "none", "handler response contract", _ROW, derivation="recorded"),
}


def collector_contract(metric: Metric, runtime: str = "docker") -> CollectorContract:
    """Resolve explicit exceptions; a new source must supply its own contract."""
    contract = SOURCE_CONTRACTS[metric.source]
    if metric.source == "pcap" and runtime == "process":
        return replace(contract, scope="loopback_correlated_http_connection_frames", prerequisites="native Linux; loopback capture permission; exact service port and request IDs")
    if metric.source == "perf_stat" and runtime == "process":
        return replace(contract, scope="observed_processes_threads_and_inheritance", prerequisites="native Linux; perf attach to owned service and observed descendants; matching PID start times")
    if metric.name.startswith("throughput_app_"):
        return replace(contract, derivation="derived")
    if metric.name in {"throughput_samples_per_s", "throughput_samples_per_s_per_cpu_core"}:
        return replace(contract, scope="application_or_packet_latency", producer="acprof/packet/merge_packet_latency.py")
    if metric.name in {"response_payload_bytes", "request_payload_bytes"}:
        scope = "decoded_http_response_body" if metric.name == "response_payload_bytes" else "prepared_http_request_body"
        return replace(contract, scope=scope, producer="acprof/host/measurement/requests.py")
    if metric.name.startswith("process_leader_"):
        return replace(contract, scope="observed_process_tree_leaders")
    if metric.source in {'cgroup', 'service_cgroup', 'psutil_observed_process_tree', 'host_cpufreq'} and (
            '_avg_' in metric.name or '_peak_' in metric.name or metric.name.startswith('process_cpu_')):
        return replace(contract, cost='medium')
    if metric.source == "nvml_selected_device":
        if metric.row_group == "gpu_energy" or metric.name.startswith("gpu_idle_") or metric.name in {
            "gpu_energy_source", "gpu_energy_fallback_reason",
        }:
            return replace(contract, producer="acprof/monitors/energy_nvml.py",
                           prerequisites="native Linux; NVML counter or power integration; effective energy also needs matched control")
    if metric.source == "derived":
        if metric.name.startswith("container_attributed_"):
            return replace(contract, scope="cgroup_cpu_share_plus_selected_gpu", runtimes=("docker",), derivation="estimated")
        if metric.name.startswith("cpu_cycles_est_"):
            return replace(contract, scope="cgroup_cpu_time_times_host_frequency", runtimes=("docker",), derivation="estimated")
        return replace(contract, scope="client_http_requests", producer=_ROW)
    return contract


def metric_coverage(name: str, runtime: str) -> dict:
    """Registration and implementation never imply runtime or hardware success."""
    if runtime not in {"docker", "process"}:
        raise ValueError(f"unknown runtime: {runtime}")
    metric = METRICS[name]
    contract = collector_contract(metric, runtime)
    implemented = runtime in contract.runtimes
    return {"metric": name, "runtime": runtime, "registered": True,
            "implemented": implemented, "source": metric.source, "unit": metric.unit,
            "window": metric.window, "scope": contract.scope, "derivation": contract.derivation,
            "cost": contract.cost, "prerequisites": contract.prerequisites,
            "producer": contract.producer,
            "collectable": "conditional" if implemented else "unsupported",
            "hardware_validation": "not_verified",
            "missing_reason": "runtime_not_implemented" if not implemented else "requires_current_run_evidence"}


@lru_cache(maxsize=2)
def planned_fields(runtime: str) -> dict[str, list[str]]:
    """Link collector decisions to registry fields without duplicating their definitions."""
    by_source = {"client_http": "client_http", "pcap": "packet_latency", "perf_stat": "cpu_instructions",
                 "rapl_package": "cpu_energy", "rapl_cgroup_attribution": "cpu_energy", "rapl_dram": "dram_energy",
                 "cgroup": "cgroup", "service_cgroup": "cgroup", "nvml_selected_device": "gpu_power"}
    groups: dict[str, list[str]] = {}
    for name, metric in METRICS.items():
        if runtime not in collector_contract(metric, runtime).runtimes:
            continue
        key = by_source.get(metric.source)
        if key is None and metric.kind == "number":
            key = metric.tool or metric.source
        if metric.source in {"psutil_observed_process_tree", "procfs_observed_process_tree"}:
            key = ("process_io" if "_io_" in name else "process_ctx" if "_ctx_" in name else
                   "process_fds" if "_fds_" in name else "process_threads" if "_threads_" in name or "_count_" in name else
                   "process_cpu" if "_cpu_" in name else "process_memory")
        if metric.source == "nvml_selected_device" and metric.row_group != "gpu_energy" and not name.startswith("gpu_idle_"):
            continue
        if key:
            groups.setdefault(key, []).append(name)
    groups.update({"gpu_" + name: list(items) for name, items in GPU_QUERIES.items()})
    return groups


@lru_cache(maxsize=2)
def field_collectors(runtime: str) -> dict[str, str]:
    return {name: group for group, fields in planned_fields(runtime).items() for name in fields}
