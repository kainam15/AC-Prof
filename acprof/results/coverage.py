"""Static producer contracts, separate from live capability and hardware evidence."""
from __future__ import annotations

from dataclasses import dataclass, replace

from acprof.results.metrics import METRICS, Metric


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
                                ("process",), "diagnostic"),
    "psutil_observed_process_tree": CollectorContract(
        "observed_process_tree", "low", "procfs access; stable PID + start time", _PROCESS, ("process",)),
    "managed_process_to_ready": CollectorContract("managed_service_lifetime", "none", "Process readiness", _ROW, ("process",)),
    "nvml_selected_device": CollectorContract("selected_gpu_device", "medium", "GPU UUID; NVML query support and permission", _RESOURCE),
    "pcap": CollectorContract("correlated_http_connection_frames", "high", "Full; capture permission; complete request matching", "acprof/packet/merge.py", ("docker",)),
    "input_plan": CollectorContract("materialized_request", "none", "frozen input plan", "acprof/host/input_plan/__init__.py", derivation="recorded"),
    "client_http": CollectorContract("client_http_requests", "none", "successful requests; tokenizer/geometry when applicable", _ROW),
    "derived": CollectorContract("source_dependent", "none", "finite source fields and positive denominator", "acprof/host/measurement/collector_metrics.py", derivation="derived"),
    "torch": CollectorContract("independent_model_forward", "high", "explicit profiler; supported backend", "acprof/host/profiling/compute_plan.py", ("docker",)),
    "ncu": CollectorContract("independent_gpu_kernels", "high", "explicit profiler; GPU counter permission", "acprof/host/profiling/compute_plan.py", ("docker",)),
    "massif": CollectorContract("independent_process_heap", "high", "explicit profiler; CPU; Valgrind", "acprof/host/profiling/execution_plan.py", ("docker",)),
    "nsys": CollectorContract("independent_cuda_timeline", "high", "explicit profiler; NVIDIA GPU; Nsys", "acprof/host/profiling/execution_plan.py", ("docker",)),
    "rapl_package": CollectorContract("host_cpu_packages", "medium", "Full; readable RAPL package domains", "acprof/monitors/energy_cpu.py", ("docker",)),
    "rapl_dram": CollectorContract("host_dram_domains", "medium", "Full; DRAM requested; complete readable domains", "acprof/monitors/energy_cpu.py", ("docker",)),
    "rapl_cgroup_attribution": CollectorContract("cgroup_cpu_time_share_of_host_packages", "medium", "Full; RAPL + cgroup + host CPU counters", "acprof/monitors/energy_cpu.py", ("docker",), "estimated"),
    "cgroup": CollectorContract("container_cgroup", "low", "local Docker; cgroup v2; readable metric files", _RESOURCE, ("docker",)),
    "host_cpufreq": CollectorContract("host_visible_cpus", "low", "readable cpufreq or cpuinfo; unavailable on WSL2", _RESOURCE, ("docker",)),
    "perf_stat": CollectorContract("container_cgroup_pmu", "medium", "Full; perf; PMU permission; valid running ratio", "acprof/monitors/perf_mips.py", ("docker",)),
    "docker_and_server": CollectorContract("container_service_lifetime", "none", "Docker readiness timestamps", _ROW, ("docker",)),
    "handler": CollectorContract("actual_request", "none", "handler response contract", _ROW, derivation="recorded"),
}


def collector_contract(metric: Metric) -> CollectorContract:
    """Resolve explicit exceptions; a new source must supply its own contract."""
    contract = SOURCE_CONTRACTS[metric.source]
    if metric.source == "nvml_selected_device":
        if metric.row_group == "gpu_energy" or metric.name.startswith("gpu_idle_") or metric.name in {
            "gpu_energy_source", "gpu_energy_fallback_reason",
        }:
            return replace(contract, producer="acprof/monitors/energy_nvml.py", runtimes=("docker",),
                           prerequisites="Full; NVML counter or power integration; matched control")
        if metric.name not in {"gpu_utilization_status", "gpu_memory_status", "gpu_util_avg_pct",
                               "gpu_util_peak_pct", "gpu_mem_used_avg_bytes", "gpu_mem_used_peak_bytes"}:
            return replace(contract, runtimes=("docker",))
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
    contract = collector_contract(metric)
    implemented = runtime in contract.runtimes
    return {"metric": name, "runtime": runtime, "registered": True,
            "implemented": implemented, "source": metric.source, "unit": metric.unit,
            "window": metric.window, "scope": contract.scope, "derivation": contract.derivation,
            "cost": contract.cost, "prerequisites": contract.prerequisites,
            "producer": contract.producer,
            "collectable": "conditional" if implemented else "unsupported",
            "hardware_validation": "not_verified",
            "missing_reason": "runtime_not_implemented" if not implemented else "requires_current_run_evidence"}
