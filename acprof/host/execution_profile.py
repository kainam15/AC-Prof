"""Host-side high-overhead execution profiling for AC-Prof.

Massif and Nsight Systems are deliberately collected outside the normal
benchmark path: both tools materially perturb latency.  Every tool and input
scale is isolated so a missing profiler or an invalid report remains a
diagnostic entry rather than aborting the other profiler.
"""
from __future__ import annotations

import copy
import os
import shutil
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from acprof.host.detect import TaskInfo
from acprof.host.profiler_progress import (
    ProfilerProgressCallback,
    report_profiler_completion,
)
from acprof.host.profiler_support import (
    load_input_scale_plan_entries,
    write_profile_json,
)
from acprof.host.profilers import massif, nsys
from acprof.host.profilers.execution_environment import (
    get_massif_version,
    get_nsys_version,
    require_execution_image,
    validate_nsys_container_runtime,
)
from acprof.host.profilers.massif import MASSIF_TOOL
from acprof.host.profilers.nsys import NSYS_TOOL
from acprof.host.profilers.tool_discovery import find_nsys_executable, find_nsys_mount_root

EXECUTION_PROFILE_PLAN_NAME = "execution_profile_plan.json"
EXECUTION_PROFILE_DIRNAME = "execution_profiles"
EXECUTION_PROFILE_SCHEMA_VERSION = 1
EXECUTION_PROFILE_TOOL_MODES = {"none", "both", "massif", "nsys"}
MASSIF_SAMPLING_MODES = {"per-scale", "full"}
NSYS_SAMPLING_MODES = {"per-cpu-scale", "per-scale", "full"}
SAMPLING_STRATEGY_METADATA = {
    "full": "full_resource_matrix",
    "per-scale": "representative_per_scale",
    "per-cpu-scale": "representative_per_cpu_scale",
}


def _normalize_gpu_modes(gpu_list: Iterable[str]) -> List[str]:
    modes: List[str] = []
    for gpu in gpu_list:
        mode = "on" if str(gpu).strip().lower() == "on" else "off"
        if mode not in modes:
            modes.append(mode)
    return modes


def _normalize_resources(values: Iterable[int], name: str) -> List[int]:
    normalized: List[int] = []
    for value in values:
        integer = int(value)
        if integer <= 0:
            raise ValueError(f"{name} values must be positive, got {value!r}")
        if integer not in normalized:
            normalized.append(integer)
    return normalized


def _normalize_sampling_mode(
    value: str,
    *,
    name: str,
    allowed: Iterable[str],
) -> str:
    normalized = str(value or "").strip().lower()
    allowed_values = set(allowed)
    if normalized not in allowed_values:
        raise ValueError(
            f"{name} must be one of {', '.join(sorted(allowed_values))}, "
            f"got {value!r}"
        )
    return normalized


def _resolve_reference_resource(
    values: Sequence[int],
    requested: Optional[int],
    *,
    name: str,
) -> int:
    reference = max(values) if requested is None else int(requested)
    if reference not in values:
        raise ValueError(
            f"{name}={reference} is not present in the requested resource "
            f"matrix {list(values)}"
        )
    return reference


def _sampled_resource_cases(
    *,
    tool: str,
    cpus: Sequence[int],
    memories: Sequence[int],
    sampling: str,
    reference_cpu: Optional[int],
    reference_mem: Optional[int],
) -> Tuple[List[Tuple[int, int]], Optional[int], Optional[int]]:
    """Return actual profiler resources and resolved representative values."""
    if sampling == "full":
        return (
            [(cpu, mem) for cpu in cpus for mem in memories],
            None,
            None,
        )

    resolved_mem = _resolve_reference_resource(
        memories,
        reference_mem,
        name=f"{tool}_reference_mem",
    )
    if tool == NSYS_TOOL and sampling == "per-cpu-scale":
        return ([(cpu, resolved_mem) for cpu in cpus], None, resolved_mem)

    resolved_cpu = _resolve_reference_resource(
        cpus,
        reference_cpu,
        name=f"{tool}_reference_cpu",
    )
    return ([(resolved_cpu, resolved_mem)], resolved_cpu, resolved_mem)


def _profile_source_resource(
    *,
    cpu: int,
    mem: int,
    sampling: str,
    reference_cpu: Optional[int],
    reference_mem: Optional[int],
) -> Tuple[int, int]:
    if sampling == "full":
        return cpu, mem
    if sampling == "per-cpu-scale":
        if reference_mem is None:
            raise ValueError("per-cpu-scale sampling requires reference memory")
        return cpu, reference_mem
    if reference_cpu is None or reference_mem is None:
        raise ValueError("per-scale sampling requires reference CPU and memory")
    return reference_cpu, reference_mem


def _copy_profile_with_provenance(
    profile: Mapping[str, Any],
    *,
    source_cpu: int,
    source_mem: int,
    sampling: str,
) -> Dict[str, Any]:
    copied = copy.deepcopy(dict(profile))
    strategy = SAMPLING_STRATEGY_METADATA[sampling]
    entries = copied.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry["profile_source_cpu_cores"] = source_cpu
            entry["profile_source_mem_cap_gb"] = source_mem
            entry["profile_sampling_strategy"] = strategy
    return copied


def _strip_artifact_paths(profiles: Sequence[Mapping[str, Any]]) -> None:
    for profile in profiles:
        tools = profile.get("tools")
        if not isinstance(tools, Mapping):
            continue
        for tool_profile in tools.values():
            if not isinstance(tool_profile, Mapping):
                continue
            entries = tool_profile.get("entries")
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict) and "report" in entry:
                    entry["report"] = None


def collect_execution_profile_plan(
    task_info: TaskInfo,
    image_tag: str,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    output_dir: str,
    input_scale_plan_file: str,
    project_dir: str,
    tool_mode: str = "both",
    massif_repeat: int = 1,
    nsys_repeat: int = 1,
    nsys_root: Optional[str] = None,
    keep_profiles: bool = True,
    massif_sampling: str = "per-scale",
    massif_reference_cpu: Optional[int] = None,
    massif_reference_mem: Optional[int] = None,
    nsys_sampling: str = "per-cpu-scale",
    nsys_reference_cpu: Optional[int] = None,
    nsys_reference_mem: Optional[int] = None,
    resume_existing_profiles: bool = False,
    progress_callback: Optional[ProfilerProgressCallback] = None,
) -> str:
    """Collect sampled probes and expand them to a full resource-grid plan.

    The calling CLI must run isolated runtime validation before collection.
    This internal planner does not repeat output validation inside profilers.
    """
    normalized_tool_mode = (tool_mode or "both").strip().lower()
    if normalized_tool_mode not in EXECUTION_PROFILE_TOOL_MODES:
        raise ValueError(
            "tool_mode must be one of "
            f"{', '.join(sorted(EXECUTION_PROFILE_TOOL_MODES))}, "
            f"got {tool_mode!r}"
        )

    cpus = _normalize_resources(cpu_list, "cpu_list")
    memories = _normalize_resources(mem_list, "mem_list")
    gpu_modes = _normalize_gpu_modes(gpu_list)
    normalized_massif_sampling = _normalize_sampling_mode(
        massif_sampling,
        name="massif_sampling",
        allowed=MASSIF_SAMPLING_MODES,
    )
    normalized_nsys_sampling = _normalize_sampling_mode(
        nsys_sampling,
        name="nsys_sampling",
        allowed=NSYS_SAMPLING_MODES,
    )
    entries = load_input_scale_plan_entries(input_scale_plan_file)
    normalized_massif_repeat = max(1, int(massif_repeat))
    normalized_nsys_repeat = max(1, int(nsys_repeat))
    output_dir = os.path.abspath(os.fspath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    from acprof.artifact_layout import ArtifactLayout
    layout = ArtifactLayout.discover(output_dir)
    profile_root = str(layout.path(EXECUTION_PROFILE_DIRNAME))

    collect_massif = (
        normalized_tool_mode in {"both", MASSIF_TOOL}
        and "off" in gpu_modes
    )
    collect_nsys = (
        normalized_tool_mode in {"both", NSYS_TOOL}
        and "on" in gpu_modes
    )
    massif_sources: List[Tuple[int, int]] = []
    massif_reference: Tuple[Optional[int], Optional[int]] = (None, None)
    if collect_massif:
        (
            massif_sources,
            massif_reference_cpu_resolved,
            massif_reference_mem_resolved,
        ) = _sampled_resource_cases(
            tool=MASSIF_TOOL,
            cpus=cpus,
            memories=memories,
            sampling=normalized_massif_sampling,
            reference_cpu=massif_reference_cpu,
            reference_mem=massif_reference_mem,
        )
        massif_reference = (
            massif_reference_cpu_resolved,
            massif_reference_mem_resolved,
        )

    nsys_sources: List[Tuple[int, int]] = []
    nsys_reference: Tuple[Optional[int], Optional[int]] = (None, None)
    if collect_nsys:
        (
            nsys_sources,
            nsys_reference_cpu_resolved,
            nsys_reference_mem_resolved,
        ) = _sampled_resource_cases(
            tool=NSYS_TOOL,
            cpus=cpus,
            memories=memories,
            sampling=normalized_nsys_sampling,
            reference_cpu=nsys_reference_cpu,
            reference_mem=nsys_reference_mem,
        )
        nsys_reference = (
            nsys_reference_cpu_resolved,
            nsys_reference_mem_resolved,
        )

    if collect_massif or collect_nsys:
        os.makedirs(profile_root, exist_ok=True)
        probe_count = len(entries) * (
            len(massif_sources) + len(nsys_sources)
        )
        print(
            "[execution-profile] Collecting "
            f"{probe_count} sampled isolated probe(s) before the normal CSV "
            "sweep; "
            "result CSV files appear after this stage completes."
        )
        if collect_massif:
            print(
                "[execution-profile][massif] Sampling="
                f"{normalized_massif_sampling}, resources={massif_sources}"
            )
        if collect_nsys:
            print(
                "[execution-profile][nsys] Sampling="
                f"{normalized_nsys_sampling}, resources={nsys_sources}"
            )

    derived_image: Optional[str] = None
    from acprof.capabilities import declared_profiler_error
    massif_error = declared_profiler_error(task_info, MASSIF_TOOL) if collect_massif else ""
    massif_version = "unknown"
    if collect_massif and not massif_error:
        try:
            derived_image = require_execution_image(image_tag, MASSIF_TOOL)
        except Exception as exc:
            massif_error = str(exc)
            if not massif_error.startswith("massif_"):
                massif_error = f"massif_runtime_check_failed:{exc!r}"
        if derived_image:
            massif_version = get_massif_version(derived_image)

    nsys_bin: Optional[str] = None
    nsys_mount_root: Optional[str] = None
    nsys_profile_image = image_tag
    nsys_error = declared_profiler_error(task_info, NSYS_TOOL) if collect_nsys else ""
    nsys_version = "unknown"
    if collect_nsys and not nsys_error:
        try:
            nsys_bin = find_nsys_executable(nsys_root)
        except Exception as exc:
            nsys_error = f"nsys_discovery_failed:{exc!r}"
        if nsys_bin:
            try:
                nsys_mount_root = find_nsys_mount_root(nsys_bin)
            except Exception as exc:
                nsys_error = f"nsys_mount_failed:{exc!r}"
            nsys_version = get_nsys_version(nsys_bin)
            if nsys_mount_root and not nsys_error:
                try:
                    nsys_profile_image = require_execution_image(image_tag, NSYS_TOOL)
                    validate_nsys_container_runtime(
                        nsys_profile_image,
                        nsys_mount_root,
                    )
                    print(
                        "[execution-profile][nsys] QdstrmImporter preflight "
                        f"passed in {nsys_profile_image}"
                    )
                except Exception as exc:
                    detail = str(exc)
                    nsys_error = (
                        detail
                        if detail.startswith("nsys_")
                        else f"nsys_runtime_preflight_failed:{exc!r}"
                    )
        elif not nsys_error:
            nsys_error = "nsys_not_found"

    source_profiles: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    massif_started = perf_counter()
    for cpu, mem in massif_sources:
        source_profiles[(MASSIF_TOOL, cpu, mem)] = massif.profile(
            entries=entries,
            global_error=massif_error,
            task_info=task_info,
            derived_image=derived_image,
            cpu=cpu,
            mem=mem,
            payload_file=input_scale_plan_file,
            profile_root=profile_root,
            output_dir=output_dir,
            repeat=normalized_massif_repeat,
            resume_existing=resume_existing_profiles,
        )
    if collect_massif:
        report_profiler_completion(
            progress_callback,
            profiler="Massif",
            profiles=(
                source_profiles[(MASSIF_TOOL, cpu, mem)]
                for cpu, mem in massif_sources
            ),
            elapsed_seconds=perf_counter() - massif_started,
        )
    nsys_started = perf_counter()
    for cpu, mem in nsys_sources:
        source_profiles[(NSYS_TOOL, cpu, mem)] = nsys.profile(
            entries=entries,
            global_error=nsys_error,
            task_info=task_info,
            image_tag=nsys_profile_image,
            nsys_bin=nsys_bin,
            nsys_mount_root=nsys_mount_root,
            cpu=cpu,
            mem=mem,
            payload_file=input_scale_plan_file,
            profile_root=profile_root,
            output_dir=output_dir,
            repeat=normalized_nsys_repeat,
        )
    if collect_nsys:
        report_profiler_completion(
            progress_callback,
            profiler="Nsys",
            profiles=(
                source_profiles[(NSYS_TOOL, cpu, mem)]
                for cpu, mem in nsys_sources
            ),
            elapsed_seconds=perf_counter() - nsys_started,
        )

    profiles: List[Dict[str, Any]] = []
    for cpu in cpus:
        for mem in memories:
            for gpu_mode in gpu_modes:
                tools: Dict[str, Any] = {}
                if gpu_mode == "off" and collect_massif:
                    source_cpu, source_mem = _profile_source_resource(
                        cpu=cpu,
                        mem=mem,
                        sampling=normalized_massif_sampling,
                        reference_cpu=massif_reference[0],
                        reference_mem=massif_reference[1],
                    )
                    tools[MASSIF_TOOL] = _copy_profile_with_provenance(
                        source_profiles[(MASSIF_TOOL, source_cpu, source_mem)],
                        source_cpu=source_cpu,
                        source_mem=source_mem,
                        sampling=normalized_massif_sampling,
                    )
                if gpu_mode == "on" and collect_nsys:
                    source_cpu, source_mem = _profile_source_resource(
                        cpu=cpu,
                        mem=mem,
                        sampling=normalized_nsys_sampling,
                        reference_cpu=nsys_reference[0],
                        reference_mem=nsys_reference[1],
                    )
                    tools[NSYS_TOOL] = _copy_profile_with_provenance(
                        source_profiles[(NSYS_TOOL, source_cpu, source_mem)],
                        source_cpu=source_cpu,
                        source_mem=source_mem,
                        sampling=normalized_nsys_sampling,
                    )
                if tools:
                    profiles.append(
                        {
                            "cpu_cores": cpu,
                            "mem_cap_gb": mem,
                            "gpu_mode": gpu_mode,
                            "tools": tools,
                        }
                    )

    enabled_tools = [
        tool
        for tool, enabled in (
            (MASSIF_TOOL, collect_massif),
            (NSYS_TOOL, collect_nsys),
        )
        if enabled
    ]
    static_metadata = {
        "execution_profile_schema_version": EXECUTION_PROFILE_SCHEMA_VERSION,
        "execution_profile_tools": enabled_tools,
        "massif_peak_semantics": (
            "process_lifetime_including_model_load_and_warmup; "
            "independent_component_maxima; total=max_snapshot("
            "heap+heap_extra+stack), time=that_snapshot"
        ),
        "massif_repeat": (
            normalized_massif_repeat if collect_massif else None
        ),
        "massif_version": massif_version,
        "massif_sampling_strategy": (
            SAMPLING_STRATEGY_METADATA[normalized_massif_sampling]
            if collect_massif
            else None
        ),
        "massif_reference_cpu_cores": (
            massif_reference[0] if collect_massif else None
        ),
        "massif_reference_mem_cap_gb": (
            massif_reference[1] if collect_massif else None
        ),
        "massif_reused_across_resource_cases": bool(
            collect_massif and normalized_massif_sampling != "full"
        ),
        "nsys_timeline_semantics": (
            "NVTX acprof_compute capture; CUDA API, kernel, and memcpy "
            "sums/counts normalized per request"
        ),
        "nsys_repeat": normalized_nsys_repeat if collect_nsys else None,
        "nsys_version": nsys_version,
        "nsys_sampling_strategy": (
            SAMPLING_STRATEGY_METADATA[normalized_nsys_sampling]
            if collect_nsys
            else None
        ),
        "nsys_reference_cpu_cores": (
            nsys_reference[0] if collect_nsys else None
        ),
        "nsys_reference_mem_cap_gb": (
            nsys_reference[1] if collect_nsys else None
        ),
        "nsys_reused_across_resource_cases": bool(
            collect_nsys and normalized_nsys_sampling != "full"
        ),
        "execution_profiles_retained": bool(keep_profiles and enabled_tools),
        "execution_profile_provenance": (
            "collected" if enabled_tools else "disabled"
        ),
    }
    plan = {
        "schema_version": EXECUTION_PROFILE_SCHEMA_VERSION,
        "model_id": task_info.model_id,
        "model_revision": task_info.model_revision or "main",
        "task_family": task_info.task_family,
        "pipeline_tag": task_info.pipeline_tag,
        "runtime_backend": task_info.runtime_backend,
        "execution_profile_tool_mode": normalized_tool_mode,
        "massif_sampling": (
            normalized_massif_sampling if collect_massif else None
        ),
        "nsys_sampling": normalized_nsys_sampling if collect_nsys else None,
        "static_metadata": static_metadata,
        "profiles": profiles,
    }
    plan_path = str(layout.path(EXECUTION_PROFILE_PLAN_NAME))

    if not keep_profiles and enabled_tools:
        _strip_artifact_paths(profiles)
        shutil.rmtree(profile_root, ignore_errors=True)

    write_profile_json(plan_path, plan)
    print(f"[execution] Execution profile plan: {plan_path}")
    return plan_path
