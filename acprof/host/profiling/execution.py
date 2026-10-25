"""Host-side high-overhead execution profiling for AC-Prof.

Massif and Nsight Systems are deliberately collected outside the normal
benchmark path: both tools materially perturb latency.  Every tool and input
scale is isolated so a missing profiler or an invalid report remains a
diagnostic entry rather than aborting the other profiler.
"""
from __future__ import annotations

import copy
import logging
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

from acprof.host.detect import TaskInfo
from acprof.host.profiling.progress import (
    ProfilerProgressCallback,
    report_profiler_completion,
)
from acprof.host.profiling.support import (
    load_input_scale_plan_entries,
    write_profile_json,
)
from acprof.host.profiling.tools import massif, nsys
from acprof.host.profiling.tools.execution_environment import (
    get_massif_version,
    get_nsys_version,
    require_execution_image,
    validate_nsys_container_runtime,
)
from acprof.host.profiling.tools.massif import MASSIF_TOOL
from acprof.host.profiling.tools.nsys import NSYS_TOOL
from acprof.host.profiling.tools.tool_discovery import find_nsys_executable, find_nsys_mount_root

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


def _normalize_gpu_modes(gpu_list: Iterable[str]) -> list[str]:
    modes: list[str] = []
    for gpu in gpu_list:
        mode = "on" if str(gpu).strip().lower() == "on" else "off"
        if mode not in modes:
            modes.append(mode)
    return modes


def _normalize_resources(values: Iterable[int], name: str) -> list[int]:
    normalized: list[int] = []
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
    requested: int | None,
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
    reference_cpu: int | None,
    reference_mem: int | None,
) -> tuple[list[tuple[int, int]], int | None, int | None]:
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
    reference_cpu: int | None,
    reference_mem: int | None,
) -> tuple[int, int]:
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
) -> dict[str, Any]:
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


@dataclass(frozen=True)
class ExecutionProfileRequest:
    """Workload and resource matrix whose independent profiles are collected."""

    task_info: TaskInfo
    image_tag: str
    cpu_list: list[int]
    mem_list: list[int]
    gpu_list: list[str]
    output_dir: str
    input_scale_plan_file: str


@dataclass(frozen=True)
class ProfilerSampling:
    """Sampling resources and request count for one profiler."""

    sampling: str = "per-scale"
    repeat: int = 1
    reference_cpu: int | None = None
    reference_mem: int | None = None


@dataclass(frozen=True)
class ExecutionProfileOptions:
    tool_mode: str = "both"
    massif: ProfilerSampling = ProfilerSampling()
    nsys: ProfilerSampling = ProfilerSampling(sampling="per-cpu-scale")
    nsys_root: str | None = None
    keep_profiles: bool = True
    resume_existing_profiles: bool = False


@dataclass(frozen=True)
class _ToolPlan:
    tool: str
    enabled: bool
    sampling: str
    repeat: int
    sources: list[tuple[int, int]]
    reference_cpu: int | None
    reference_mem: int | None


@dataclass(frozen=True)
class _ExecutionPlan:
    cpus: list[int]
    memories: list[int]
    gpu_modes: list[str]
    tool_mode: str
    massif: _ToolPlan
    nsys: _ToolPlan

    @property
    def enabled_tools(self) -> list[str]:
        return [tool.tool for tool in (self.massif, self.nsys) if tool.enabled]


@dataclass(frozen=True)
class _ProfileWorkspace:
    output_dir: str
    profile_root: str
    plan_path: str
    entries: list[dict[str, Any]]


@dataclass(frozen=True)
class _MassifRuntime:
    image: str | None
    error: str
    version: str


@dataclass(frozen=True)
class _NsysRuntime:
    image: str
    binary: str | None
    mount_root: str | None
    error: str
    version: str


def _tool_plan(
    tool: str, enabled: bool, settings: ProfilerSampling,
    cpus: Sequence[int], memories: Sequence[int],
) -> _ToolPlan:
    sources: list[tuple[int, int]] = []
    reference_cpu = reference_mem = None
    if enabled:
        sources, reference_cpu, reference_mem = _sampled_resource_cases(
            tool=tool, cpus=cpus, memories=memories, sampling=settings.sampling,
            reference_cpu=settings.reference_cpu, reference_mem=settings.reference_mem,
        )
    return _ToolPlan(
        tool, enabled, settings.sampling, max(1, int(settings.repeat)),
        sources, reference_cpu, reference_mem,
    )


def _execution_plan(
    request: ExecutionProfileRequest, options: ExecutionProfileOptions,
) -> _ExecutionPlan:
    mode = (options.tool_mode or "both").strip().lower()
    if mode not in EXECUTION_PROFILE_TOOL_MODES:
        raise ValueError(
            "tool_mode must be one of "
            f"{', '.join(sorted(EXECUTION_PROFILE_TOOL_MODES))}, got {options.tool_mode!r}"
        )
    cpus = _normalize_resources(request.cpu_list, "cpu_list")
    memories = _normalize_resources(request.mem_list, "mem_list")
    gpu_modes = _normalize_gpu_modes(request.gpu_list)
    massif_settings = replace(options.massif, sampling=_normalize_sampling_mode(
        options.massif.sampling, name="massif_sampling", allowed=MASSIF_SAMPLING_MODES,
    ))
    nsys_settings = replace(options.nsys, sampling=_normalize_sampling_mode(
        options.nsys.sampling, name="nsys_sampling", allowed=NSYS_SAMPLING_MODES,
    ))
    return _ExecutionPlan(
        cpus, memories, gpu_modes, mode,
        _tool_plan(MASSIF_TOOL, mode in {"both", MASSIF_TOOL} and "off" in gpu_modes,
                   massif_settings, cpus, memories),
        _tool_plan(NSYS_TOOL, mode in {"both", NSYS_TOOL} and "on" in gpu_modes,
                   nsys_settings, cpus, memories),
    )


def _profile_workspace(request: ExecutionProfileRequest) -> _ProfileWorkspace:
    from acprof.artifact_layout import ArtifactLayout

    entries = load_input_scale_plan_entries(request.input_scale_plan_file)
    output_dir = os.path.abspath(os.fspath(request.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    layout = ArtifactLayout.discover(output_dir)
    return _ProfileWorkspace(
        output_dir, str(layout.path(EXECUTION_PROFILE_DIRNAME)),
        str(layout.path(EXECUTION_PROFILE_PLAN_NAME)), entries,
    )


def _announce_collection(plan: _ExecutionPlan, workspace: _ProfileWorkspace) -> None:
    if not plan.enabled_tools:
        return
    os.makedirs(workspace.profile_root, exist_ok=True)
    count = len(workspace.entries) * (len(plan.massif.sources) + len(plan.nsys.sources))
    print(
        f"[execution-profile] Collecting {count} sampled isolated probe(s) "
        "before the normal CSV sweep; result CSV files appear after this stage completes."
    )
    for tool in (plan.massif, plan.nsys):
        if tool.enabled:
            print(f"[execution-profile][{tool.tool}] Sampling={tool.sampling}, resources={tool.sources}")


def _prepare_massif(request: ExecutionProfileRequest, plan: _ToolPlan) -> _MassifRuntime:
    from acprof.capabilities import declared_profiler_error

    image = None
    error = declared_profiler_error(request.task_info, MASSIF_TOOL) if plan.enabled else ""
    version = "unknown"
    if plan.enabled and not error:
        try:
            image = require_execution_image(request.image_tag, MASSIF_TOOL)
        except Exception as exc:
            error = str(exc)
            if not error.startswith("massif_"):
                error = f"massif_runtime_check_failed:{exc!r}"
        if image:
            version = get_massif_version(image)
    return _MassifRuntime(image, error, version)


def _prepare_nsys(
    request: ExecutionProfileRequest, plan: _ToolPlan, root: str | None,
) -> _NsysRuntime:
    from acprof.capabilities import declared_profiler_error

    binary = mount_root = None
    image = request.image_tag
    error = declared_profiler_error(request.task_info, NSYS_TOOL) if plan.enabled else ""
    version = "unknown"
    if plan.enabled and not error:
        try:
            binary = find_nsys_executable(root)
        except Exception as exc:
            error = f"nsys_discovery_failed:{exc!r}"
        if binary:
            try:
                mount_root = find_nsys_mount_root(binary)
            except Exception as exc:
                error = f"nsys_mount_failed:{exc!r}"
            version = get_nsys_version(binary)
            if mount_root and not error:
                try:
                    image = require_execution_image(request.image_tag, NSYS_TOOL)
                    validate_nsys_container_runtime(image, mount_root)
                    logging.getLogger(__name__).debug("nsys importer preflight passed: image=%s", image)
                except Exception as exc:
                    detail = str(exc)
                    error = detail if detail.startswith("nsys_") else f"nsys_runtime_preflight_failed:{exc!r}"
        elif not error:
            error = "nsys_not_found"
    return _NsysRuntime(image, binary, mount_root, error, version)


def _collect_massif(
    request: ExecutionProfileRequest, plan: _ToolPlan, runtime: _MassifRuntime,
    workspace: _ProfileWorkspace, *, resume: bool,
    progress_callback: ProfilerProgressCallback | None,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    profiles: dict[tuple[str, int, int], dict[str, Any]] = {}
    started = perf_counter()
    for cpu, mem in plan.sources:
        profiles[(MASSIF_TOOL, cpu, mem)] = massif.profile(
            entries=workspace.entries, global_error=runtime.error,
            task_info=request.task_info, derived_image=runtime.image, cpu=cpu, mem=mem,
            payload_file=request.input_scale_plan_file, profile_root=workspace.profile_root,
            output_dir=workspace.output_dir, repeat=plan.repeat, resume_existing=resume,
        )
    if plan.enabled:
        report_profiler_completion(
            progress_callback, profiler="Massif", profiles=profiles.values(),
            elapsed_seconds=perf_counter() - started,
        )
    return profiles


def _collect_nsys(
    request: ExecutionProfileRequest, plan: _ToolPlan, runtime: _NsysRuntime,
    workspace: _ProfileWorkspace, *, progress_callback: ProfilerProgressCallback | None,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    profiles: dict[tuple[str, int, int], dict[str, Any]] = {}
    started = perf_counter()
    for cpu, mem in plan.sources:
        profiles[(NSYS_TOOL, cpu, mem)] = nsys.profile(
            entries=workspace.entries, global_error=runtime.error,
            task_info=request.task_info, image_tag=runtime.image,
            nsys_bin=runtime.binary, nsys_mount_root=runtime.mount_root, cpu=cpu, mem=mem,
            payload_file=request.input_scale_plan_file, profile_root=workspace.profile_root,
            output_dir=workspace.output_dir, repeat=plan.repeat,
        )
    if plan.enabled:
        report_profiler_completion(
            progress_callback, profiler="Nsys", profiles=profiles.values(),
            elapsed_seconds=perf_counter() - started,
        )
    return profiles


def _expand_profiles(
    plan: _ExecutionPlan, sources: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for cpu in plan.cpus:
        for mem in plan.memories:
            for gpu in plan.gpu_modes:
                tool = plan.massif if gpu == "off" else plan.nsys
                if not tool.enabled:
                    continue
                source_cpu, source_mem = _profile_source_resource(
                    cpu=cpu, mem=mem, sampling=tool.sampling,
                    reference_cpu=tool.reference_cpu, reference_mem=tool.reference_mem,
                )
                profile = _copy_profile_with_provenance(
                    sources[(tool.tool, source_cpu, source_mem)],
                    source_cpu=source_cpu, source_mem=source_mem, sampling=tool.sampling,
                )
                profiles.append({
                    "cpu_cores": cpu, "mem_cap_gb": mem, "gpu_mode": gpu,
                    "tools": {tool.tool: profile},
                })
    return profiles


def _profile_static_metadata(
    plan: _ExecutionPlan, massif_runtime: _MassifRuntime, nsys_runtime: _NsysRuntime,
    *, keep_profiles: bool,
) -> dict[str, Any]:
    massif_plan, nsys_plan = plan.massif, plan.nsys
    return {
        "execution_profile_schema_version": EXECUTION_PROFILE_SCHEMA_VERSION,
        "execution_profile_tools": plan.enabled_tools,
        "massif_peak_semantics": (
            "process_lifetime_including_model_load_and_warmup; "
            "independent_component_maxima; total=max_snapshot("
            "heap+heap_extra+stack), time=that_snapshot"
        ),
        "massif_repeat": massif_plan.repeat if massif_plan.enabled else None,
        "massif_version": massif_runtime.version,
        "massif_sampling_strategy": (
            SAMPLING_STRATEGY_METADATA[massif_plan.sampling] if massif_plan.enabled else None
        ),
        "massif_reference_cpu_cores": massif_plan.reference_cpu,
        "massif_reference_mem_cap_gb": massif_plan.reference_mem,
        "massif_reused_across_resource_cases": bool(massif_plan.enabled and massif_plan.sampling != "full"),
        "nsys_timeline_semantics": (
            "NVTX acprof_compute capture; CUDA API, kernel, and memcpy "
            "sums/counts normalized per request"
        ),
        "nsys_repeat": nsys_plan.repeat if nsys_plan.enabled else None,
        "nsys_version": nsys_runtime.version,
        "nsys_sampling_strategy": (
            SAMPLING_STRATEGY_METADATA[nsys_plan.sampling] if nsys_plan.enabled else None
        ),
        "nsys_reference_cpu_cores": nsys_plan.reference_cpu,
        "nsys_reference_mem_cap_gb": nsys_plan.reference_mem,
        "nsys_reused_across_resource_cases": bool(nsys_plan.enabled and nsys_plan.sampling != "full"),
        "execution_profiles_retained": bool(keep_profiles and plan.enabled_tools),
        "execution_profile_provenance": "collected" if plan.enabled_tools else "disabled",
    }


def collect_execution_profile_plan(
    request: ExecutionProfileRequest,
    options: ExecutionProfileOptions = ExecutionProfileOptions(),
    *,
    progress_callback: ProfilerProgressCallback | None = None,
) -> str:
    """Collect independent tool samples and publish their resource-grid plan.

    Runtime validation belongs to the caller. Tool setup, execution, expansion,
    and publication remain separate from normal measurement windows.
    """
    plan = _execution_plan(request, options)
    workspace = _profile_workspace(request)
    _announce_collection(plan, workspace)
    massif_runtime = _prepare_massif(request, plan.massif)
    nsys_runtime = _prepare_nsys(request, plan.nsys, options.nsys_root)
    sources = _collect_massif(
        request, plan.massif, massif_runtime, workspace,
        resume=options.resume_existing_profiles, progress_callback=progress_callback,
    )
    sources.update(_collect_nsys(
        request, plan.nsys, nsys_runtime, workspace, progress_callback=progress_callback,
    ))
    profiles = _expand_profiles(plan, sources)
    metadata = _profile_static_metadata(
        plan, massif_runtime, nsys_runtime, keep_profiles=options.keep_profiles,
    )
    payload = {
        "schema_version": EXECUTION_PROFILE_SCHEMA_VERSION,
        "model_id": request.task_info.model_id,
        "model_revision": request.task_info.model_revision or "main",
        "task_family": request.task_info.task_family,
        "pipeline_tag": request.task_info.pipeline_tag,
        "runtime_backend": request.task_info.runtime_backend,
        "execution_profile_tool_mode": plan.tool_mode,
        "massif_sampling": plan.massif.sampling if plan.massif.enabled else None,
        "nsys_sampling": plan.nsys.sampling if plan.nsys.enabled else None,
        "static_metadata": metadata,
        "profiles": profiles,
    }
    if not options.keep_profiles and plan.enabled_tools:
        _strip_artifact_paths(profiles)
        shutil.rmtree(workspace.profile_root, ignore_errors=True)
    write_profile_json(workspace.plan_path, payload)
    print(f"[execution] Execution profile plan: {workspace.plan_path}")
    return workspace.plan_path
