"""Host-side FLOP profiling plan generation for AC-Prof."""
from __future__ import annotations

import logging
import os
import shutil
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from acprof.capabilities import declared_profiler_error
from acprof.config import DEFAULT_COMPUTE_PROFILE_TOOL
from acprof.host.command import run_command
from acprof.host.detect import TaskInfo
from acprof.host.profiler_progress import (
    ProfilerProgressCallback,
    report_profiler_completion,
)
from acprof.host.profiler_support import load_input_scale_plan_entries, write_profile_json
from acprof.host.profilers import advisor, ncu, torch
from acprof.host.profilers.tool_discovery import find_executable

_LOG = logging.getLogger(__name__)


COMPUTE_PROFILE_PLAN_NAME = "compute_profile_plan.json"
COMPUTE_PROFILE_TOOL_MODES = {"none", "both", "ncu", "torch", "vendor"}


def _normal_gpu_mode(gpu: str) -> str:
    return "on" if str(gpu).lower() == "on" else "off"


def _host_logical_cpus() -> int:
    return max(1, int(os.cpu_count() or 1))


def _host_memory_gb_fraction(fraction: float = 0.75) -> int:
    total_bytes = 0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_bytes = int(line.split()[1]) * 1024
                    break
    except OSError:
        total_bytes = 0

    if total_bytes <= 0:
        return 1
    return max(1, int((total_bytes * fraction) // (1024 ** 3)))


def _default_compute_profile_resources(
    compute_profile_cpus: Optional[int],
    compute_profile_mem: Optional[int],
) -> Tuple[int, int]:
    cpu = max(1, int(compute_profile_cpus or _host_logical_cpus()))
    mem = max(1, int(compute_profile_mem or _host_memory_gb_fraction(0.75)))
    return cpu, mem


def _failed_tool_profile(
    *,
    tool: str,
    entries: List[Dict[str, Any]],
    repeat: int,
    error: str,
) -> Dict[str, Any]:
    error_entries = (
        ncu._ncu_error_entries(entries, error)
        if tool == ncu.NCU_TOOL
        else torch._torch_error_entries(entries, error)
        if tool == torch.TORCH_PROFILER_TOOL
        else advisor._tool_error_entries(entries, error, tool)
    )
    return {
        "tool": tool,
        "repeat": max(1, int(repeat)),
        "error": error,
        "entries": error_entries,
    }


def _safe_profile_tool(
    *,
    tool: str,
    entries: List[Dict[str, Any]],
    repeat: int,
    callback: Any,
    task_info: Optional[TaskInfo] = None,
) -> Dict[str, Any]:
    try:
        if task_info is not None:
            error = declared_profiler_error(task_info, tool)
            if error:
                return _failed_tool_profile(tool=tool, entries=entries, repeat=repeat, error=error)
        return callback()
    except Exception as exc:
        _LOG.debug("profiler failed: tool=%s error_type=%s", tool, type(exc).__name__)
        return _failed_tool_profile(
            tool=tool,
            entries=entries,
            repeat=repeat,
            error=f"{tool}_failed:{exc!r}",
        )


def _executable_version(executable: Optional[str]) -> str:
    if not executable:
        return "unknown"
    try:
        result = run_command([executable, "--version"], check=False)
    except Exception:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    text = (result.stdout or result.stderr or "").strip()
    if not text:
        return "unknown"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "unknown"


def _profile_tool_maps(profiles: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for device_profile in profiles.values():
        if not isinstance(device_profile, dict):
            continue
        for tool in (torch.TORCH_PROFILER_TOOL, ncu.NCU_TOOL, "intel_advisor"):
            profile = device_profile.get(tool)
            if isinstance(profile, dict):
                yield profile


def _first_profile_value(
    profiles: Dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    for profile in _profile_tool_maps(profiles):
        value = profile.get(key)
        if value not in (None, "", "unknown"):
            return value
        for entry in profile.get("entries", []):
            if not isinstance(entry, dict):
                continue
            value = entry.get(key)
            if value not in (None, "", "unknown"):
                if not isinstance(value, float) or value == value:
                    return value
    return default


def _strip_discarded_profile_paths(profiles: Dict[str, Any]) -> None:
    for profile in _profile_tool_maps(profiles):
        for entry in profile.get("entries", []):
            if not isinstance(entry, dict):
                continue
            if entry.get("tool") == ncu.NCU_TOOL:
                entry["report"] = None


def collect_compute_profile_plan(
    *,
    task_info: TaskInfo,
    image_tag: str,
    cpu_list: List[int],
    mem_list: List[int],
    gpu_list: List[str],
    output_dir: str,
    input_scale_plan_file: str,
    advisor_root: Optional[str],
    ncu_root: Optional[str],
    advisor_repeat: int,
    ncu_repeat: int,
    keep_profiles: bool,
    compute_profile_cpus: Optional[int] = None,
    compute_profile_mem: Optional[int] = None,
    compute_profile_tool: str = DEFAULT_COMPUTE_PROFILE_TOOL,
    torch_profiler_repeat: int = 1,
    resume_existing_ncu_profiles: bool = False,
    progress_callback: Optional[ProfilerProgressCallback] = None,
) -> str:
    """Collect or synthesize compute profiles and write a plan file.

    The calling CLI must run isolated runtime validation before collection.
    This internal planner does not repeat output validation inside profilers.
    """
    os.makedirs(output_dir, exist_ok=True)
    tool_mode = (
        compute_profile_tool or DEFAULT_COMPUTE_PROFILE_TOOL
    ).strip().lower()
    if tool_mode not in COMPUTE_PROFILE_TOOL_MODES:
        raise ValueError(
            "compute_profile_tool must be one of "
            f"{', '.join(sorted(COMPUTE_PROFILE_TOOL_MODES))}, got {compute_profile_tool!r}"
        )

    from acprof.artifact_layout import ArtifactLayout
    layout = ArtifactLayout.discover(output_dir)
    profile_root = str(layout.path("compute_profiles"))
    entries = load_input_scale_plan_entries(input_scale_plan_file)
    payload_file = input_scale_plan_file

    normalized_gpus = {_normal_gpu_mode(gpu) for gpu in gpu_list}
    collect_torch_cpu = (
        "off" in normalized_gpus
        and tool_mode in {"both", "torch"}
    )
    collect_torch_gpu = (
        "on" in normalized_gpus
        and tool_mode in {"both", "torch"}
    )
    collect_advisor_cpu = (
        "off" in normalized_gpus
        and tool_mode == "vendor"
    )
    collect_ncu_gpu = (
        "on" in normalized_gpus
        and tool_mode in {"both", "ncu", "vendor"}
    )
    if (
        collect_torch_cpu
        or collect_torch_gpu
        or collect_advisor_cpu
        or collect_ncu_gpu
    ):
        os.makedirs(profile_root, exist_ok=True)
    advisor_bin = (
        find_executable(advisor_root, ("advisor", "advixe-cl"))
        if collect_advisor_cpu and not declared_profiler_error(task_info, "intel_advisor")
        else None
    )
    ncu_bin = (
        find_executable(ncu_root, ("ncu", "nv-nsight-cu-cli"))
        if collect_ncu_gpu and not declared_profiler_error(task_info, ncu.NCU_TOOL)
        else None
    )
    max_cpu, max_mem = _default_compute_profile_resources(
        compute_profile_cpus,
        compute_profile_mem,
    )

    profiles: Dict[str, Any] = {}
    if "off" in normalized_gpus:
        cpu_tools: Dict[str, Dict[str, Any]] = {}
        if collect_advisor_cpu:
            started_at = time.perf_counter()
            cpu_tools["intel_advisor"] = _safe_profile_tool(
                task_info=task_info,
                tool="intel_advisor",
                entries=entries,
                repeat=advisor_repeat,
                callback=lambda: advisor._profile_cpu_entries(
                    entries=entries,
                    advisor_bin=advisor_bin,
                    advisor_root=advisor_root,
                    task_info=task_info,
                    image_tag=image_tag,
                    cpu=max_cpu,
                    mem=max_mem,
                    payload_file=payload_file,
                    profile_root=profile_root,
                    repeat=advisor_repeat,
                ),
            )
            report_profiler_completion(
                progress_callback,
                profiler="CPU Advisor",
                profiles=[cpu_tools["intel_advisor"]],
                elapsed_seconds=time.perf_counter() - started_at,
            )
        if collect_torch_cpu:
            started_at = time.perf_counter()
            cpu_tools[torch.TORCH_PROFILER_TOOL] = _safe_profile_tool(
                task_info=task_info,
                tool=torch.TORCH_PROFILER_TOOL,
                entries=entries,
                repeat=torch_profiler_repeat,
                callback=lambda: torch._profile_torch_entries(
                    entries=entries,
                    task_info=task_info,
                    image_tag=image_tag,
                    cpu=max_cpu,
                    mem=max_mem,
                    use_gpu=False,
                    profile_key="cpu",
                    payload_file=payload_file,
                    profile_root=profile_root,
                    repeat=torch_profiler_repeat,
                ),
            )
            report_profiler_completion(
                progress_callback,
                profiler="CPU Torch",
                profiles=[cpu_tools[torch.TORCH_PROFILER_TOOL]],
                elapsed_seconds=time.perf_counter() - started_at,
            )
        if cpu_tools:
            profiles["cpu"] = cpu_tools

    if "on" in normalized_gpus:
        gpu_tools: Dict[str, Dict[str, Any]] = {}
        if collect_torch_gpu:
            started_at = time.perf_counter()
            gpu_tools[torch.TORCH_PROFILER_TOOL] = _safe_profile_tool(
                task_info=task_info,
                tool=torch.TORCH_PROFILER_TOOL,
                entries=entries,
                repeat=torch_profiler_repeat,
                callback=lambda: torch._profile_torch_entries(
                    entries=entries,
                    task_info=task_info,
                    image_tag=image_tag,
                    cpu=max_cpu,
                    mem=max_mem,
                    use_gpu=True,
                    profile_key="gpu",
                    payload_file=payload_file,
                    profile_root=profile_root,
                    repeat=torch_profiler_repeat,
                ),
            )
            report_profiler_completion(
                progress_callback,
                profiler="GPU Torch",
                profiles=[gpu_tools[torch.TORCH_PROFILER_TOOL]],
                elapsed_seconds=time.perf_counter() - started_at,
            )
        if collect_ncu_gpu:
            started_at = time.perf_counter()
            gpu_tools[ncu.NCU_TOOL] = _safe_profile_tool(
                task_info=task_info,
                tool=ncu.NCU_TOOL,
                entries=entries,
                repeat=ncu_repeat,
                callback=lambda: ncu._profile_gpu_entries(
                    entries=entries,
                    ncu_bin=ncu_bin,
                    ncu_root=ncu_root,
                    task_info=task_info,
                    image_tag=image_tag,
                    cpu=max_cpu,
                    mem=max_mem,
                    payload_file=payload_file,
                    profile_root=profile_root,
                    repeat=ncu_repeat,
                    resume_existing=resume_existing_ncu_profiles,
                ),
            )
            report_profiler_completion(
                progress_callback,
                profiler="NCU",
                profiles=[gpu_tools[ncu.NCU_TOOL]],
                elapsed_seconds=time.perf_counter() - started_at,
            )
        if gpu_tools:
            profiles["gpu"] = gpu_tools

    enabled_tools = [
        tool
        for tool, enabled in (
            (torch.TORCH_PROFILER_TOOL, collect_torch_cpu or collect_torch_gpu),
            (ncu.NCU_TOOL, collect_ncu_gpu),
            ("intel_advisor", collect_advisor_cpu),
        )
        if enabled
    ]
    ncu_metrics: List[str] = []
    gpu_profile = profiles.get("gpu", {})
    if isinstance(gpu_profile, dict):
        ncu_profile = gpu_profile.get(ncu.NCU_TOOL, {})
        if isinstance(ncu_profile, dict):
            ncu_metrics = list(ncu_profile.get("metrics") or [])

    static_metadata = {
        "compute_profile_tools": enabled_tools,
        "torch_profiler_eager_flop_semantics": "logical_operator_shape_flops",
        "torch_profiler_eager_attention_implementation": "eager",
        "torch_profiler_eager_repeat_cpu": (
            max(1, int(torch_profiler_repeat)) if collect_torch_cpu else None
        ),
        "torch_profiler_eager_repeat_gpu": (
            max(1, int(torch_profiler_repeat)) if collect_torch_gpu else None
        ),
        "ncu_flop_semantics": "gpu_executed_floating_point_operations",
        "ncu_repeat": max(1, int(ncu_repeat)) if collect_ncu_gpu else None,
        "ncu_fma_flop_weight": ncu.NCU_FMA_FLOP_WEIGHT,
        "ncu_metrics": ncu_metrics,
        "torch_version": _first_profile_value(
            profiles,
            "torch_version",
            "unknown",
        ),
        "transformers_version": _first_profile_value(
            profiles,
            "transformers_version",
            "unknown",
        ),
        "ncu_version": (
            _executable_version(ncu_bin) if collect_ncu_gpu else "unknown"
        ),
        "gpu_compute_capability": _first_profile_value(
            profiles,
            "gpu_compute_capability",
            "unknown",
        ),
        "gpu_sm_count": _first_profile_value(
            profiles,
            "gpu_sm_count",
            "unknown",
        ),
        "compute_profiles_retained": bool(keep_profiles and enabled_tools),
        "compute_profile_provenance": (
            "collected" if enabled_tools else "disabled"
        ),
    }

    plan = {
        "model_id": task_info.model_id,
        "task_family": task_info.task_family,
        "pipeline_tag": task_info.pipeline_tag,
        "runtime_backend": task_info.runtime_backend,
        "compute_profile_tool_mode": tool_mode,
        "static_metadata": static_metadata,
        "profiles": profiles,
    }
    plan_path = str(layout.path(COMPUTE_PROFILE_PLAN_NAME))

    if not keep_profiles:
        _strip_discarded_profile_paths(profiles)
        shutil.rmtree(profile_root, ignore_errors=True)

    write_profile_json(plan_path, plan)
    print(f"[compute] Compute profile plan: {plan_path}")
    return plan_path
