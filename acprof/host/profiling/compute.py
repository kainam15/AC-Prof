"""Host-side FLOP profiling plan generation for AC-Prof."""
from __future__ import annotations

import copy
import logging
import math
import os
import time
from collections.abc import Iterable
from typing import Any

from acprof.capabilities import declared_profiler_error
from acprof.config import DEFAULT_COMPUTE_PROFILE_TOOL
from acprof.gpu_mode import normalize_gpu_mode
from acprof.gpu_probe import gpu_compute_value, is_gpu_probe_report, record_gpu_probe_error
from acprof.host.detect import TaskInfo
from acprof.host.gpu_device import selected_gpu_device
from acprof.host.profiling.cleanup import discard_profile_artifacts
from acprof.host.profiling.progress import (
    ProfilerProgressCallback,
    report_profiler_completion,
)
from acprof.host.profiling.support import load_input_scale_plan_entries, write_profile_json
from acprof.host.profiling.tools import advisor, ncu, torch
from acprof.host.profiling.tools.tool_discovery import find_executable
from acprof.system.command import run_command

_LOG = logging.getLogger(__name__)


COMPUTE_PROFILE_PLAN_NAME = "compute_profile_plan.json"
COMPUTE_PROFILE_TOOL_MODES = {"none", "both", "ncu", "torch", "vendor"}


def _host_logical_cpus() -> int:
    return max(1, int(os.cpu_count() or 1))


def _host_memory_gb_fraction(fraction: float = 0.75) -> int:
    total_bytes = 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
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
    compute_profile_cpus: int | None,
    compute_profile_mem: int | None,
) -> tuple[int, int]:
    cpu = max(1, int(compute_profile_cpus or _host_logical_cpus()))
    mem = max(1, int(compute_profile_mem or _host_memory_gb_fraction(0.75)))
    return cpu, mem


def _failed_tool_profile(
    *,
    tool: str,
    entries: list[dict[str, Any]],
    repeat: int,
    error: str,
) -> dict[str, Any]:
    error_entries = (
        ncu.ncu_error_entries(entries, error)
        if tool == ncu.NCU_TOOL
        else torch.torch_error_entries(entries, error)
        if tool == torch.TORCH_PROFILER_TOOL
        else advisor.tool_error_entries(entries, error, tool)
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
    entries: list[dict[str, Any]],
    repeat: int,
    callback: Any,
    task_info: TaskInfo | None = None,
) -> dict[str, Any]:
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


def _executable_version(executable: str | None) -> str:
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


def _profile_tool_maps(profiles: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for device_profile in profiles.values():
        if not isinstance(device_profile, dict):
            continue
        for tool in (torch.TORCH_PROFILER_TOOL, ncu.NCU_TOOL, "intel_advisor"):
            profile = device_profile.get(tool)
            if isinstance(profile, dict):
                yield profile


def _profile_records(profile: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield profile
    for entry in profile.get("entries", []):
        if isinstance(entry, dict):
            yield entry


def _first_profile_value(
    profiles: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    if key in ("gpu_compute_capability", "gpu_sm_count"):
        profiles = {"gpu": profiles.get("gpu", {})}
    for profile in _profile_tool_maps(profiles):
        for record in _profile_records(profile):
            probe = record.get("gpu_compute_metadata")
            if isinstance(probe, dict):
                if probe.get("device", {}).get("identity_status") == "mismatch":
                    continue
                if key in probe.get("fields", {}) and probe["fields"][key].get("status") != "available":
                    continue
            value = record.get(key)
            if value in (None, "", "unknown", "unavailable", "not_applicable", "not_requested"):
                continue
            if key in ("gpu_compute_capability", "gpu_sm_count"):
                try:
                    return gpu_compute_value(key, value)
                except ValueError:
                    continue
            if not isinstance(value, float) or math.isfinite(value):
                return value
    return default


def _bind_gpu_profile_identity(profiles: dict[str, Any], expected_uuid: str) -> None:
    """Keep raw evidence but exclude mismatched CUDA devices from static facts."""
    for profile in _profile_tool_maps({"gpu": profiles.get("gpu", {})}):
        for record in _profile_records(profile):
            report = record.get("gpu_compute_metadata")
            if report is None:
                continue
            if not is_gpu_probe_report(report):
                raise ValueError("invalid GPU compute metadata report")
            device = report["device"]
            device["expected_uuid"] = expected_uuid or None
            observed = device.get("cuda_uuid") or device.get("uuid")
            if not expected_uuid or not isinstance(observed, str) or not observed.strip():
                device["identity_status"] = "unverified"
                continue
            if observed.lower().removeprefix("gpu-") == expected_uuid.lower().removeprefix("gpu-"):
                device["identity_status"] = "verified"
                continue
            device["identity_status"] = "mismatch"
            error = ValueError(f"GPU UUID mismatch: expected {expected_uuid}, received {observed}")
            record_gpu_probe_error(report, "gpu_identity", "resolve", error)
            record["error"] = ";".join(filter(None, (record.get("error"), f"gpu_identity_mismatch:{error}")))


def _gpu_profile_reports(profiles: dict[str, Any]) -> list[dict[str, Any]]:
    reports = []
    for device, tool_maps in profiles.items():
        if not isinstance(tool_maps, dict):
            continue
        for tool, profile in tool_maps.items():
            if not isinstance(profile, dict):
                continue
            for record in _profile_records(profile):
                probe = record.get("gpu_compute_metadata")
                if isinstance(probe, dict):
                    reports.append({"profile": device, "tool": tool,
                                    "input_scale": record.get("input_scale"), "probe": copy.deepcopy(probe)})
    return reports


def _strip_discarded_profile_paths(profiles: dict[str, Any]) -> None:
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
    cpu_list: list[int],
    mem_list: list[int],
    gpu_list: list[str],
    output_dir: str,
    input_scale_plan_file: str,
    advisor_root: str | None,
    ncu_root: str | None,
    advisor_repeat: int,
    ncu_repeat: int,
    keep_profiles: bool,
    compute_profile_cpus: int | None = None,
    compute_profile_mem: int | None = None,
    compute_profile_tool: str = DEFAULT_COMPUTE_PROFILE_TOOL,
    torch_profiler_repeat: int = 1,
    resume_existing_ncu_profiles: bool = False,
    progress_callback: ProfilerProgressCallback | None = None,
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

    normalized_gpus = {normalize_gpu_mode(gpu) for gpu in gpu_list}
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

    profiles: dict[str, Any] = {}
    if "off" in normalized_gpus:
        cpu_tools: dict[str, dict[str, Any]] = {}
        if collect_advisor_cpu:
            started_at = time.perf_counter()
            cpu_tools["intel_advisor"] = _safe_profile_tool(
                task_info=task_info,
                tool="intel_advisor",
                entries=entries,
                repeat=advisor_repeat,
                callback=lambda: advisor.profile_cpu_entries(
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
                callback=lambda: torch.profile_torch_entries(
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
        gpu_tools: dict[str, dict[str, Any]] = {}
        if collect_torch_gpu:
            started_at = time.perf_counter()
            gpu_tools[torch.TORCH_PROFILER_TOOL] = _safe_profile_tool(
                task_info=task_info,
                tool=torch.TORCH_PROFILER_TOOL,
                entries=entries,
                repeat=torch_profiler_repeat,
                callback=lambda: torch.profile_torch_entries(
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
                callback=lambda: ncu.profile_gpu_entries(
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
    ncu_metrics: list[str] = []
    gpu_profile = profiles.get("gpu", {})
    if isinstance(gpu_profile, dict):
        ncu_profile = gpu_profile.get(ncu.NCU_TOOL, {})
        if isinstance(ncu_profile, dict):
            ncu_metrics = list(ncu_profile.get("metrics") or [])

    _bind_gpu_profile_identity(profiles, str(selected_gpu_device().get("uuid") or ""))

    static_metadata = {
        "gpu_compute_metadata": _gpu_profile_reports(profiles),
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
        discard_profile_artifacts(profile_root, plan_path)
        _strip_discarded_profile_paths(profiles)

    write_profile_json(plan_path, plan)
    print(f"[compute] Compute profile plan: {plan_path}")
    return plan_path
