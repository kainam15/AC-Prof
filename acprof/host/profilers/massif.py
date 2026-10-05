"""Massif isolated execution, report recovery and cleanup."""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

from acprof.host.command import run_command
from acprof.host.detect import TaskInfo
from acprof.host.profiler_support import (
    execution_thread_environment,
    format_scale_value,
    profile_runner_args,
    profiler_container_command,
    relative_artifact,
    safe_filename_token,
    write_profile_json,
)
from acprof.host.profilers.execution_environment import (
    command_detail,
)
from acprof.host.profilers.execution_parsers import (
    finite_float,
    parse_massif_output,
)

MASSIF_TOOL = "massif"


MASSIF_FIELDS = (
    "cpu_heap_peak_bytes_massif",
    "cpu_heap_extra_peak_bytes_massif",
    "cpu_stack_peak_bytes_massif",
    "cpu_heap_peak_total_bytes_massif",
    "cpu_heap_peak_at_ms_massif",
)


MASSIF_CHECKPOINT_SCHEMA_VERSION = 1


def error_entry(
    entry: Mapping[str, Any],
    error: str,
    *,
    report: Optional[str] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "input_scale": float(entry["input_scale"]),
        "tool": MASSIF_TOOL,
        **{field: None for field in MASSIF_FIELDS},
        "compute_profile_error_massif": error,
        "error": error,
    }
    if report is not None:
        result["report"] = report
    return result


def _massif_artifact_paths(
    *,
    profile_root: str,
    cpu: int,
    mem: int,
    input_scale: float,
) -> Tuple[str, str]:
    scale_label = safe_filename_token(
        format_scale_value(input_scale)
    )
    filename = f"massif_cpu_{cpu}_mem_{mem}_scale_{scale_label}.out"
    host_report = os.path.join(profile_root, filename)
    checkpoint = os.path.join(
        profile_root,
        f"massif_cpu_{cpu}_mem_{mem}_scale_{scale_label}.checkpoint.json",
    )
    return host_report, checkpoint


def _massif_entry_complete(entry: Mapping[str, Any]) -> bool:
    return not str(entry.get("error") or "").strip() and all(
        finite_float(entry.get(field)) is not None
        for field in MASSIF_FIELDS
    )


def _massif_entry_from_report(
    *,
    entry: Mapping[str, Any],
    host_report: str,
    output_dir: str,
) -> Dict[str, Any]:
    relative_report = relative_artifact(host_report, output_dir)
    try:
        parsed = parse_massif_output(host_report)
    except Exception as exc:
        detail = str(exc)
        error = (
            detail
            if detail.startswith("massif_parse_failed:")
            else f"massif_parse_failed:{detail}"
        )
        return error_entry(
            entry,
            error,
            report=relative_report if os.path.isfile(host_report) else None,
        )
    return {
        "input_scale": float(entry["input_scale"]),
        "tool": MASSIF_TOOL,
        **parsed,
        "compute_profile_error_massif": "",
        "error": "",
        "report": relative_report,
    }


def _write_massif_checkpoint(
    *,
    checkpoint_path: str,
    task_info: TaskInfo,
    derived_image: str,
    cpu: int,
    mem: int,
    input_scale: float,
    repeat: int,
    host_report: str,
    entry: Mapping[str, Any],
) -> None:
    write_profile_json(
        checkpoint_path,
        {
            "schema_version": MASSIF_CHECKPOINT_SCHEMA_VERSION,
            "model_id": task_info.model_id,
            "model_revision": task_info.model_revision or "main",
            "derived_image": derived_image,
            "cpu_cores": int(cpu),
            "mem_cap_gb": int(mem),
            "input_scale": float(input_scale),
            "repeat": max(1, int(repeat)),
            "report_size_bytes": os.path.getsize(host_report),
            "entry": dict(entry),
        },
    )


def _read_massif_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as checkpoint_file:
            payload = json.load(checkpoint_file)
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _massif_checkpoint_matches(
    checkpoint: Mapping[str, Any],
    *,
    task_info: TaskInfo,
    derived_image: str,
    cpu: int,
    mem: int,
    input_scale: float,
    repeat: int,
) -> bool:
    try:
        schema_version = int(checkpoint.get("schema_version"))
        checkpoint_cpu = int(checkpoint.get("cpu_cores"))
        checkpoint_mem = int(checkpoint.get("mem_cap_gb"))
        checkpoint_scale = float(checkpoint.get("input_scale"))
        checkpoint_repeat = int(checkpoint.get("repeat"))
    except (TypeError, ValueError):
        return False
    return (
        schema_version == MASSIF_CHECKPOINT_SCHEMA_VERSION
        and str(checkpoint.get("model_id") or "") == task_info.model_id
        and str(checkpoint.get("model_revision") or "main")
        == str(task_info.model_revision or "main")
        and str(checkpoint.get("derived_image") or "") == derived_image
        and checkpoint_cpu == int(cpu)
        and checkpoint_mem == int(mem)
        and math.isclose(checkpoint_scale, input_scale, abs_tol=1e-9)
        and checkpoint_repeat == max(1, int(repeat))
    )


def _resume_massif_entry(
    *,
    task_info: TaskInfo,
    derived_image: str,
    cpu: int,
    mem: int,
    profile_root: str,
    output_dir: str,
    entry: Mapping[str, Any],
    repeat: int,
) -> Optional[Dict[str, Any]]:
    input_scale = float(entry["input_scale"])
    scale_label = format_scale_value(input_scale)
    host_report, checkpoint_path = _massif_artifact_paths(
        profile_root=profile_root,
        cpu=cpu,
        mem=mem,
        input_scale=input_scale,
    )
    checkpoint_exists = os.path.isfile(checkpoint_path)
    checkpoint = (
        _read_massif_checkpoint(checkpoint_path)
        if checkpoint_exists
        else None
    )
    if checkpoint_exists and (
        checkpoint is None
        or not _massif_checkpoint_matches(
            checkpoint,
            task_info=task_info,
            derived_image=derived_image,
            cpu=cpu,
            mem=mem,
            input_scale=input_scale,
            repeat=repeat,
        )
    ):
        print(
            f"[execution-profile][massif][resume] scale={scale_label}: "
            "checkpoint does not match this run; recollecting"
        )
        return None

    if checkpoint is not None and os.path.isfile(host_report):
        checkpoint_entry = checkpoint.get("entry")
        expected_size = finite_float(checkpoint.get("report_size_bytes"))
        if (
            isinstance(checkpoint_entry, Mapping)
            and _massif_entry_complete(checkpoint_entry)
            and expected_size is not None
            and os.path.getsize(host_report) == int(expected_size)
        ):
            resumed = dict(checkpoint_entry)
            resumed["report"] = relative_artifact(host_report, output_dir)
            print(
                f"[execution-profile][massif][resume] scale={scale_label}: "
                f"reusing checkpoint {checkpoint_path}"
            )
            return resumed

    if checkpoint is None and os.path.isfile(host_report):
        resumed = _massif_entry_from_report(
            entry=entry,
            host_report=host_report,
            output_dir=output_dir,
        )
        if _massif_entry_complete(resumed):
            _write_massif_checkpoint(
                checkpoint_path=checkpoint_path,
                task_info=task_info,
                derived_image=derived_image,
                cpu=cpu,
                mem=mem,
                input_scale=input_scale,
                repeat=repeat,
                host_report=host_report,
                entry=resumed,
            )
            print(
                f"[execution-profile][massif][resume] scale={scale_label}: "
                f"reusing valid report {host_report}"
            )
            return resumed
        print(
            f"[execution-profile][massif][resume] scale={scale_label}: "
            "existing report is incomplete; recollecting"
        )
    return None


def _collect_massif_entry(
    *,
    task_info: TaskInfo,
    derived_image: str,
    cpu: int,
    mem: int,
    payload_file: str,
    profile_root: str,
    output_dir: str,
    entry: Mapping[str, Any],
    repeat: int,
) -> Dict[str, Any]:
    input_scale = float(entry["input_scale"])
    host_report, checkpoint_path = _massif_artifact_paths(
        profile_root=profile_root,
        cpu=cpu,
        mem=mem,
        input_scale=input_scale,
    )
    filename = os.path.basename(host_report)
    relative_report = relative_artifact(host_report, output_dir)
    with profiler_container_command(
        task_info=task_info,
        image_tag=derived_image,
        cpu=cpu,
        mem=mem,
        use_gpu=False,
        payload_file=payload_file,
        profile_root=profile_root,
        tool_mount_roots=(),
    ) as base_cmd:
        base_cmd = execution_thread_environment(base_cmd)
        command = [
            *base_cmd,
            "valgrind",
            "--tool=massif",
            "--time-unit=ms",
            "--stacks=yes",
            f"--massif-out-file=/profiles/{filename}",
            *profile_runner_args(dict(entry), repeat, "cpu"),
        ]
        result = run_command(command, check=False)
    if result.returncode != 0:
        return error_entry(
            entry,
            f"massif_failed:{command_detail(result)}",
            report=relative_report if os.path.isfile(host_report) else None,
        )
    profiled = _massif_entry_from_report(
        entry=entry,
        host_report=host_report,
        output_dir=output_dir,
    )
    if _massif_entry_complete(profiled):
        _write_massif_checkpoint(
            checkpoint_path=checkpoint_path,
            task_info=task_info,
            derived_image=derived_image,
            cpu=cpu,
            mem=mem,
            input_scale=input_scale,
            repeat=repeat,
            host_report=host_report,
            entry=profiled,
        )
    return profiled


def profile(
    *,
    entries: List[Dict[str, Any]],
    global_error: str,
    task_info: TaskInfo,
    derived_image: Optional[str],
    cpu: int,
    mem: int,
    payload_file: str,
    profile_root: str,
    output_dir: str,
    repeat: int,
    resume_existing: bool = False,
) -> Dict[str, Any]:
    if global_error or not derived_image:
        error = global_error or "massif_not_found"
        profiled_entries = [
            error_entry(entry, error)
            for entry in entries
        ]
    else:
        profiled_entries = []
        for entry in entries:
            try:
                profiled = None
                if resume_existing:
                    profiled = _resume_massif_entry(
                        task_info=task_info,
                        derived_image=derived_image,
                        cpu=cpu,
                        mem=mem,
                        profile_root=profile_root,
                        output_dir=output_dir,
                        entry=entry,
                        repeat=repeat,
                    )
                if profiled is None:
                    profiled = _collect_massif_entry(
                        task_info=task_info,
                        derived_image=derived_image,
                        cpu=cpu,
                        mem=mem,
                        payload_file=payload_file,
                        profile_root=profile_root,
                        output_dir=output_dir,
                        entry=entry,
                        repeat=repeat,
                    )
                profiled_entries.append(profiled)
            except Exception as exc:
                profiled_entries.append(
                    error_entry(
                        entry,
                        f"massif_failed:{exc!r}",
                    )
                )
    return {
        "tool": MASSIF_TOOL,
        "repeat": repeat,
        "error": global_error,
        "entries": profiled_entries,
    }
