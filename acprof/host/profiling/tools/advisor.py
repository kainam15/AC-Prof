"""Intel Advisor execution and artifact collection in isolated containers."""
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from acprof.host.detect import TaskInfo
from acprof.host.profiling.support import (
    format_scale_value,
    profile_runner_args,
    profiler_container_command,
)
from acprof.host.profiling.tools.compute_parsers import (
    parse_advisor_self_gflop_csv,
)
from acprof.host.profiling.tools.tool_discovery import tool_mount_roots
from acprof.system.command import run_command


def tool_error_entries(
    entries: list[dict[str, Any]],
    error: str,
    tool: str,
) -> list[dict[str, Any]]:
    return [
        {
            "input_scale": float(entry["input_scale"]),
            "tool": tool,
            "model_mflop_per_request": None,
            "error": error,
        }
        for entry in entries
    ]


def _run_advisor_for_entry(
    *,
    advisor_bin: str,
    task_info: TaskInfo,
    image_tag: str,
    cpu: int,
    mem: int,
    payload_file: str,
    profile_root: str,
    tool_mount_roots: Sequence[str],
    entry: dict[str, Any],
    repeat: int,
) -> dict[str, Any]:
    scale_label = format_scale_value(float(entry["input_scale"]))
    project_dir = f"/profiles/advisor_scale_{scale_label}"
    report_path = f"/profiles/advisor_scale_{scale_label}.csv"
    host_report_path = os.path.join(profile_root, f"advisor_scale_{scale_label}.csv")
    runner_args = profile_runner_args(entry, repeat, "cpu")
    commands = [
        [
            advisor_bin,
            "--collect=survey",
            "--profile-python=off",
            "--start-paused",
            "--project-dir", project_dir,
            "--",
            *runner_args,
        ],
        [
            advisor_bin,
            "--collect=tripcounts",
            "--flop",
            "--profile-jit",
            "--start-paused",
            "--project-dir", project_dir,
            "--",
            *runner_args,
        ],
        [
            advisor_bin,
            "--report=survey",
            "--format=csv",
            "--show-all-columns",
            "--project-dir", project_dir,
            "--report-output", report_path,
        ],
    ]
    for command in commands:
        with profiler_container_command(
            task_info=task_info,
            image_tag=image_tag,
            cpu=cpu,
            mem=mem,
            use_gpu=False,
            payload_file=payload_file,
            profile_root=profile_root,
            tool_mount_roots=tool_mount_roots,
        ) as base_cmd:
            result = run_command([*base_cmd, *command], check=False)
        if result.returncode != 0:
            return {
                "input_scale": float(entry["input_scale"]),
                "model_mflop_per_request": None,
                "error": f"advisor_failed:{result.stderr.strip() or result.stdout.strip()}",
            }
    gflop = parse_advisor_self_gflop_csv(host_report_path)
    if gflop != gflop:
        return {
            "input_scale": float(entry["input_scale"]),
            "model_mflop_per_request": None,
            "error": "advisor_parse_failed:self_gflop_missing",
        }
    return {
        "input_scale": float(entry["input_scale"]),
        "tool": "intel_advisor",
        "model_mflop_per_request": (gflop * 1000.0) / float(max(1, int(repeat))),
        "error": "",
        "report": host_report_path,
    }


def profile_cpu_entries(
    *,
    entries: list[dict[str, Any]],
    advisor_bin: str | None,
    advisor_root: str | None,
    task_info: TaskInfo,
    image_tag: str,
    cpu: int,
    mem: int,
    payload_file: str,
    profile_root: str,
    repeat: int,
) -> dict[str, Any]:
    if advisor_bin is None:
        return {
            "tool": "intel_advisor",
            "repeat": max(1, int(repeat)),
            "error": "advisor_not_found",
            "entries": tool_error_entries(
                entries,
                "advisor_not_found",
                "intel_advisor",
            ),
        }
    mount_roots = tool_mount_roots(advisor_bin, advisor_root)
    profile_entries = [
        _run_advisor_for_entry(
            advisor_bin=advisor_bin,
            task_info=task_info,
            image_tag=image_tag,
            cpu=cpu,
            mem=mem,
            payload_file=payload_file,
            profile_root=profile_root,
            tool_mount_roots=mount_roots,
            entry=entry,
            repeat=repeat,
        )
        for entry in entries
    ]
    errors = [entry["error"] for entry in profile_entries if entry.get("error")]
    return {
        "tool": "intel_advisor",
        "repeat": max(1, int(repeat)),
        "error": "; ".join(errors),
        "entries": profile_entries,
    }
