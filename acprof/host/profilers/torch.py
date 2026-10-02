"""Torch profiler execution in isolated containers."""
from __future__ import annotations

from typing import Any, Dict, List

from acprof.host.command import run_command
from acprof.host.detect import TaskInfo
from acprof.host.profiler_support import (
    parse_last_json_line,
    profile_runner_args,
    profiler_container_command,
)
from acprof.host.profilers.compute_parsers import (
    _to_float,
)

TORCH_PROFILER_TOOL = "torch_profiler_eager"


def _torch_error_entries(
    entries: List[Dict[str, Any]],
    error: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "input_scale": float(entry["input_scale"]),
            "tool": TORCH_PROFILER_TOOL,
            "model_logical_mflop_per_request_torch_profiler_eager": None,
            "error": error,
        }
        for entry in entries
    ]


def _run_torch_profiler_for_entry(
    *,
    task_info: TaskInfo,
    image_tag: str,
    cpu: int,
    mem: int,
    use_gpu: bool,
    payload_file: str,
    profile_root: str,
    entry: Dict[str, Any],
    repeat: int,
) -> Dict[str, Any]:
    base_cmd = profiler_container_command(
        task_info=task_info,
        image_tag=image_tag,
        cpu=cpu,
        mem=mem,
        use_gpu=use_gpu,
        payload_file=payload_file,
        profile_root=profile_root,
        tool_mount_roots=(),
    )
    runner_mode = "torch_eager_gpu" if use_gpu else "torch_eager_cpu"
    result = run_command([*base_cmd, *profile_runner_args(entry, repeat, runner_mode)], check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return _torch_error_entries(
            [entry],
            f"torch_profiler_eager_failed:{detail}",
        )[0]

    payload = parse_last_json_line(result.stdout)
    attention_implementation = str(
        payload.get("attention_implementation") or ""
    )
    attention_verified = payload.get("attention_implementation_verified") is True
    if attention_implementation != "eager" or not attention_verified:
        return _torch_error_entries(
            [entry],
            "torch_profiler_eager_parse_failed:"
            "attention_implementation_not_verified",
        )[0]
    mflop = _to_float(
        payload.get("model_logical_mflop_per_request_torch_profiler_eager")
    )
    total_flops = _to_float(payload.get("total_flops"))
    if mflop != mflop and total_flops == total_flops:
        mflop = (total_flops / 1_000_000.0) / float(max(1, int(repeat)))
    if mflop != mflop or mflop <= 0:
        return _torch_error_entries(
            [entry],
            "torch_profiler_eager_parse_failed:model_logical_mflop_missing",
        )[0]

    return {
        "input_scale": float(entry["input_scale"]),
        "tool": TORCH_PROFILER_TOOL,
        "model_logical_mflop_per_request_torch_profiler_eager": mflop,
        "error": "",
        "total_flops": total_flops if total_flops == total_flops else None,
        "attention_implementation": attention_implementation,
        "attention_implementation_verified": attention_verified,
        "torch_version": str(payload.get("torch_version") or "unknown"),
        "transformers_version": str(
            payload.get("transformers_version") or "unknown"
        ),
    }


def _profile_torch_entries(
    *,
    entries: List[Dict[str, Any]],
    task_info: TaskInfo,
    image_tag: str,
    cpu: int,
    mem: int,
    use_gpu: bool,
    profile_key: str,
    payload_file: str,
    profile_root: str,
    repeat: int,
) -> Dict[str, Any]:
    profile_entries = []
    for entry in entries:
        try:
            profile_entry = _run_torch_profiler_for_entry(
                task_info=task_info,
                image_tag=image_tag,
                cpu=cpu,
                mem=mem,
                use_gpu=use_gpu,
                payload_file=payload_file,
                profile_root=profile_root,
                entry=entry,
                repeat=repeat,
            )
        except Exception as exc:
            profile_entry = _torch_error_entries(
                [entry],
                f"torch_profiler_eager_failed:{exc!r}",
            )[0]
        profile_entries.append(profile_entry)
    errors = [entry["error"] for entry in profile_entries if entry.get("error")]
    successful_entry = next(
        (entry for entry in profile_entries if not entry.get("error")),
        {},
    )
    return {
        "tool": TORCH_PROFILER_TOOL,
        "repeat": max(1, int(repeat)),
        "profile": profile_key,
        "flop_semantics": "logical_operator_shape_flops",
        "attention_implementation": "eager",
        "torch_version": successful_entry.get("torch_version", "unknown"),
        "transformers_version": successful_entry.get(
            "transformers_version",
            "unknown",
        ),
        "error": "; ".join(errors),
        "entries": profile_entries,
    }
