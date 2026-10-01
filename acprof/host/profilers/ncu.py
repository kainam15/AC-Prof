"""NCU execution, report export, checkpoint validation and resume."""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from acprof.host.command import run_command
from acprof.host.detect import TaskInfo
from acprof.host.profiler_common import (
    _base_docker_cmd,
    _format_scale_value,
    _parse_last_json_line,
    _runner_args,
    _write_json_atomic,
)
from acprof.host.profilers.compute_parsers import (
    NCU_DURATION_METRIC,
    NCU_SASS_FLOP_WEIGHTS,
    NCU_TENSOR_METRIC_RE,
    _finite_or_none,
    _is_ncu_flop_metric,
    _to_float,
    parse_ncu_profile_csv,
)
from acprof.host.profilers.tool_discovery import _tool_mount_roots

NCU_TOOL = "ncu"


NCU_FMA_FLOP_WEIGHT = 2.0


NCU_CHECKPOINT_SCHEMA_VERSION = 1


NCU_COMPLETE_ENTRY_FIELDS = (
    "gpu_executed_mflop_per_request_ncu",
    "gpu_executed_tensor_mflop_per_request_ncu",
    "gpu_executed_scalar_mflop_per_request_ncu",
    "gpu_executed_tensor_share_pct_ncu",
    "gpu_kernel_launch_count_per_request_ncu",
    "gpu_kernel_time_sum_ms_per_request_ncu",
)


def _ncu_error_entries(
    entries: List[Dict[str, Any]],
    error: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "input_scale": float(entry["input_scale"]),
            "tool": NCU_TOOL,
            "gpu_executed_mflop_per_request_ncu": None,
            "gpu_executed_tensor_mflop_per_request_ncu": None,
            "gpu_executed_scalar_mflop_per_request_ncu": None,
            "gpu_executed_tensor_share_pct_ncu": None,
            "gpu_kernel_launch_count_per_request_ncu": None,
            "gpu_kernel_time_sum_ms_per_request_ncu": None,
            "error": error,
        }
        for entry in entries
    ]


def _parse_ncu_metric_names(query_output: str) -> List[str]:
    names = set()
    for line in query_output.splitlines():
        for token in re.split(r"[\s,]+", line.strip()):
            token = token.strip()
            if not token:
                continue
            if _is_ncu_flop_metric(token):
                names.add(token)
    return sorted(names)


def _select_ncu_flop_metrics(available_metrics: Iterable[str]) -> List[str]:
    available = set(available_metrics)
    sass_metrics = []
    for metric in NCU_SASS_FLOP_WEIGHTS:
        rollup_metric = f"{metric}.sum"
        if rollup_metric in available:
            sass_metrics.append(rollup_metric)
        elif metric in available:
            sass_metrics.append(metric)

    tensor_metrics = sorted(
        metric for metric in available
        if NCU_TENSOR_METRIC_RE.match(metric)
    )
    return [*sass_metrics, *tensor_metrics]


def _resolve_ncu_metrics(
    ncu_bin: str,
    *,
    container_base_cmd: Optional[Sequence[str]] = None,
) -> Tuple[List[str], str]:
    query_errors = []
    command_prefixes: List[Sequence[str]] = [()]
    if container_base_cmd:
        command_prefixes.append(container_base_cmd)

    for command_prefix in command_prefixes:
        for query_args in (
            ["--query-metrics", "--query-metrics-mode", "all"],
            ["--query-metrics"],
        ):
            result = run_command(
                [*command_prefix, ncu_bin, *query_args],
                check=False,
            )
            if result.returncode != 0:
                query_errors.append(result.stderr.strip() or result.stdout.strip())
                continue
            metrics = _select_ncu_flop_metrics(
                _parse_ncu_metric_names(result.stdout)
            )
            if metrics:
                return metrics, ""
            detail = result.stderr.strip() or result.stdout.strip()
            if detail:
                query_errors.append(detail)
    error = "; ".join(error for error in query_errors if error)
    if error:
        return [], f"ncu_no_flop_metrics_found:{error}"
    return [], "ncu_no_flop_metrics_found"


def _ncu_supports_nvtx_filter(ncu_bin: str) -> bool:
    result = run_command([ncu_bin, "--help"], check=False)
    return result.returncode == 0 and "--nvtx-include" in result.stdout


def _ncu_collect_filter_args(ncu_bin: str) -> List[str]:
    if _ncu_supports_nvtx_filter(ncu_bin):
        return ["--nvtx", "--nvtx-include", "acprof_compute/"]
    return ["--nvtx", "--nvtx-include", "acprof_compute"]


def _ncu_section_args(ncu_bin: str) -> List[str]:
    section_dir = os.path.join(os.path.dirname(os.path.realpath(ncu_bin)), "sections")
    if os.path.isdir(section_dir):
        return ["--section-folder", section_dir, "--apply-rules", "no"]
    return []


def _write_text_atomic(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def _ncu_artifact_paths(
    profile_root: str,
    input_scale: float,
) -> Tuple[str, str, str, str]:
    scale_label = _format_scale_value(input_scale)
    report_base = f"/profiles/ncu_scale_{scale_label}"
    host_csv = os.path.join(profile_root, f"ncu_scale_{scale_label}.csv")
    host_report = os.path.join(profile_root, f"ncu_scale_{scale_label}.ncu-rep")
    checkpoint = os.path.join(
        profile_root,
        f"ncu_scale_{scale_label}.checkpoint.json",
    )
    return report_base, host_csv, host_report, checkpoint


def _ncu_report_reference(profile_root: str, host_csv: str) -> str:
    from pathlib import Path

    from acprof.artifact_layout import MANIFEST_NAME, ArtifactLayout
    base = Path(profile_root).absolute().parent
    if base.name == "raw" and (base.parent / MANIFEST_NAME).exists():
        if ArtifactLayout.discover(base.parent).layout_version == 2:
            base = base.parent
    return os.path.relpath(
        host_csv,
        start=base,
    )


def _ncu_entry_complete(entry: Dict[str, Any]) -> bool:
    return not str(entry.get("error") or "").strip() and all(
        math.isfinite(_to_float(entry.get(field)))
        for field in NCU_COMPLETE_ENTRY_FIELDS
    )


def _ncu_entry_from_csv(
    *,
    entry: Dict[str, Any],
    host_csv: str,
    profile_root: str,
    repeat: int,
    runner_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report_path = _ncu_report_reference(profile_root, host_csv)
    try:
        parsed = parse_ncu_profile_csv(host_csv, repeat=repeat)
    except Exception as exc:
        error_entry = _ncu_error_entries(
            [entry],
            f"ncu_parse_failed:{exc!r}",
        )[0]
        error_entry["report"] = report_path
        return error_entry

    payload = runner_payload or {}
    total_flops = _to_float(parsed.get("total_flops_per_request"))
    if not math.isfinite(total_flops):
        error_entry = _ncu_error_entries(
            [entry],
            "ncu_parse_failed:flop_metrics_missing",
        )[0]
        error_entry["report"] = report_path
        return error_entry

    tensor_flops = _finite_or_none(parsed.get("tensor_flops_per_request"))
    scalar_flops = _finite_or_none(parsed.get("scalar_flops_per_request"))
    result = {
        "input_scale": float(entry["input_scale"]),
        "tool": NCU_TOOL,
        "gpu_executed_mflop_per_request_ncu": total_flops / 1_000_000.0,
        "gpu_executed_tensor_mflop_per_request_ncu": (
            tensor_flops / 1_000_000.0 if tensor_flops is not None else None
        ),
        "gpu_executed_scalar_mflop_per_request_ncu": (
            scalar_flops / 1_000_000.0 if scalar_flops is not None else None
        ),
        "gpu_executed_tensor_share_pct_ncu": _finite_or_none(
            parsed.get("tensor_share_pct")
        ),
        "gpu_kernel_launch_count_per_request_ncu": _finite_or_none(
            parsed.get("kernel_launch_count_per_request")
        ),
        "gpu_kernel_time_sum_ms_per_request_ncu": _finite_or_none(
            parsed.get("kernel_time_sum_ms_per_request")
        ),
        "gpu_compute_capability": str(
            payload.get("gpu_compute_capability")
            or parsed.get("gpu_compute_capability")
            or ""
        ),
        "gpu_sm_count": _finite_or_none(
            payload.get("gpu_sm_count", parsed.get("gpu_sm_count"))
        ),
        "error": "",
        "report": report_path,
    }
    missing = [
        field
        for field in NCU_COMPLETE_ENTRY_FIELDS
        if not math.isfinite(_to_float(result.get(field)))
    ]
    if missing:
        error_entry = _ncu_error_entries(
            [entry],
            "ncu_parse_failed:required_metrics_missing:" + ",".join(missing),
        )[0]
        error_entry["report"] = report_path
        return error_entry
    return result


def _export_ncu_report(
    *,
    ncu_bin: str,
    base_cmd: Sequence[str],
    report_base: str,
    host_csv: str,
) -> Tuple[bool, str]:
    import_cmd = [
        ncu_bin,
        "--import", f"{report_base}.ncu-rep",
        "--page", "raw",
        "--csv",
    ]
    result = run_command([*base_cmd, *import_cmd], check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "ncu import failed").strip()
        return False, detail
    if not result.stdout:
        return False, "ncu import produced an empty CSV"
    _write_text_atomic(host_csv, result.stdout)
    return True, ""


def _read_ncu_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _ncu_checkpoint_matches(
    checkpoint: Dict[str, Any],
    *,
    task_info: TaskInfo,
    image_tag: str,
    input_scale: float,
    repeat: int,
    metrics: Sequence[str],
) -> bool:
    try:
        checkpoint_scale = float(checkpoint.get("input_scale"))
        checkpoint_repeat = int(checkpoint.get("repeat"))
        schema_version = int(checkpoint.get("schema_version"))
    except (TypeError, ValueError):
        return False
    return (
        schema_version == NCU_CHECKPOINT_SCHEMA_VERSION
        and str(checkpoint.get("model_id") or "") == task_info.model_id
        and str(checkpoint.get("model_revision") or "main")
        == str(task_info.model_revision or "main")
        and str(checkpoint.get("image_tag") or "") == image_tag
        and math.isclose(checkpoint_scale, input_scale, abs_tol=1e-9)
        and checkpoint_repeat == max(1, int(repeat))
        and list(checkpoint.get("metrics") or []) == list(metrics)
    )


def _write_ncu_checkpoint(
    *,
    checkpoint_path: str,
    task_info: TaskInfo,
    image_tag: str,
    input_scale: float,
    repeat: int,
    metrics: Sequence[str],
    host_csv: str,
    entry: Dict[str, Any],
) -> None:
    _write_json_atomic(
        checkpoint_path,
        {
            "schema_version": NCU_CHECKPOINT_SCHEMA_VERSION,
            "model_id": task_info.model_id,
            "model_revision": task_info.model_revision or "main",
            "image_tag": image_tag,
            "input_scale": float(input_scale),
            "repeat": max(1, int(repeat)),
            "metrics": list(metrics),
            "csv_size_bytes": os.path.getsize(host_csv),
            "entry": entry,
        },
    )


def _resume_ncu_for_entry(
    *,
    ncu_bin: str,
    ncu_metrics: Sequence[str],
    task_info: TaskInfo,
    image_tag: str,
    base_cmd: Sequence[str],
    profile_root: str,
    entry: Dict[str, Any],
    repeat: int,
) -> Optional[Dict[str, Any]]:
    input_scale = float(entry["input_scale"])
    scale_label = _format_scale_value(input_scale)
    report_base, host_csv, host_report, checkpoint_path = _ncu_artifact_paths(
        profile_root,
        input_scale,
    )
    checkpoint_exists = os.path.isfile(checkpoint_path)
    checkpoint = _read_ncu_checkpoint(checkpoint_path) if checkpoint_exists else None
    if checkpoint_exists and (
        checkpoint is None
        or not _ncu_checkpoint_matches(
            checkpoint,
            task_info=task_info,
            image_tag=image_tag,
            input_scale=input_scale,
            repeat=repeat,
            metrics=ncu_metrics,
        )
    ):
        print(
            f"[compute][ncu][resume] scale={scale_label}: "
            "checkpoint does not match this run; recollecting"
        )
        return None

    if checkpoint is not None and os.path.isfile(host_csv):
        checkpoint_entry = checkpoint.get("entry")
        expected_size = _to_float(checkpoint.get("csv_size_bytes"))
        if (
            isinstance(checkpoint_entry, dict)
            and _ncu_entry_complete(checkpoint_entry)
            and math.isfinite(expected_size)
            and os.path.getsize(host_csv) == int(expected_size)
        ):
            resumed = dict(checkpoint_entry)
            resumed["report"] = _ncu_report_reference(profile_root, host_csv)
            print(
                f"[compute][ncu][resume] scale={scale_label}: "
                f"reusing checkpoint {checkpoint_path}"
            )
            return resumed

    if checkpoint is None and os.path.isfile(host_csv):
        resumed = _ncu_entry_from_csv(
            entry=entry,
            host_csv=host_csv,
            profile_root=profile_root,
            repeat=repeat,
        )
        if _ncu_entry_complete(resumed):
            _write_ncu_checkpoint(
                checkpoint_path=checkpoint_path,
                task_info=task_info,
                image_tag=image_tag,
                input_scale=input_scale,
                repeat=repeat,
                metrics=ncu_metrics,
                host_csv=host_csv,
                entry=resumed,
            )
            print(
                f"[compute][ncu][resume] scale={scale_label}: "
                f"reusing valid CSV {host_csv}"
            )
            return resumed
        print(
            f"[compute][ncu][resume] scale={scale_label}: "
            "existing CSV is incomplete"
        )

    if os.path.isfile(host_report):
        print(
            f"[compute][ncu][resume] scale={scale_label}: "
            f"exporting existing report {host_report}"
        )
        exported, detail = _export_ncu_report(
            ncu_bin=ncu_bin,
            base_cmd=base_cmd,
            report_base=report_base,
            host_csv=host_csv,
        )
        if exported:
            resumed = _ncu_entry_from_csv(
                entry=entry,
                host_csv=host_csv,
                profile_root=profile_root,
                repeat=repeat,
            )
            if _ncu_entry_complete(resumed):
                _write_ncu_checkpoint(
                    checkpoint_path=checkpoint_path,
                    task_info=task_info,
                    image_tag=image_tag,
                    input_scale=input_scale,
                    repeat=repeat,
                    metrics=ncu_metrics,
                    host_csv=host_csv,
                    entry=resumed,
                )
                print(
                    f"[compute][ncu][resume] scale={scale_label}: "
                    "recovered from existing .ncu-rep"
                )
                return resumed
            detail = str(resumed.get("error") or "exported CSV is incomplete")
        print(
            f"[compute][ncu][resume] scale={scale_label}: "
            f"report recovery failed ({detail[:300]}); recollecting"
        )
    return None


def _run_ncu_for_entry(
    *,
    ncu_bin: str,
    ncu_metrics: List[str],
    task_info: TaskInfo,
    image_tag: str,
    cpu: int,
    mem: int,
    payload_file: str,
    profile_root: str,
    tool_mount_roots: Sequence[str],
    entry: Dict[str, Any],
    repeat: int,
) -> Dict[str, Any]:
    report_base, host_csv, _host_report, _checkpoint = _ncu_artifact_paths(
        profile_root,
        float(entry["input_scale"]),
    )
    base_cmd = _base_docker_cmd(
        task_info=task_info,
        image_tag=image_tag,
        cpu=cpu,
        mem=mem,
        use_gpu=True,
        payload_file=payload_file,
        profile_root=profile_root,
        tool_mount_roots=tool_mount_roots,
    )
    collect_cmd = [
        ncu_bin,
        "--target-processes", "all",
        *_ncu_collect_filter_args(ncu_bin),
        *_ncu_section_args(ncu_bin),
        "--page", "raw",
        "--csv",
        "--metrics", ",".join(ncu_metrics),
        "-f",
        "-o", report_base,
        *_runner_args(entry, repeat, "gpu"),
    ]
    result = run_command([*base_cmd, *collect_cmd], check=False)
    if result.returncode != 0:
        return _ncu_error_entries(
            [entry],
            f"ncu_failed:{result.stderr.strip() or result.stdout.strip()}",
        )[0]

    exported, detail = _export_ncu_report(
        ncu_bin=ncu_bin,
        base_cmd=base_cmd,
        report_base=report_base,
        host_csv=host_csv,
    )
    if not exported:
        error_entry = _ncu_error_entries(
            [entry],
            f"ncu_import_failed:{detail}",
        )[0]
        error_entry["report"] = _ncu_report_reference(profile_root, host_csv)
        return error_entry
    return _ncu_entry_from_csv(
        entry=entry,
        host_csv=host_csv,
        profile_root=profile_root,
        repeat=repeat,
        runner_payload=_parse_last_json_line(result.stdout),
    )


def _profile_gpu_entries(
    *,
    entries: List[Dict[str, Any]],
    ncu_bin: Optional[str],
    ncu_root: Optional[str],
    task_info: TaskInfo,
    image_tag: str,
    cpu: int,
    mem: int,
    payload_file: str,
    profile_root: str,
    repeat: int,
    resume_existing: bool = False,
) -> Dict[str, Any]:
    if ncu_bin is None:
        return {
            "tool": NCU_TOOL,
            "repeat": max(1, int(repeat)),
            "error": "ncu_not_found",
            "entries": _ncu_error_entries(entries, "ncu_not_found"),
        }
    mount_roots = _tool_mount_roots(ncu_bin, ncu_root)
    metric_query_base_cmd = _base_docker_cmd(
        task_info=task_info,
        image_tag=image_tag,
        cpu=cpu,
        mem=mem,
        use_gpu=True,
        payload_file=payload_file,
        profile_root=profile_root,
        tool_mount_roots=mount_roots,
    )
    ncu_metrics, metric_error = _resolve_ncu_metrics(
        ncu_bin,
        container_base_cmd=metric_query_base_cmd,
    )
    if metric_error:
        return {
            "tool": NCU_TOOL,
            "repeat": max(1, int(repeat)),
            "error": metric_error,
            "entries": _ncu_error_entries(entries, metric_error),
        }
    collection_metrics = list(dict.fromkeys([*ncu_metrics, NCU_DURATION_METRIC]))
    profile_entries = []
    for entry in entries:
        try:
            profile_entry = None
            if resume_existing:
                profile_entry = _resume_ncu_for_entry(
                    ncu_bin=ncu_bin,
                    ncu_metrics=collection_metrics,
                    task_info=task_info,
                    image_tag=image_tag,
                    base_cmd=metric_query_base_cmd,
                    profile_root=profile_root,
                    entry=entry,
                    repeat=repeat,
                )
            if profile_entry is None:
                scale_label = _format_scale_value(float(entry["input_scale"]))
                print(f"[compute][ncu] scale={scale_label}: collecting")
                profile_entry = _run_ncu_for_entry(
                    ncu_bin=ncu_bin,
                    ncu_metrics=collection_metrics,
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
                if _ncu_entry_complete(profile_entry):
                    _report_base, host_csv, _host_report, checkpoint_path = (
                        _ncu_artifact_paths(
                            profile_root,
                            float(entry["input_scale"]),
                        )
                    )
                    _write_ncu_checkpoint(
                        checkpoint_path=checkpoint_path,
                        task_info=task_info,
                        image_tag=image_tag,
                        input_scale=float(entry["input_scale"]),
                        repeat=repeat,
                        metrics=collection_metrics,
                        host_csv=host_csv,
                        entry=profile_entry,
                    )
        except Exception as exc:
            profile_entry = _ncu_error_entries(
                [entry],
                f"ncu_failed:{exc!r}",
            )[0]
        profile_entries.append(profile_entry)
    errors = [entry["error"] for entry in profile_entries if entry.get("error")]
    return {
        "tool": NCU_TOOL,
        "repeat": max(1, int(repeat)),
        "flop_semantics": "gpu_executed_floating_point_operations",
        "fma_flop_weight": NCU_FMA_FLOP_WEIGHT,
        "metrics": collection_metrics,
        "error": "; ".join(errors),
        "entries": profile_entries,
    }
