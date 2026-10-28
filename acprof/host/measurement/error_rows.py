"""Reconcile failed case rows before publishing CSV and sniff sidecars."""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, TextIO

from acprof.artifact_layout import ArtifactLayout, case_sidecar
from acprof.artifacts import atomic_write, read_json_object
from acprof.config import CSV_FIELDS, DEFAULT_REQUEST_TIMEOUT_SECONDS
from acprof.gpu_mode import normalize_gpu_mode
from acprof.host.detect import TaskInfo
from acprof.host.input_plan import (
    format_scale_value,
    resolve_input_scales,
)
from acprof.host.measurement.energy_validation import row_has_error_status
from acprof.host.measurement.publication import reconcile_sniff_group_sidecar
from acprof.host.profiling.compute_plan import NCU_ERROR_FIELD, TORCH_ERROR_FIELD
from acprof.pixel_metrics import PIXEL_COUNT_FIELDS, PIXEL_RATE_SOURCES
from acprof.result_csv import allow_large_csv_fields


@dataclass(frozen=True)
class CaseErrorRows:
    rows: list[dict[str, Any]]
    missing_rows: list[dict[str, Any]]
    preserved_success_rows: int


def _measurement_key(row: dict[str, Any]) -> tuple[str, str, int]:
    try:
        scale = format_scale_value(float(row.get("input_scale", "nan")))
        repeat_idx = int(row.get("repeat_idx", ""))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid measurement key in partial case CSV: {row!r}") from exc
    warmup_flag = "1" if str(row.get("warmup") or "0").strip() == "1" else "0"
    return scale, warmup_flag, repeat_idx


def _planned_error_rows(
    *,
    scales: list[float],
    cpu: int | None,
    mem: int | None,
    gpu: str,
    warmup: int,
    repeat: int,
    repeat_in_window: int,
    error: str,
    result_origin: str,
    execution_kind: str,
    runtime_id: str,
    environment_class: str,
) -> list[dict[str, Any]]:
    def make_row(scale: float, repeat_idx: int, is_warmup: bool) -> dict[str, Any]:
        row = {field: "nan" for field in CSV_FIELDS}
        row.update({
            "environment_class": environment_class,
            "runtime_kind": execution_kind,
            "runtime_id": runtime_id,
            "resource_scope": (
                "observed_process_tree" if execution_kind == "process" else "container_cgroup"
            ),
            "cpu_cores": "nan" if cpu is None else str(cpu),
            "mem_cap_gb": "nan" if mem is None else str(mem),
            "gpu_mode": gpu,
            "input_scale": format_scale_value(scale),
            "task_param": "",
            "repeat_idx": str(repeat_idx),
            "warmup": "1" if is_warmup else "0",
            "repeat_in_window": str(repeat_in_window),
            TORCH_ERROR_FIELD: "not_run",
            "status": "error",
            "error": error,
            "result_origin": result_origin,
        })
        if gpu == "on":
            row[NCU_ERROR_FIELD] = "not_run"
            row["compute_profile_error_nsys"] = "not_run"
        else:
            row["compute_profile_error_massif"] = "not_run"
        return row

    planned_rows: list[dict[str, Any]] = []
    for scale in scales:
        for idx in range(warmup):
            planned_rows.append(make_row(scale, idx, True))
        for idx in range(repeat):
            planned_rows.append(make_row(scale, idx, False))
    return planned_rows


def reconcile_case_error_rows(
    planned_rows: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
    *,
    error: str,
    preserve_existing: bool = False,
    annotate_existing_error_rows: bool = False,
    timeout_context: dict[str, Any] | None = None,
) -> CaseErrorRows:
    """Merge planned and committed rows without performing filesystem I/O."""
    planned_by_key = {_measurement_key(row): row for row in planned_rows}
    normalized_existing: list[dict[str, Any]] = []
    existing_keys = set()
    if preserve_existing:
        for row in existing_rows:
            key = _measurement_key(row)
            if key not in planned_by_key:
                raise RuntimeError(f"partial case CSV contains an unplanned measurement: {key!r}")
            if key in existing_keys:
                raise RuntimeError(f"partial case CSV contains a duplicate measurement: {key!r}")
            existing_keys.add(key)
            normalized_row = {field: row.get(field, "nan") for field in CSV_FIELDS}
            if annotate_existing_error_rows and row_has_error_status(normalized_row):
                existing_error = str(normalized_row.get("error") or "").strip()
                if error not in existing_error:
                    normalized_row["error"] = (
                        f"{existing_error}; {error}" if existing_error else error
                    )
            normalized_existing.append(normalized_row)

    missing_rows = [
        row for row in planned_rows if _measurement_key(row) not in existing_keys
    ]
    if timeout_context is not None and missing_rows:
        _annotate_timeout_placeholder_rows(missing_rows, timeout_context=timeout_context)
    return CaseErrorRows(
        rows=[*normalized_existing, *missing_rows],
        missing_rows=missing_rows,
        preserved_success_rows=sum(
            not row_has_error_status(row) for row in normalized_existing
        ),
    )


def timeout_context_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return parsed if math.isfinite(parsed) and parsed > 0.0 else float(fallback)


def load_client_error_context(path: str) -> dict[str, Any]:
    try:
        return read_json_object(path, label="client error context")
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        print(
            f"[case][WARN] could not read structured client error context {path}: {exc}",
            file=sys.stderr,
        )
        return {}


def _annotate_timeout_placeholder_rows(
    rows: list[dict[str, Any]],
    *,
    timeout_context: dict[str, Any],
) -> None:
    first_missing_scale = _measurement_key(rows[0])[0]
    try:
        triggering_scale = format_scale_value(
            float(timeout_context.get("input_scale", first_missing_scale))
        )
    except (TypeError, ValueError):
        triggering_scale = first_missing_scale

    timeout_s = timeout_context_float(
        timeout_context.get("request_timeout_s"), DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    request_id = str(timeout_context.get("request_id") or "unknown")
    request_phase = str(timeout_context.get("request_phase") or "unknown")
    repeat_idx = timeout_context.get("measurement_repeat_idx")
    attempted_key = None
    if request_phase == "measurement_warmup" and repeat_idx is not None:
        attempted_key = (triggering_scale, "1", int(repeat_idx))
    elif request_phase == "measurement_repeat" and repeat_idx is not None:
        attempted_key = (triggering_scale, "0", int(repeat_idx))

    context_source = (
        "client_timeout_sidecar"
        if timeout_context.get("input_scale") is not None
        else "inferred_from_first_missing_scale"
    )
    trigger_fields = (
        f"trigger_error=client_request_timeout; triggering_input_scale={triggering_scale}; "
        f"request_timeout_s={timeout_s:g}; triggering_request_latency_s>{timeout_s:g}; "
        f"trigger_phase={request_phase}; trigger_request_id={request_id}; "
        f"context_source={context_source}"
    )

    for row in rows:
        key = _measurement_key(row)
        planned_scale = key[0]
        if attempted_key is not None and key == attempted_key:
            row["error"] = (
                "client_request_timeout: planned_request_attempted=true; "
                "measurement_row_completed=false; "
                f"planned_input_scale={planned_scale}; {trigger_fields}"
            )
            continue

        reason = (
            "triggering_scale_probe_timed_out"
            if planned_scale == triggering_scale
            else "skipped_after_prior_scale_timeout"
        )
        row["error"] = (
            "not_measured_after_timeout: planned_request_attempted=false; "
            "measurement_row_completed=false; "
            f"planned_input_scale={planned_scale}; reason={reason}; {trigger_fields}"
        )


def write_case_error_csv(
    *,
    task_info: TaskInfo,
    out_csv: str,
    cpu: int | None,
    mem: int | None,
    gpu: str,
    warmup: int,
    repeat: int,
    repeat_in_window: int,
    input_scales: str | None,
    error: str,
    preserve_existing: bool = False,
    annotate_existing_error_rows: bool = False,
    timeout_context: dict[str, Any] | None = None,
    result_origin: str = "formal_attempt",
    execution_kind: str = "docker",
    runtime_id: str = "",
) -> tuple[int, int]:
    """Write missing error rows and return successful-preserved/added counts."""
    if not str(error or "").strip():
        raise ValueError("error rows require a non-empty diagnostic")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    scales = resolve_input_scales(task_info.task_family, input_scales)

    from acprof.platform import detect_environment
    planned_rows = _planned_error_rows(
        scales=scales,
        cpu=cpu,
        mem=mem,
        gpu=gpu,
        warmup=warmup,
        repeat=repeat,
        repeat_in_window=repeat_in_window,
        error=error,
        result_origin=result_origin,
        execution_kind=execution_kind,
        runtime_id=runtime_id,
        environment_class=detect_environment().environment,
    )
    existing_rows: list[dict[str, Any]] = []
    if preserve_existing and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        with open(out_csv, encoding="utf-8", newline="") as stream:
            allow_large_csv_fields()
            reader = csv.DictReader(stream)
            optional_pixel_fields = {*PIXEL_COUNT_FIELDS, *PIXEL_RATE_SOURCES}
            missing_fields = set(CSV_FIELDS) - set(reader.fieldnames or []) - optional_pixel_fields
            if missing_fields:
                raise RuntimeError(
                    "partial case CSV is missing required fields: "
                    + ", ".join(sorted(missing_fields))
                )
            existing_rows.extend(reader)

    reconciled = reconcile_case_error_rows(
        planned_rows,
        existing_rows,
        error=error,
        preserve_existing=preserve_existing,
        annotate_existing_error_rows=annotate_existing_error_rows,
        timeout_context=timeout_context,
    )
    rows = reconciled.rows
    missing_rows = reconciled.missing_rows

    def write_error_rows(stream: TextIO) -> None:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    sidecar = case_sidecar(out_csv, "sniff_groups")
    if preserve_existing:
        if sidecar.exists():
            reconcile_sniff_group_sidecar(out_csv, sidecar)
            if missing_rows:
                def write_error_groups(stream: TextIO) -> None:
                    with sidecar.open("r", encoding="utf-8", newline="") as source:
                        stream.writelines(source)
                    for _ in missing_rows:
                        stream.write('{"sniff_group_id": ""}\n')

                atomic_write(sidecar, write_error_groups)
    elif sidecar.exists():
        atomic_write(
            sidecar,
            lambda stream: stream.writelines(
                '{"sniff_group_id": ""}\n' for _ in rows
            ),
        )

    atomic_write(out_csv, write_error_rows)

    existing_count = len(rows) - len(missing_rows)
    if preserve_existing:
        print(
            f"[case] Preserved existing rows: {existing_count} "
            f"(successful: {reconciled.preserved_success_rows}); "
            f"wrote error rows: {len(missing_rows)}; total: {len(rows)}"
        )
    else:
        print(f"[case] Wrote error rows: {out_csv} ({len(rows)} rows)")
    return reconciled.preserved_success_rows, len(missing_rows)


def write_startup_oom_pruned_case_csv(
    *,
    task_info: TaskInfo,
    output_dir: str,
    cpu: int,
    mem: int,
    gpu: str,
    reference_cpu: int,
    warmup: int,
    repeat: int,
    repeat_in_window: int,
    input_scales: str | None,
) -> str:
    case = ArtifactLayout.discover(output_dir).case(task_info.model_id, cpu, mem, gpu)
    case.csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv = str(case.csv)
    error = (
        "not_measured_after_startup_oom_pruning: "
        "planned_request_attempted=false; measurement_row_completed=false; "
        "reason=confirmed_startup_oom_at_reference_cpu; "
        f"reference_cpu_cores={reference_cpu}; reference_mem_cap_gb={mem}; "
        f"gpu_mode={normalize_gpu_mode(gpu)}; "
        "pruning_scope=container_startup_only; "
        "result_origin=inferred_not_measured"
    )
    write_case_error_csv(
        task_info=task_info,
        out_csv=out_csv,
        cpu=cpu,
        mem=mem,
        gpu=gpu,
        warmup=warmup,
        repeat=repeat,
        repeat_in_window=repeat_in_window,
        input_scales=input_scales,
        error=error,
        result_origin="inferred_not_measured",
    )
    return out_csv
