"""Reconcile failed case rows before publishing CSV and sniff sidecars."""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, TextIO

from acprof.artifacts import atomic_write, read_json_object
from acprof.capabilities import collection_outcomes
from acprof.config import CSV_FIELDS, DEFAULT_REQUEST_TIMEOUT_SECONDS
from acprof.gpu_mode import normalize_gpu_mode
from acprof.host.detect import TaskInfo
from acprof.host.input_plan import (
    format_scale_value,
    resolve_input_scales,
)
from acprof.host.measurement.energy_validation import row_has_error_status
from acprof.host.measurement.publication import reconcile_sniff_group_sidecar
from acprof.host.measurement.requests import request_phase_context
from acprof.host.profiling.compute_plan import NCU_ERROR_FIELD, TORCH_ERROR_FIELD
from acprof.pixel_metrics import PIXEL_COUNT_FIELDS, PIXEL_RATE_SOURCES
from acprof.results import UNRESOLVED_TIMEOUT_ATTEMPT
from acprof.results.csv import (
    MISSING_CSV_TEXT,
    MeasurementKey,
    ResultValidationError,
    measurement_key,
    parse_result_stream,
    require_current_fields,
)
from acprof.results.layout import ArtifactLayout, case_sidecar
from acprof.results.metrics import order_csv_fields, supported_csv_header


@dataclass(frozen=True)
class CaseErrorRows:
    rows: list[dict[str, Any]]
    missing_rows: list[dict[str, Any]]
    preserved_success_rows: int


def _case_measurement_key(row: dict[str, Any]) -> MeasurementKey:
    try:
        return measurement_key(row)
    except ResultValidationError as exc:
        raise RuntimeError(f"invalid measurement key in partial case CSV: {exc}") from exc


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
    planned_by_key = {_case_measurement_key(row): row for row in planned_rows}
    if len(planned_by_key) != len(planned_rows):
        raise RuntimeError("duplicate planned measurement in case recovery")
    normalized_existing: list[dict[str, Any]] = []
    existing_keys = set()
    if preserve_existing:
        for row in existing_rows:
            key = _case_measurement_key(row)
            if key not in planned_by_key:
                raise RuntimeError(f"partial case CSV contains an unplanned measurement: {key!r}")
            if key in existing_keys:
                raise RuntimeError(f"partial case CSV contains a duplicate measurement: {key!r}")
            for field in ("runtime_kind", "runtime_id", "resource_scope"):
                actual = str(row.get(field) or "").strip()
                planned = str(planned_by_key[key].get(field) or "").strip()
                if (actual != planned and actual.lower() not in MISSING_CSV_TEXT
                        and planned.lower() not in MISSING_CSV_TEXT):
                    raise RuntimeError(f"partial case CSV has conflicting {field} for measurement {key!r}")
            existing_keys.add(key)
            normalized_row = dict.fromkeys(CSV_FIELDS, "nan") | row
            if row_has_error_status(normalized_row):
                existing_error = str(normalized_row.get("error") or "").strip()
                if not existing_error:
                    raise RuntimeError(f"partial case CSV has status=error without an error diagnostic: {key!r}")
                if annotate_existing_error_rows and error not in existing_error:
                    normalized_row["error"] = f"{existing_error}; {error}"
            normalized_existing.append(normalized_row)

    missing = {key: row for key, row in planned_by_key.items() if key not in existing_keys}
    missing_rows = (_timeout_placeholder_rows(missing, planned_by_key, timeout_context)
                    if timeout_context is not None and missing else list(missing.values()))
    return CaseErrorRows(
        rows=[*normalized_existing, *missing_rows],
        missing_rows=missing_rows,
        preserved_success_rows=collection_outcomes(normalized_existing)["row_counts"]["succeeded"],
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


def _timeout_window(
    planned: dict[MeasurementKey, dict[str, Any]], context: dict[str, Any],
) -> tuple[str | None, MeasurementKey | None, bool]:
    # Historical sidecars can omit the type and request ID. Explicit conflicts
    # carry no evidence about which planned window the request belonged to.
    if (context.get("error_type", "client_request_timeout") != "client_request_timeout"
            or context.get("measurement_completed", False) is not False):
        return None, None, False
    template = next(iter(planned.values()))
    try:
        scale = measurement_key({**template, "input_scale": context.get("input_scale")})[3]
    except ResultValidationError:
        return None, None, False
    phase = context.get("request_phase")
    try:
        request = request_phase_context(str(context.get("request_id") or ""))
    except ValueError:
        return scale, None, False
    if request["request_phase"] not in ("unknown", phase):
        return scale, None, False
    if phase == "auto_repeat_window_warmup":
        confirmed = context.get("measurement_repeat_idx") is None and any(key[3] == scale for key in planned)
        return scale, None, confirmed
    if phase not in ("measurement_warmup", "measurement_repeat"):
        return scale, None, False
    try:
        target = measurement_key({**template, "input_scale": scale, "warmup": int(phase == "measurement_warmup"),
                                  "repeat_idx": context.get("measurement_repeat_idx")})
    except ResultValidationError:
        return scale, None, False
    if "measurement_repeat_idx" in request and Decimal(target[5]) != request["measurement_repeat_idx"]:
        return scale, None, False
    matches = [key for key in planned if key[3:] == target[3:]]
    return scale, matches[0] if len(matches) == 1 else None, False


def _timeout_placeholder_rows(
    missing: dict[MeasurementKey, dict[str, Any]],
    planned: dict[MeasurementKey, dict[str, Any]],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    scale, attempted, phase_is_probe = _timeout_window(planned, context)
    scale_label = next((row["input_scale"] for key, row in planned.items() if key[3] == scale), scale or "unknown")
    timeout_s = timeout_context_float(context.get("request_timeout_s"), DEFAULT_REQUEST_TIMEOUT_SECONDS)
    phase = str(context.get("request_phase") or "unknown")
    request_id = str(context.get("request_id") or "unknown")
    source = "client_timeout_sidecar" if scale is not None else "unavailable"
    trigger_fields = (
        f"trigger_error=client_request_timeout; triggering_input_scale={scale_label}; "
        f"request_timeout_s={timeout_s:g}; triggering_request_latency_s>{timeout_s:g}; "
        f"trigger_phase={phase}; trigger_request_id={request_id}; context_source={source}"
    )
    identified = attempted is not None and attempted in missing
    rows = []
    for key, row in missing.items():
        if key == attempted:
            diagnostic = "client_request_timeout: planned_request_attempted=true"
        elif phase_is_probe or identified:
            reason = "skipped_after_prior_scale_timeout"
            if key[3] == scale:
                reason = "triggering_scale_probe_timed_out" if phase_is_probe else "skipped_after_request_timeout"
            diagnostic = f"not_measured_after_timeout: planned_request_attempted=false; reason={reason}"
        else:
            diagnostic = f"{UNRESOLVED_TIMEOUT_ATTEMPT}; reason=unresolved_timeout_window"
        rows.append({**row, "error": f"{diagnostic}; measurement_row_completed=false; "
                     f"planned_input_scale={row['input_scale']}; {trigger_fields}"})
    return rows


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
    fields = CSV_FIELDS
    layout = ArtifactLayout.from_csv(out_csv)
    if layout.layout_version == 2 and layout.result_csv.is_file():
        manifest = read_json_object(layout.result_csv)
        if manifest.get("schema_version") == 3:
            fields = manifest["wide_fields"]
    existing_rows: list[dict[str, Any]] = []
    if preserve_existing and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        with open(out_csv, encoding="utf-8-sig", newline="") as stream:
            try:
                source_fields, parsed = parse_result_stream(stream, Path(out_csv), None, allow_empty=True,
                                                            validate_measurements=False)
                require_current_fields(source_fields)
            except (ResultValidationError, csv.Error) as exc:
                raise RuntimeError(f"invalid partial case CSV {out_csv}: {exc}") from exc
        if layout.layout_version == 1:
            fields = (source_fields if supported_csv_header(source_fields) else
                      order_csv_fields([*CSV_FIELDS, *(name for name in source_fields if name not in CSV_FIELDS)]))
        unexpected = set(source_fields) - set(fields)
        if unexpected:
            raise RuntimeError(f"partial case CSV contains fields outside its result schema: {sorted(unexpected)}")
        missing_fields = set(fields) - set(source_fields) - {*PIXEL_COUNT_FIELDS, *PIXEL_RATE_SOURCES}
        if missing_fields:
            raise RuntimeError("partial case CSV is missing required fields: " + ", ".join(sorted(missing_fields)))
        existing_rows = parsed.rows

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
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", restval="nan")
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
