"""Publish buffered request, result and diagnostic records after monitor shutdown."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

from acprof.artifacts import atomic_write, loads_finite_json
from acprof.config import (
    CSV_FIELDS,
)
from acprof.window_records import (
    WindowPublicationContext,
    commit_window_record,
    prepare_window_telemetry,
)


def reconcile_sniff_group_sidecar(csv_path: str | Path, sidecar_path: str | Path) -> None:
    """Repair only crash-created sidecar tail rows before appending new results."""
    csv_file = Path(csv_path)
    sidecar_file = Path(sidecar_path)
    if not csv_file.exists() or not sidecar_file.exists():
        return

    with csv_file.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) is None:
            committed_rows = 0
        else:
            committed_rows = sum(1 for _ in reader)

    with sidecar_file.open("r", encoding="utf-8", newline="") as stream:
        lines = stream.readlines()

    if len(lines) < committed_rows:
        raise RuntimeError(
            "sniff-group sidecar is missing committed rows: "
            f"{len(lines)} sniff-group rows for {committed_rows} committed CSV rows"
        )

    for line_number, line in enumerate(lines[:committed_rows], start=1):
        if not line.endswith("\n"):
            raise RuntimeError(
                f"sniff-group sidecar committed row {line_number} is truncated"
            )
        try:
            payload = loads_finite_json(line)
        except (ValueError, RecursionError) as exc:
            raise RuntimeError(
                f"sniff-group sidecar committed row {line_number} is malformed"
            ) from exc
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("sniff_group_id"), str)
        ):
            raise RuntimeError(
                f"sniff-group sidecar committed row {line_number} is invalid"
            )

    if len(lines) == committed_rows:
        return

    atomic_write(
        sidecar_file,
        lambda stream: stream.writelines(lines[:committed_rows]),
    )


def _append_sniff_group(sidecar_f, sniff_group_id: str) -> None:
    sidecar_f.write(json.dumps({"sniff_group_id": sniff_group_id}, ensure_ascii=True) + "\n")
    sidecar_f.flush()
    os.fsync(sidecar_f.fileno())


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _append_idle_diag(diag_f, record: Dict[str, Any]) -> None:
    diag_f.write(json.dumps(_json_safe(record), ensure_ascii=True, sort_keys=True) + "\n")
    diag_f.flush()
    os.fsync(diag_f.fileno())


def append_request_window(stream, *, sniff_group_id, input_scale, warmup, repeat_idx,
                           latencies, error="", failed_request_id="") -> dict[str, Any]:
    """Persist buffered successful requests after every monitor has stopped.

    Array position is the request index in ``<sniff_group_id>:<index>``.
    Failed attempts are identified separately and never enter latency statistics.
    """
    record = {
        "schema_version": 1, "sniff_group_id": sniff_group_id,
        "input_scale": input_scale, "warmup": warmup, "repeat_idx": repeat_idx,
        "source": "client_http", "latency_app_s": latencies,
        "status": "error" if error else "ok",
    }
    if error:
        record["error"] = error
    if failed_request_id:
        record["failed_request_id"] = failed_request_id
    payload = json.dumps(_json_safe(record), separators=(",", ":"), allow_nan=False) + "\n"
    tell = getattr(stream, "tell", None)
    start = tell() if callable(tell) else None
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
    return {"record": _json_safe(record), "journal": {
        "path": getattr(stream, "name", None), "start_byte": start,
        "length_bytes": len(payload.encode("utf-8")),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }}


def _append_row(
    writer: csv.DictWriter,
    row: Dict[str, Any],
    f,
    sidecar_f,
    sniff_group_id: str,
    diag_f=None,
    idle_diag_record: Optional[Dict[str, Any]] = None,
    publication: WindowPublicationContext | None = None,
    telemetry_evidence: dict[str, Any] | None = None,
    request_record: dict[str, Any] | None = None,
) -> None:
    out = {k: row.get(k, "") for k in CSV_FIELDS}
    if str(out.get("status") or "").strip().lower() == "error" and not str(
        out.get("error") or ""
    ).strip():
        raise RuntimeError("refusing to write status=error without an error diagnostic")
    reference = None
    if publication is not None and sniff_group_id:
        if telemetry_evidence is None:
            raise RuntimeError("measured window publication requires request and telemetry evidence")
        if publication.record_path(sniff_group_id).exists():
            raise RuntimeError("window is already committed; start a new case attempt")
        reference = prepare_window_telemetry(publication, sniff_group_id, telemetry_evidence)
        if request_record is None:
            raise RuntimeError("measured window publication requires a durable request record")
    csv_start = f.tell() if publication is not None else None
    sidecar_start = sidecar_f.tell() if publication is not None else None
    _append_sniff_group(sidecar_f, sniff_group_id)
    writer.writerow(out)
    f.flush()
    os.fsync(f.fileno())
    if diag_f is not None and idle_diag_record is not None:
        _append_idle_diag(diag_f, idle_diag_record)
    if reference is not None and publication is not None and request_record is not None:
        record_parent = publication.record_path(sniff_group_id).parent
        commit_window_record(
            publication, window_id=sniff_group_id, row=out,
            request_record=request_record, telemetry_reference=reference,
            projections={
                "csv": {"path": os.path.relpath(f.name, record_parent),
                        "start_byte": csv_start, "end_byte": f.tell()},
                "sniff_groups": {"path": os.path.relpath(sidecar_f.name, record_parent),
                                 "start_byte": sidecar_start, "end_byte": sidecar_f.tell()},
            },
        )
