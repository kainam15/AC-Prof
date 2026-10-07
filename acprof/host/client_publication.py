"""Publish buffered request, result and diagnostic records after monitor shutdown."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

from acprof.artifacts import atomic_write, loads_finite_json
from acprof.config import (
    CSV_FIELDS,
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


def _append_request_window(stream, *, sniff_group_id, input_scale, warmup, repeat_idx,
                           latencies, error="", failed_request_id="") -> None:
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
    stream.write(json.dumps(_json_safe(record), separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _append_row(
    writer: csv.DictWriter,
    row: Dict[str, Any],
    f,
    sidecar_f,
    sniff_group_id: str,
    diag_f=None,
    idle_diag_record: Optional[Dict[str, Any]] = None,
) -> None:
    out = {k: row.get(k, "") for k in CSV_FIELDS}
    if str(out.get("status") or "").strip().lower() == "error" and not str(
        out.get("error") or ""
    ).strip():
        raise RuntimeError("refusing to write status=error without an error diagnostic")
    _append_sniff_group(sidecar_f, sniff_group_id)
    writer.writerow(out)
    f.flush()
    os.fsync(f.fileno())
    if diag_f is not None and idle_diag_record is not None:
        _append_idle_diag(diag_f, idle_diag_record)
