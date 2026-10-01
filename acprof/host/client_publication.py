"""Publish buffered request, result and diagnostic records after monitor shutdown."""
from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Dict, Optional

from acprof.config import (
    CSV_FIELDS,
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
    writer.writerow(out)
    f.flush()
    os.fsync(f.fileno())
    _append_sniff_group(sidecar_f, sniff_group_id)
    if diag_f is not None and idle_diag_record is not None:
        _append_idle_diag(diag_f, idle_diag_record)
