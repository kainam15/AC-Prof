"""Versioned, low-frequency control records emitted outside measurement windows."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager

from acprof.artifacts import loads_finite_json

PREFIX = "ACPROF_EVENT "
VERSION = 1
EVENTS = frozenset({"case_started", "measurement_started", "measurement_stopped", "case_finished"})


def parse_event(line: str) -> dict | None:
    if not line.startswith(PREFIX):
        return None
    payload = loads_finite_json(line[len(PREFIX):])
    if not isinstance(payload, dict) or type(payload.get("version")) is not int or payload["version"] != VERSION:
        raise ValueError("unsupported AC-Prof progress event version")
    if payload.get("event") not in EVENTS or not isinstance(payload.get("case_id"), str) or not payload["case_id"]:
        raise ValueError("invalid AC-Prof progress event")
    if payload["event"] == "case_finished" and payload.get("status") not in {"ok", "error", "cancelled"}:
        raise ValueError("invalid AC-Prof case status")
    return payload


def emit_event(event: str, case_id: str, **fields) -> None:
    if os.environ.get("ACPROF_TUI") == "1":
        print(PREFIX + json.dumps({"version": VERSION, "event": event, "case_id": case_id, **fields},
                                 ensure_ascii=True, allow_nan=False, separators=(",", ":")), flush=True)


@contextmanager
def measurement_boundary(case_id: str):
    """One pair around the client lifetime, with no per-request control I/O."""
    emit_event("measurement_started", case_id)
    try:
        yield
    finally:
        emit_event("measurement_stopped", case_id)
