"""Versioned, bounded control records emitted outside measurement windows."""
from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager

from acprof.artifacts import loads_finite_json

PREFIX = "ACPROF_EVENT "
VERSION = 1
MAX_MESSAGE = 64 * 1024
CASE_EVENTS = frozenset({"case_selected", "case_started", "measurement_started", "measurement_stopped",
                         "case_finished", "case_skipped"})
PROBE_EVENTS = frozenset({"probe_started", "probe_memory_started", "probe_request_started",
                          "probe_request_stopped", "probe_memory_finished", "probe_finished"})
EVENTS = CASE_EVENTS | PROBE_EVENTS | {"matrix_planned", "results_merging", "result_published",
                                      "run_finished", "probe_summary"}
PROBE_STATUSES = frozenset({"ok", "startup_oom", "runtime_oom", "cuda_oom", "oom", "timeout", "error"})


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _positive_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def parse_event(line: str) -> dict | None:
    if not line.startswith(PREFIX):
        return None
    if len(line) > MAX_MESSAGE:
        raise ValueError("progress message exceeds size limit")
    payload = loads_finite_json(line[len(PREFIX):])
    if not isinstance(payload, dict) or type(payload.get("version")) is not int or payload["version"] != VERSION:
        raise ValueError("unsupported AC-Prof progress event version")
    name = payload.get("event")
    if not isinstance(name, str) or name not in EVENTS:
        raise ValueError("invalid AC-Prof progress event")
    if name in CASE_EVENTS | PROBE_EVENTS:
        if not isinstance(payload.get("case_id"), str) or not payload["case_id"]:
            raise ValueError("invalid AC-Prof case identity")
    if name == "case_started" and "phase" in payload and payload["phase"] != "postprocessing":
        raise ValueError("invalid AC-Prof case phase")
    if name == "case_finished" and payload.get("status") not in {"ok", "error", "cancelled"}:
        raise ValueError("invalid AC-Prof case status")
    if name == "case_skipped" and payload.get("status") not in {"completed", "oom_pruned"}:
        raise ValueError("invalid AC-Prof skipped case status")
    if name == "run_finished" and payload.get("status") not in {"ok", "no_results", "failed", "cancelled"}:
        raise ValueError("invalid AC-Prof run status")
    if name == "matrix_planned" and not _positive_int(payload.get("total_cases")):
        raise ValueError("invalid AC-Prof matrix size")
    if name in {"case_selected", "probe_memory_started"}:
        inherited = name == "case_selected" and payload.get("runtime_kind") == "process"
        valid_resources = all(key in payload and (_positive_int(payload[key]) or inherited and payload[key] is None)
                              for key in ("cpu", "mem"))
        if (not valid_resources or not all(_positive_int(payload.get(key)) for key in ("current", "total"))
                or payload.get("runtime_kind", "docker") not in {"docker", "process"}
                or payload["current"] > payload["total"] or payload.get("gpu") not in {"off", "on"}):
            raise ValueError("invalid AC-Prof resource progress")
    if name in {"probe_started", "probe_finished"}:
        if not _positive_int(payload.get("cpu")) or payload.get("gpu") not in {"off", "on"}:
            raise ValueError("invalid AC-Prof probe resources")
    if name in {"probe_started", "probe_memory_started", "probe_finished"}:
        if not _positive_number(payload.get("input_scale")):
            raise ValueError("invalid AC-Prof probe scale")
    if name == "probe_started":
        candidates = payload.get("memory_candidates")
        if not isinstance(candidates, list) or not candidates or not all(_positive_int(x) for x in candidates):
            raise ValueError("invalid AC-Prof probe memory candidates")
    if name in {"probe_memory_finished", "probe_finished"}:
        if payload.get("status") not in PROBE_STATUSES or not _positive_int(payload.get("mem")):
            raise ValueError("invalid AC-Prof probe result")
    if name == "probe_finished":
        for field in ("request_s", "cold_start_s", "ready_plus_request_s"):
            value = payload.get(field)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError("invalid AC-Prof probe duration")
    if name in {"result_published", "probe_summary"}:
        field = "final_csv" if name == "result_published" else "path"
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError("invalid AC-Prof output path")
    return payload


def encode_event(event: str, case_id: str | None = None, **fields) -> str:
    payload = {"version": VERSION, "event": event, **fields}
    if case_id is not None:
        payload["case_id"] = case_id
    line = PREFIX + json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    parse_event(line)
    return line


def emit_event(event: str, case_id: str | None = None, **fields) -> None:
    if os.environ.get("ACPROF_TUI") == "1":
        print(encode_event(event, case_id, **fields), flush=True)


@contextmanager
def measurement_boundary(case_id: str):
    """One pair around the client lifetime, with no per-request control I/O."""
    emit_event("measurement_started", case_id)
    try:
        yield
    finally:
        emit_event("measurement_stopped", case_id)
