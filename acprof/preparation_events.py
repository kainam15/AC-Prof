"""Bounded preparation messages; never used inside a measurement window."""
from __future__ import annotations

import json

PREFIX = "ACPROF_PREPARATION "
MAX_MESSAGE = 256 * 1024
STAGES = frozenset({"resolution", "dependencies", "preflight", "image", "input", "runtime"})
STATUSES = frozenset({"not_started", "running", "passed", "failed", "waiting"})


def encode_event(stage: str, status: str, **fields) -> str:
    line = PREFIX + json.dumps({"version": 1, "stage": stage, "status": status, **fields},
                               ensure_ascii=True, allow_nan=False)
    parse_event(line)
    return line


def parse_event(line: str) -> dict | None:
    if not line.startswith(PREFIX):
        return None
    if len(line) > MAX_MESSAGE:
        raise ValueError("preparation message exceeds size limit")
    value = json.loads(line[len(PREFIX):])
    if (not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1
            or value.get("stage") not in STAGES or value.get("status") not in STATUSES):
        raise ValueError("unsupported preparation message")
    request = value.get("request")
    if request is not None:
        if (not isinstance(request, dict) or type(request.get("id")) is not int or request["id"] < 1
                or request.get("kind") not in {"review", "error"}):
            raise ValueError("invalid preparation request")
        if request["kind"] == "review" and (not isinstance(request.get("questions"), list)
                                            or not request["questions"]):
            raise ValueError("review requires unresolved fields")
    return value


def encode_reply(request_id: int, action: str, *, answers: dict | None = None) -> str:
    value: dict[str, object] = {"id": request_id, "action": action}
    if answers is not None:
        value["answers"] = answers
    line = json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n"
    if len(line) > MAX_MESSAGE:
        raise ValueError("preparation reply exceeds size limit")
    return line
