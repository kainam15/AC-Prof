"""Bounded preparation messages; never used inside a measurement window."""
from __future__ import annotations

import json
import math
import os

PREFIX = "ACPROF_PREPARATION "
MAX_MESSAGE = 256 * 1024
STAGES = frozenset({"resolution", "interface", "dependencies", "preflight", "image", "environment", "model", "input", "runtime"})
STATUSES = frozenset({"not_started", "running", "passed", "failed", "waiting"})


def _finite_json_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite number")
    return number


def emit_progress(stage: str) -> None:
    """Called at actual preparation boundaries, never from measurement."""
    if os.environ.get("ACPROF_INTERACTIVE_PREPARATION") == "1":
        print(encode_event(stage, "running"), flush=True)


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
    value = json.loads(
        line[len(PREFIX):],
        parse_float=_finite_json_number,
        parse_constant=_finite_json_number,
    )
    if (not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1
            or value.get("stage") not in STAGES or value.get("status") not in STATUSES):
        raise ValueError("unsupported preparation message")
    request = value.get("request")
    if request is not None:
        if (not isinstance(request, dict) or type(request.get("id")) is not int or request["id"] < 1
                or request.get("kind") not in {"review", "error"}):
            raise ValueError("invalid preparation request")
        if request["kind"] == "review" and (not isinstance(request.get("questions"), list)
                                            or not request["questions"] and request.get("resolved") is not True):
            raise ValueError("review requires unresolved fields")
    if "input_plan" in value:
        validate_input_plan(value["input_plan"])
    return value


def validate_input_plan(value: dict) -> None:
    if (not isinstance(value, dict) or not isinstance(value.get("scales"), list) or not value["scales"]
            or len(value["scales"]) > 10000 or not isinstance(value.get("scale_type"), str)
            or not value["scale_type"] or any(type(scale) not in (int, float) or not math.isfinite(scale)
                or scale <= 0 for scale in value["scales"])):
        raise ValueError("invalid preparation input plan")


def encode_reply(request_id: int, action: str, *, answers: dict | None = None,
                 overrides: dict | None = None) -> str:
    value: dict[str, object] = {"id": request_id, "action": action}
    if answers is not None:
        value["answers"] = answers
    if overrides is not None:
        value["overrides"] = overrides
    line = json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n"
    if len(line) > MAX_MESSAGE:
        raise ValueError("preparation reply exceeds size limit")
    return line
