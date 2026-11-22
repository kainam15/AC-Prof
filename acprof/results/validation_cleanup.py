"""Publish cleanup diagnostics without copying unrelated Docker inspect output."""
from __future__ import annotations

from acprof.system.redaction import redact_text


def _public_fields(record: dict, keys: tuple[str, ...]) -> dict:
    fields: dict[str, object] = {}
    for key in keys:
        if key not in record:
            continue
        value = record[key]
        if isinstance(value, str):
            fields[key] = redact_text(value)
        elif value is None or isinstance(value, (bool, float, int)):
            fields[key] = value
    return fields


def public_cleanup_error(error: dict) -> dict:
    public = _public_fields(error, ("schema_version", "status", "detail", "error", "exception_type",
                                    "container_id", "final_state", "run_error", "docker_state"))
    if isinstance(error.get("run_error"), dict):
        public["run_error"] = _public_fields(error["run_error"], ("exception_type", "detail"))
    if isinstance(error.get("docker_state"), dict):
        public["docker_state"] = _public_fields(error["docker_state"], ("Status", "Running", "Dead", "OOMKilled", "ExitCode"))
    if "operations" in error:
        operations = error["operations"]
        if not isinstance(operations, list) or not all(isinstance(op, dict) for op in operations):
            raise ValueError("cleanup operations must be a list of objects")
        public["operations"] = [
            _public_fields(op, ("operation", "returncode", "timeout_s", "exception_type", "detail", "stderr"))
            for op in operations
        ]
    return public


def cleanup_summary(errors: list[dict]) -> str:
    """Show cleanup itself. The earlier run exception remains separate in structured evidence."""
    reasons = []
    for error in errors:
        reason = error.get("detail") or error.get("error")
        state = error.get("final_state")
        if not reason:
            for op in reversed(error.get("operations", [])):
                message = op.get("detail") or (op.get("stderr") if op.get("returncode") != 0 else "")
                if message:
                    reason = f"{op.get('operation') or 'cleanup'}: {message}"
                    break
        if state:
            reason = f"{state}: {reason}" if reason else str(state)
        if reason:
            reasons.append(str(reason))
    return redact_text(" / ".join(dict.fromkeys(reasons)), limit=512)
