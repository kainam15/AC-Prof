"""Persist complete service startup logs before the owned container is removed."""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

from acprof.failures import FAILURE_PREFIX, Failure
from acprof.host.validation_logs import ValidationLog
from acprof.system import command as host_command


def capture_startup_diagnostics(container_id: str, reason: str, *, state=None, timed_out=False) -> tuple[str, Failure]:
    root = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state") / "acprof/container-logs"
    path = root / f"{container_id}.log"
    evidence = {"container_id": container_id, "log_path": str(path), "docker_state": state}
    code = "request_timeout" if timed_out else "resource_limit" if (
        isinstance(state, dict) and state.get("OOMKilled") is True and state.get("Running") is False
        and state.get("Restarting") is not True) else "runtime_initialization_failed"
    failure = Failure("load", code, reason, evidence=evidence, exception_type="ContainerStartupError")
    log = ValidationLog(path)
    try:
        result = log.run(["docker", "logs", container_id], host_command.run_command, timeout=15)
        evidence["log_capture_complete"] = result.returncode == 0
        if result.returncode:
            evidence["log_capture_error"] = f"docker logs exited with {result.returncode}"
        for raw in log.records(FAILURE_PREFIX, include_stderr=True):
            try:
                reported = Failure(**json.loads(raw))
            except (ValueError, TypeError) as exc:
                evidence["failure_record_error"] = str(exc)
                continue
            if code == "runtime_initialization_failed":
                failure = reported
    except (OSError, subprocess.SubprocessError) as exc:
        evidence.update(log_capture_complete=False, log_capture_error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            log.close()
        except OSError as exc:
            evidence.update(log_capture_complete=False, log_capture_error=f"{type(exc).__name__}: {exc}")
    return str(path), replace(failure, evidence={**failure.evidence, **evidence})
