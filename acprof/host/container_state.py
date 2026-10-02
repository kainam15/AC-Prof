"""Docker state evidence and startup failure classification; no service ownership."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from acprof.host.command import run_command

_LOG = logging.getLogger(__name__)


class ContainerStartupError(RuntimeError):
    """Typed evidence captured before removing a container that never reached /ready."""

    def __init__(self, message: str, *, state=None, timed_out=False, container_name="", container_id=""):
        super().__init__(message)
        self.state = state
        self.container_name = container_name
        self.container_id = container_id
        confirmed = (isinstance(state, dict) and state.get("OOMKilled") is True
                     and state.get("Running") is False and state.get("Restarting") is not True)
        # A timeout boundary is ambiguous even if a later inspect observes OOM.
        self.outcome = "timeout" if timed_out else "startup_oom" if confirmed else "error"


def inspect_container_state(container_name: str) -> Optional[Dict[str, Any]]:
    """Return Docker's runtime state without flooding readiness logs."""
    try:
        result = run_command(
            [
                "docker",
                "inspect",
                container_name,
                "--format",
                "{{json .State}}",
            ],
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        _LOG.debug("container inspect unavailable: error_type=%s", type(exc).__name__)
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        state = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        return None
    return state if isinstance(state, dict) else None


def container_startup_exit_error(
    container_name: str,
    memory_limit_gb: int,
) -> Optional[str]:
    """Describe a container that exited while the server was starting."""
    state = inspect_container_state(container_name)
    if not state:
        return None

    status = str(state.get("Status") or "").strip().lower()
    running = bool(state.get("Running"))
    restarting = bool(state.get("Restarting"))
    oom_killed = bool(state.get("OOMKilled"))
    if running or restarting:
        return None
    if not oom_killed and status not in {"dead", "exited", "removing"}:
        return None

    try:
        exit_code = int(state.get("ExitCode"))
    except (TypeError, ValueError):
        exit_code = -1
    docker_error = str(state.get("Error") or "").strip()
    detail = (
        f"container={container_name}, memory_limit={memory_limit_gb}g, "
        f"status={status or 'unknown'}, exit_code={exit_code}"
    )
    if docker_error:
        detail += f", docker_error={docker_error}"
    if oom_killed:
        return f"container_oom_killed during startup ({detail})"
    return f"container_exited_before_ready ({detail})"


def container_runtime_oom_error(
    container_name: str,
    memory_limit_gb: int,
    client_exit_code: int,
) -> Optional[str]:
    """Describe a workload-time cgroup OOM reported by Docker.

    Client-side monitors can observe a dead container before the orchestrator
    does and consequently return a monitor-specific exit code. Docker's
    explicit ``OOMKilled`` state is stronger evidence, so callers must consult
    it before classifying a non-zero client exit as a profiler failure.
    """
    state = inspect_container_state(container_name)
    if not state or not bool(state.get("OOMKilled")):
        return None

    status = str(state.get("Status") or "").strip().lower()
    try:
        container_exit_code = int(state.get("ExitCode"))
    except (TypeError, ValueError):
        container_exit_code = -1
    docker_error = str(state.get("Error") or "").strip()
    detail = (
        "container_runtime_oom: docker_oom_killed=true; "
        "measurement_row_completed=false; planned_request_attempted=unknown; "
        f"container={container_name}; memory_limit_gb={memory_limit_gb}; "
        f"container_status={status or 'unknown'}; "
        f"container_exit_code={container_exit_code}; "
        f"client_exit_code={client_exit_code}"
    )
    if docker_error:
        detail += f"; docker_error={docker_error}"
    return detail
