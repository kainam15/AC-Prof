"""Recover abandoned local sessions before any cold-start or measurement window."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

LIFECYCLE_LABEL = "org.acprof.container.lifecycle"
OWNER_PREFIX = "org.acprof.owner."
_OWNER_SCOPE_LABELS = (LIFECYCLE_LABEL, OWNER_PREFIX + "host", OWNER_PREFIX + "uid")


STOP_TIMEOUT_S = 15
REMOVE_TIMEOUT_S = 30
INSPECT_TIMEOUT_S = 15


class ContainerCleanupError(RuntimeError):
    """An owned container is still present or cannot be proven absent."""

    def __init__(self, identifier: str, operations: list[dict], *, final_state="unknown",
                 docker_state=None, run_error: BaseException | None = None):
        self.container_id = identifier
        self.operations = operations
        self.final_state = final_state
        self.docker_state = docker_state
        self.run_error = run_error
        super().__init__(f"Cannot remove owned container {identifier}: cleanup {final_state}; "
                         "aborting before the next case")

    def to_dict(self) -> dict:
        original = self.run_error
        return {"schema_version": 1, "status": "incomplete", "container_id": self.container_id,
                "final_state": self.final_state, "docker_state": self.docker_state,
                "operations": self.operations,
                "run_error": ({"exception_type": type(original).__name__, "detail": str(original)}
                              if original is not None else None)}


def owned_container_id(cidfile: Path) -> str:
    """A Docker-created cidfile is the ownership proof, never a mutable name."""
    try:
        identifier = cidfile.read_text().strip()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        raise ContainerCleanupError("", [{"operation": "read_cidfile", "exception_type": type(exc).__name__,
                                         "detail": str(exc)}]) from exc
    if not re.fullmatch(r"[0-9a-f]{64}", identifier):
        raise ContainerCleanupError("", [{"operation": "read_cidfile", "detail": "invalid immutable container ID"}])
    return identifier


def _execute(command: list[str], run: Callable, operations: list[dict], timeout: int):
    evidence = {"operation": command[1], "timeout_s": timeout}
    try:
        result = run(command, check=False, timeout=timeout)
        evidence.update(returncode=result.returncode, stderr=(result.stderr or "")[-4000:])
        return result
    except (OSError, subprocess.SubprocessError) as exc:
        evidence.update(returncode=None, exception_type=type(exc).__name__, detail=str(exc))
        return None
    finally:
        operations.append(evidence)


def _inspect_container(identifier: str, run: Callable, operations: list[dict], *, run_error=None) -> dict | None:
    """Return an identified container, or None only for explicit absence of its full ID."""
    inspected = _execute(["docker", "inspect", identifier], run, operations, INSPECT_TIMEOUT_S)
    if inspected is not None:
        if inspected.returncode and re.search(r"No such (?:object|container):?\s+" + identifier + r"(?=$|\s)",
                                               inspected.stderr or "", re.IGNORECASE):
            return None
        if inspected.returncode == 0:
            try:
                items = json.loads(inspected.stdout)
            except (TypeError, ValueError):
                items = None
            if (isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict)
                    and items[0].get("Id") == identifier):
                return items[0]
            operations[-1]["detail"] = "invalid inspection or container identity mismatch"
    raise ContainerCleanupError(identifier, operations, run_error=run_error)


def _list_owner_containers(current: dict[str, str], run: Callable, operations: list[dict]) -> tuple[str, ...]:
    command = ["docker", "ps", "-aq", "--no-trunc"]
    for key in _OWNER_SCOPE_LABELS:
        command += ["--filter", f"label={key}={current[key]}"]
    listed = _execute(command, run, operations, INSPECT_TIMEOUT_S)
    if listed is None or listed.returncode or not isinstance(listed.stdout, str):
        raise ContainerCleanupError("", operations)
    identifiers = tuple(listed.stdout.split())
    if any(not re.fullmatch(r"[0-9a-f]{64}", identifier) for identifier in identifiers):
        operations[-1]["detail"] = "invalid container ID"
        raise ContainerCleanupError("", operations)
    return identifiers


def remove_owned_container(identifier: str, run: Callable, *, stop: bool = False) -> None:
    """Bound every command and confirm failed removal using the immutable ID."""
    if not re.fullmatch(r"[0-9a-f]{64}", identifier):
        raise ValueError("refusing to remove a container without its owned immutable ID")
    original = sys.exc_info()[1]
    operations: list[dict] = []
    if stop:
        _execute(["docker", "stop", "--time", "10", identifier], run, operations, STOP_TIMEOUT_S)
    removed = _execute(["docker", "rm", "-f", identifier], run, operations, REMOVE_TIMEOUT_S)
    if removed is not None and removed.returncode == 0:
        return
    item = _inspect_container(identifier, run, operations, run_error=original)
    if item is None:
        return
    if isinstance(item.get("State"), dict):
        raise ContainerCleanupError(identifier, operations, final_state="present",
                                    docker_state=item["State"], run_error=original)
    raise ContainerCleanupError(identifier, operations, run_error=original)


def _process_identity(pid: int) -> tuple[str, str]:
    # comm may contain spaces and parentheses; field 22 is starttime in clock ticks.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return fields[19], fields[0]


def container_owner_labels() -> dict[str, str]:
    start, _ = _process_identity(os.getpid())
    return {
        LIFECYCLE_LABEL: "1",
        OWNER_PREFIX + "host": hashlib.sha256(Path("/etc/machine-id").read_bytes()).hexdigest(),
        OWNER_PREFIX + "uid": str(os.getuid()),
        OWNER_PREFIX + "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        OWNER_PREFIX + "pid": str(os.getpid()),
        OWNER_PREFIX + "start": start,
    }


def _owner_has_exited(labels: dict, current: dict[str, str]) -> bool:
    if any(labels.get(key) != current[key] for key in _OWNER_SCOPE_LABELS):
        return False
    boot = labels.get(OWNER_PREFIX + "boot")
    pid = labels.get(OWNER_PREFIX + "pid", "")
    start = labels.get(OWNER_PREFIX + "start", "")
    if (not isinstance(boot, str) or not boot
            or not isinstance(pid, str) or not pid.isdecimal() or int(pid) <= 0
            or not isinstance(start, str) or not start.isdecimal()):
        return False
    if boot != current[OWNER_PREFIX + "boot"]:
        return True
    try:
        actual_start, state = _process_identity(int(pid))
    except FileNotFoundError:
        return True
    except (OSError, ValueError, IndexError):
        # Permission failures and malformed procfs data do not prove abandonment.
        return False
    return actual_start != start or state in {"Z", "X"}


def recover_abandoned_containers(current: dict[str, str], run: Callable) -> tuple[str, ...]:
    """Reject current-owner residue; remove only positively abandoned containers."""
    operations: list[dict] = []
    identifiers = _list_owner_containers(current, run, operations)
    removed = []
    for identifier in identifiers:
        inspection: list[dict] = []
        item = _inspect_container(identifier, run, inspection)
        if item is None:
            continue  # Another cleanup finished first.
        if not isinstance(item.get("Config"), dict):
            raise ContainerCleanupError(identifier, [{"operation": "inspect", "detail": "invalid container config"}])
        labels = item["Config"].get("Labels") or {}
        if not isinstance(labels, dict):
            continue
        if all(labels.get(key) == value for key, value in current.items()):
            state = item.get("State")
            raise ContainerCleanupError(identifier, [{"operation": "preflight", "detail":
                "Current owner has an unremoved container; refusing a new start"}], final_state="present",
                docker_state=state if isinstance(state, dict) else None)
        if not _owner_has_exited(labels, current):
            continue
        remove_owned_container(identifier, run)
        removed.append(identifier)
    return tuple(removed)


def recover_cleanup_debt(cleanup_errors: list[dict], run: Callable) -> dict:
    """Prove prior cleanup complete before retrying; caller pins daemon and holds the measurement lock."""
    try:
        current = container_owner_labels()
    except (OSError, ValueError, IndexError) as exc:
        raise ContainerCleanupError("", [{"operation": "owner_scope", "exception_type": type(exc).__name__,
                                         "detail": str(exc)}]) from exc
    recovered = recover_abandoned_containers(current, run)
    identifiers: list[str] = []
    unknown_id = not cleanup_errors
    for error in cleanup_errors:
        identifier = error.get("container_id") if isinstance(error, dict) else None
        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{64}", identifier):
            unknown_id = True
        elif identifier not in identifiers:
            identifiers.append(identifier)
    operations: list[dict] = []
    for identifier in identifiers:
        item = _inspect_container(identifier, run, operations)
        if item is not None:
            state = item.get("State")
            raise ContainerCleanupError(identifier, operations, final_state="present",
                                        docker_state=state if isinstance(state, dict) else None)
    if unknown_id:
        remaining = _list_owner_containers(current, run, operations)
        if remaining:
            operations.append({"operation": "verify_cleanup_debt", "container_ids": list(remaining),
                               "detail": "Unknown prior container ID requires an empty host/user lifecycle scope"})
            raise ContainerCleanupError("", operations)
    return {"schema_version": 1, "status": "complete", "verified_container_ids": identifiers,
            "recovered_container_ids": list(recovered), "owner_scope": {key: current[key] for key in _OWNER_SCOPE_LABELS},
            "scoped_inventory_empty": True if unknown_id else None, "operations": operations}
