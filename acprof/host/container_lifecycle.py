"""Recover abandoned local sessions before any cold-start or measurement window."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable

LIFECYCLE_LABEL = "org.acprof.container.lifecycle"
OWNER_PREFIX = "org.acprof.owner."


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
    if any(labels.get(key) != current[key] for key in (
        LIFECYCLE_LABEL, OWNER_PREFIX + "host", OWNER_PREFIX + "uid",
    )):
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
    command = ["docker", "ps", "-aq", "--no-trunc"]
    for key in (LIFECYCLE_LABEL, OWNER_PREFIX + "host", OWNER_PREFIX + "uid"):
        command += ["--filter", f"label={key}={current[key]}"]
    listed = run(command, check=True, timeout=15)
    removed = []
    for identifier in listed.stdout.split():
        if not re.fullmatch(r"[0-9a-f]{64}", identifier):
            raise RuntimeError("Docker returned an invalid container ID during recovery")
        inspected = run(["docker", "inspect", identifier], check=False, timeout=15)
        if inspected.returncode:
            if "No such" in (inspected.stderr or ""):
                continue  # Another cleanup finished first.
            raise RuntimeError(f"Cannot inspect abandoned container {identifier}: {inspected.stderr}")
        payload = json.loads(inspected.stdout)
        if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
            raise RuntimeError("Docker returned an invalid container inspection during recovery")
        item = payload[0]
        config = item.get("Config") or {}
        labels = config.get("Labels") or {}
        if item.get("Id") != identifier or not isinstance(labels, dict) or not _owner_has_exited(labels, current):
            continue
        result = run(["docker", "rm", "-f", identifier], check=False, timeout=30)
        if result.returncode and "No such" not in (result.stderr or ""):
            raise RuntimeError(f"Cannot remove abandoned container {identifier}: {result.stderr}")
        removed.append(identifier)
    return tuple(removed)
