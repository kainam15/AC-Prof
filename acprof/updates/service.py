"""Durable update state and launch of the isolated installation process."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from acprof.host.run_state import MeasurementLock
from acprof.updates import worker
from acprof.updates.installation import Installation
from acprof.updates.releases import Channel, ReleaseVersion
from acprof.updates.worker import UpdatePlan, atomic_json, validate_uv_layout


def state_directory(prefix: Path, root: Path | None = None) -> Path:
    if root is None:
        root = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state") / "acprof/updates"
    return root / hashlib.sha256(str(prefix).encode()).hexdigest()[:16]


def read_state(directory: Path, name: str = "status.json") -> dict:
    try:
        value = json.loads((directory / name).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Invalid update state: {name}")
    return value


def update_active(directory: Path) -> bool:
    # Automatic installation is Linux-only; read-only checks work elsewhere.
    if sys.platform != "linux":
        return False
    import fcntl

    try:
        descriptor = os.open(directory / "active.lock", os.O_RDWR | os.O_NOFOLLOW)
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False
    finally:
        os.close(descriptor)


def saved_channel(directory: Path, fallback: Channel) -> Channel:
    state = read_state(directory, "channel.json")
    channel = state.get("channel", fallback)
    if channel not in ("stable", "preview"):
        raise ValueError("Invalid saved update channel")
    return channel


def make_plan(installation: Installation, release: ReleaseVersion, directory: Path) -> UpdatePlan:
    if not installation.supported or installation.manager is None:
        raise ValueError("This installation does not support automatic updates")
    if release.latest is None:
        raise ValueError("No compatible release is available")
    validate_uv_layout(installation.prefix, installation.entrypoint)
    if Path.cwd().resolve().is_relative_to(installation.prefix):
        raise ValueError("Start the TUI from a working directory outside the tool installation")
    base_python = Path(sys.executable).resolve()
    if base_python.is_relative_to(installation.prefix):
        base_python = Path(sys.base_prefix) / "bin/python3"
    if not base_python.is_file() or base_python.resolve().is_relative_to(installation.prefix):
        raise ValueError("Cannot find an independent Python interpreter for recovery")
    job = directory / "jobs" / uuid4().hex
    return UpdatePlan(str(installation.prefix), str(installation.entrypoint), str(installation.manager),
                      str(base_python), str(release.current), str(release.latest), release.channel,
                      str(directory), str(job))


def launch_update(plan: UpdatePlan) -> subprocess.Popen:
    """Hold the existing machine lock through spawn, then let the child own it."""
    import fcntl

    directory, job = Path(plan.state_dir), Path(plan.job_dir)
    with MeasurementLock() as measurement:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(directory / "active.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            job.mkdir(parents=True, mode=0o700)
            copied = job / "worker.py"
            shutil.copyfile(worker.__file__, copied)
            atomic_json(job / "plan.json", asdict(plan))
            atomic_json(directory / "status.json", {
                "status": "running", "phase": "backup", "current": plan.current, "target": plan.target,
                "channel": plan.channel, "prefix": plan.prefix, "job_dir": plan.job_dir,
                "log": str(job / "install.log"),
            })
            with (job / "worker.log").open("ab") as log:
                try:
                    descriptors = (*measurement.inherited_descriptors, descriptor)
                    return subprocess.Popen([plan.base_python, "-I", str(copied), str(job / "plan.json"),
                                             ",".join(map(str, descriptors))],
                                            stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                            cwd=job, start_new_session=True,
                                            pass_fds=descriptors)
                except OSError as exc:
                    atomic_json(directory / "status.json", {
                        "status": "failed", "phase": "complete", "error": str(exc),
                        "error_kind": worker.failure_kind(exc), "rollback": "not_needed", "job_dir": str(job),
                    })
                    raise
        finally:
            os.close(descriptor)
