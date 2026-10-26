"""Owned Linux process sessions, including controller-death cleanup."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from acprof.host.runtimes.contracts import RuntimeCleanupError
from acprof.process_identity import same_process


class ProcessCleanupError(RuntimeCleanupError):
    pass


def require_process_supervision() -> str:
    from acprof.host.command import run_command
    result = run_command([sys.executable, "-I", str(Path(__file__).with_name("supervisor.py")), "--probe"],
                         check=False, timeout=10)
    if result.returncode:
        raise RuntimeError("Process ownership prerequisites unavailable: " + result.stderr.strip())
    return result.stdout.strip()


class ManagedProcess:
    """A control pipe remains owned by the caller, never inherited by its workload."""

    def __init__(self, command: list[str], *, python: str, env: dict[str, str],
                 directory: Path, name: str, cpuset: str = "", cgroup: str = "", pass_fds: tuple[int, ...] = ()):
        directory.mkdir(parents=True, exist_ok=True)
        self.log_path = directory / f"{name}.log"
        self.status_path = directory / f"{name}.process.json"
        self.owner_path = directory / f"{name}.process.owner.json"
        from acprof.artifacts import atomic_write_json
        launch = {"schema_version": 1, "status_file": self.status_path.name, "launch_state": "starting"}
        atomic_write_json(self.owner_path, launch)
        self.command = list(command)
        self.identity: dict = {}
        self._control: int | None = None
        self.process: subprocess.Popen | None = None
        read_fd, write_fd = os.pipe()
        try:
            supervisor = str(Path(__file__).with_name("supervisor.py"))
            self.process = subprocess.Popen([
                python, supervisor, str(read_fd), str(self.status_path), str(self.log_path),
                cpuset, cgroup, *command,
            ], env=env, pass_fds=(read_fd, *pass_fds), start_new_session=True,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._control = write_fd
        except BaseException:
            os.close(write_fd)
            atomic_write_json(self.owner_path, {**launch, "launch_state": "failed"})
            raise
        finally:
            os.close(read_fd)
        try:
            atomic_write_json(self.owner_path, {**launch, "launch_state": "started", "owner_pid": self.process.pid})
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if self.status_path.exists():
                    status = self.state()
                    if status.get("error"):
                        raise RuntimeError(status["error"])
                    self.identity = status["workload"]
                    self.owner_identity = status["owner"]
                    break
                if self.process.poll() is not None:
                    raise RuntimeError(f"process supervisor exited: {self.process.returncode}")
                time.sleep(0.01)
            else:
                raise TimeoutError("process supervisor did not publish ownership")
        except BaseException:
            self.close()
            raise

    @property
    def pid(self) -> int:
        return self.identity["pid"]

    def state(self) -> dict:
        return json.loads(self.status_path.read_text()) if self.status_path.exists() else {}

    def poll(self) -> int | None:
        result = self.state().get("returncode")
        if result is None and self.process is not None and self.process.poll() is not None:
            raise ProcessCleanupError('supervisor exited without a workload result; cleanup is unconfirmed')
        return result

    def wait(self, timeout: float) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.poll()
            if result is not None:
                return result
            time.sleep(0.02)
        raise subprocess.TimeoutExpired(self.command, timeout, output=self.log_tail())

    def log_tail(self) -> str:
        try:
            stream = self.log_path.open("rb")
        except FileNotFoundError:
            return ""
        with stream:
            end = stream.seek(0, os.SEEK_END)
            start = max(0, end - 16384)
            stream.seek(start)
            return stream.read(end - start).decode("utf-8", errors="replace")

    def evidence(self) -> dict:
        return {"kind": "process", "workload": self.identity,
                "owner": self.owner_identity, "status_file": str(self.status_path),
                "log_file": str(self.log_path)}

    def close(self) -> None:
        if self._control is not None:
            os.close(self._control)
            self._control = None
        if self.process is None:
            return
        try:
            self.process.wait(timeout=12)
        except subprocess.TimeoutExpired as exc:
            raise ProcessCleanupError("process supervisor cleanup timed out; retained ownership evidence") from exc
        status = self.state()
        if not status.get('cleanup_complete') or status.get("cleanup_error") or (self.identity and same_process(self.identity)):
            raise ProcessCleanupError(f"workload cleanup unconfirmed: {status}")


def recover_process(evidence: dict) -> None:
    """Never signal a recycled PID or an unverified process group during resume."""
    owner = evidence.get("owner", {})
    workload = evidence.get("workload", {})
    if owner and same_process(owner):
        raise ProcessCleanupError("original process supervisor still alive; cannot recover its run")
    try:
        status = json.loads(Path(evidence['status_file']).read_text())
    except (OSError, ValueError, KeyError) as exc:
        raise ProcessCleanupError(f"supervisor cleanup evidence is unavailable: {exc}") from exc
    if (status.get('cleanup_complete') and not status.get('cleanup_error')
            and status.get('owner') == owner and status.get('workload') == workload
            and not same_process(workload)):
        return
    raise ProcessCleanupError("supervisor cleanup evidence is missing or incomplete; recovery cannot prove all descendants exited")


def verify_process_cleanup(directory: Path) -> None:
    """Preparation/startup failures must pass the same recovery gate as cases.

    The launch ledger precedes Popen, so a supervisor dying before publishing
    its workload cannot disappear from the recovery inventory.
    """
    for path in directory.glob('*.process.owner.json'):
        try:
            launch = json.loads(path.read_text())
            if launch.get('launch_state') == 'failed':
                continue
            name = launch['status_file']
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError('invalid status filename')
            status = json.loads((directory / name).read_text())
            owner, workload = status.get('owner', {}), status.get('workload', {})
            if (launch.get('launch_state') != 'started' or owner.get('pid') != launch.get('owner_pid')
                    or not status.get('cleanup_complete') or status.get('cleanup_error')
                    or not owner or same_process(owner) or workload and same_process(workload)):
                raise ValueError('descendant cleanup is incomplete or unconfirmed')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ProcessCleanupError(f'Process recovery requires complete cleanup evidence: {path}: {exc}') from exc
