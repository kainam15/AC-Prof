"""Copied before replacement and run with a base Python; stdlib imports only.

The parent passes its measurement and update lock descriptors. They stay open
until this process exits, including when the TUI has already disappeared.
"""
from __future__ import annotations

import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


class UpdateInterrupted(RuntimeError):
    """Not InterruptedError: selectors catches that exception as an EINTR retry."""


@dataclass(frozen=True)
class UpdatePlan:
    prefix: str
    entrypoint: str
    manager: str
    base_python: str
    current: str
    target: str
    channel: str
    state_dir: str
    job_dir: str
    index_url: str = "https://pypi.org/simple"
    timeout: float = 600


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_receipt(prefix: Path) -> dict:
    data = tomllib.loads((prefix / "uv-receipt.toml").read_text(encoding="utf-8"))
    tool = data.get("tool")
    if not isinstance(tool, dict) or not isinstance(tool.get("options", {}), dict):
        raise ValueError("Invalid uv receipt: missing tool")
    return tool


def validate_uv_layout(prefix: Path, entrypoint: Path) -> dict:
    tool = read_receipt(prefix)
    requirements = tool.get("requirements")
    if (not isinstance(requirements, list) or len(requirements) != 1
            or not isinstance(requirements[0], dict) or requirements[0].get("name") != "acprof"
            or set(requirements[0]) - {"name", "specifier"}):
        raise ValueError("Custom source, extras or additional requirements need manual review")
    if tool.get("constraints") or tool.get("overrides") or tool.get("with-executables-from"):
        raise ValueError("Custom tool constraints need manual review")
    entries = tool.get("entrypoints", [])
    if (not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict)
            or entries[0].get("name") != "acprof"
            or not isinstance(entries[0].get("install-path"), str)
            or Path(entries[0].get("install-path", "")) != entrypoint):
        raise ValueError("The uv receipt does not own this command entrypoint")
    if not entrypoint.is_symlink() or entrypoint.resolve() != (prefix / "bin/acprof").resolve():
        raise ValueError("The command entrypoint points to a different installation")
    if not (prefix / "bin/acprof").is_file() or not os.access(entrypoint, os.X_OK):
        raise ValueError("The installed command is missing or not executable")
    return tool


def install_command(plan: UpdatePlan) -> list[str]:
    # Keep the receipt requirement unpinned for subsequent uv tool upgrades.
    # A newly published version or resolver backtracking fails exact verification.
    return [plan.manager, "--no-config", "--no-progress", "--color", "never",
            "tool", "install", "--force", "--upgrade", "acprof", "--prerelease",
            "allow" if plan.channel == "preview" else "disallow",
            "--python", plan.base_python, "--no-python-downloads", "--no-build",
            "--default-index", plan.index_url]


def install_environment(plan: UpdatePlan) -> dict[str, str]:
    # Remove uv configuration overrides (indexes, tool roots, constraints etc.).
    # Retain proxy/CA configuration without logging environment values.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("UV_", "PIP_", "PYTHON")) and k != "VIRTUAL_ENV"}
    env.update(UV_TOOL_DIR=str(Path(plan.prefix).parent),
               UV_TOOL_BIN_DIR=str(Path(plan.entrypoint).parent),
               UV_PYTHON_DOWNLOADS="never", UV_HTTP_TIMEOUT="30", UV_HTTP_RETRIES="1")
    if "UV_CACHE_DIR" in os.environ:
        env["UV_CACHE_DIR"] = os.environ["UV_CACHE_DIR"]
    return env


def verify_installation(plan: UpdatePlan, version: str, *, updated: bool = False) -> None:
    prefix, entry = Path(plan.prefix), Path(plan.entrypoint)
    tool = validate_uv_layout(prefix, entry)
    if updated and (tool["requirements"] != [{"name": "acprof"}]
                    or tool.get("options", {}).get("prerelease") != (
                        "allow" if plan.channel == "preview" else "disallow")):
        raise ValueError("Verification failed: uv receipt channel or unpinned requirement differs")
    code = (
        "import acprof,importlib.metadata,json,sys;"
        "print(json.dumps({'version':importlib.metadata.version('acprof'),"
        "'package_version':acprof.__version__,'prefix':sys.prefix,'file':acprof.__file__}))"
    )
    result = subprocess.run([str(prefix / "bin/python"), "-I", "-c", code],
                            capture_output=True, text=True, check=True, timeout=15,
                            cwd=plan.job_dir, env=install_environment(plan))
    identity = json.loads(result.stdout)
    if (identity.get("version") != version or identity.get("package_version") != version
            or Path(identity.get("prefix", "")).resolve() != prefix.resolve()
            or not Path(identity.get("file", "")).resolve().is_relative_to(prefix.resolve())):
        raise ValueError(f"Verification failed: expected {version} at {prefix}; got {identity}")
    command = subprocess.run([str(entry), "--version"], capture_output=True, text=True,
                             check=True, timeout=15, cwd=plan.job_dir, env=install_environment(plan))
    if command.stdout.strip() != f"AC-Prof {version}":
        raise ValueError(f"Verification failed: command returned {command.stdout.strip()!r}")


def failure_kind(error: BaseException | str) -> str:
    text = str(error).lower()
    if isinstance(error, PermissionError) or any(s in text for s in ("permission denied", "access is denied", "operation not permitted")):
        return "permission"
    if any(s in text for s in ("no space left", "disk full", "quota exceeded")):
        return "disk"
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) or "timed out" in text or "timeout" in text:
        return "timeout"
    if any(s in text for s in ("connect", "dns", "network", "certificate", "failed to fetch", "failed to download")):
        return "network"
    if any(s in text for s in ("verification", "expected", "receipt", "entrypoint", "installation", "python", "no solution")):
        return "environment"
    return "install"


class Transaction:
    def __init__(self, plan: UpdatePlan, *, lock_fds: tuple[int, ...] = ()):
        self.plan = plan
        self.lock_fds = lock_fds
        self.job = Path(plan.job_dir)
        self.backup = self.job / "backup"
        self.failed_environment = self.job / "failed-environment"
        self._staging_dir: Path | None = None
        self.state = {"status": "running", "phase": "backup", "current": plan.current,
                      "target": plan.target, "channel": plan.channel, "prefix": plan.prefix,
                      "job_dir": plan.job_dir, "backup_dir": str(self.backup),
                      "log": str(self.job / "install.log"),
                      "pid": os.getpid(), "error": "", "error_kind": "", "rollback": ""}

    def publish(self, **changes) -> None:
        self.state.update(changes, updated_at=datetime.now(UTC).isoformat())
        atomic_json(Path(self.plan.state_dir) / "status.json", self.state)
        with (self.job / "install.log").open("a", encoding="utf-8") as log:
            log.write(f"[{self.state['updated_at']}] {self.state['phase']} · {self.state['status']}\n")
            if changes.get("error"):
                log.write(f"{changes['error']}\nRecovery: {self.state['rollback']}\n")

    def install(self) -> None:
        self.publish(phase="resolve")
        with (self.job / "install.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(install_command(self.plan), cwd=self.job,
                                       env=install_environment(self.plan), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, start_new_session=True, pass_fds=self.lock_fds)
            assert process.stdout is not None
            tail = bytearray()
            deadline = time.monotonic() + self.plan.timeout
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while selector.get_map():
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Installation timed out")
                        for key, _ in selector.select(timeout=min(0.25, max(0, deadline - time.monotonic()))):
                            chunk = os.read(key.fd, 8192)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            log.write(chunk)
                            tail.extend(chunk)
                            del tail[:-16384]
                            line = chunk.decode("utf-8", errors="replace")
                            phase = ("download" if "Downloading" in line else
                                     "install" if any(s in line for s in ("Prepared", "Uninstalled", "Installed"))
                                     else None)
                            if phase and phase != self.state["phase"]:
                                self.publish(phase=phase)
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
                if process.returncode:
                    raise RuntimeError(tail.decode("utf-8", errors="replace").strip())
            finally:
                if process.poll() is None:
                    # Reap the installer before restoring any files.
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                process.stdout.close()

    def restore(self) -> None:
        self.publish(phase="rollback")
        prefix = Path(self.plan.prefix)
        # Renaming preserves shebangs and the original venv path. Retain the
        # failed environment as evidence; never delete unknown user directories.
        if prefix.exists():
            os.replace(prefix, self.failed_environment)
        os.replace(self.backup, prefix)
        entry = Path(self.plan.entrypoint)
        temporary = entry.with_name(f".acprof-restore-{os.getpid()}")
        temporary.symlink_to(json.loads((self.job / "entrypoint.json").read_text())["target"])
        os.replace(temporary, entry)
        verify_installation(self.plan, self.plan.current)

    def _prepare_backup_staging(self, prefix: Path) -> None:
        if self.job.stat().st_dev == prefix.parent.stat().st_dev:
            return
        staging = prefix.parent / f".acprof-update-{self.job.name}"
        if staging.is_symlink():
            raise ValueError("Refusing symlinked update staging directory")
        staging.mkdir(mode=0o700, exist_ok=True)
        staging_stat = staging.stat()
        if staging_stat.st_uid != os.getuid() or staging_stat.st_dev != prefix.parent.stat().st_dev:
            raise ValueError("Backup staging must share a user-owned filesystem with the installation")
        if staging.resolve().is_relative_to(prefix.resolve()):
            raise ValueError("Invalid update staging directory")
        self._staging_dir = staging
        self.backup = staging / "backup"
        self.failed_environment = staging / "failed-environment"
        self.state["backup_dir"] = str(self.backup)

    def _cleanup_backup_after_success(self) -> None:
        """Remove temporary backup copy after the updated installation is verified."""
        try:
            if self.backup.is_symlink():
                self.backup.unlink()
            elif self.backup.exists():
                shutil.rmtree(self.backup)
            if self._staging_dir is not None and self._staging_dir.is_dir() and not any(self._staging_dir.iterdir()):
                self._staging_dir.rmdir()
        except OSError as exc:
            warning = f"[cleanup][WARN] Failed to remove update backup {self.backup}: {exc}\n"
            try:
                with (self.job / "install.log").open("a", encoding="utf-8") as log:
                    log.write(warning)
            except OSError:
                print(warning.rstrip(), file=sys.stderr)

    def run(self) -> bool:
        mutated = False
        channel_written = False
        channel_path = Path(self.plan.state_dir) / "channel.json"
        original_channel: bytes | None = None
        try:
            self.publish()
            prefix = Path(self.plan.prefix)
            if (self.plan.channel not in ("stable", "preview") or prefix.name != "acprof"
                    or self.job.is_relative_to(prefix) or not prefix.is_absolute()):
                raise ValueError("Invalid installation plan")
            verify_installation(self.plan, self.plan.current)
            if channel_path.exists():
                original_channel = channel_path.read_bytes()
            if not os.access(prefix.parent, os.W_OK) or not os.access(Path(self.plan.entrypoint).parent, os.W_OK):
                raise PermissionError("Installation directory is not writable")
            self._prepare_backup_staging(prefix)
            size = sum(p.stat().st_size for p in prefix.rglob("*") if p.is_file() and not p.is_symlink())
            if shutil.disk_usage(prefix.parent).free < size * 2 + 100 * 1024 * 1024:
                raise OSError("No space left for an installation backup and replacement")
            atomic_json(self.job / "entrypoint.json", {"target": os.readlink(self.plan.entrypoint)})
            shutil.copytree(prefix, self.backup, symlinks=True)
            atomic_json(
                self.job / "backup-complete.json",
                {"version": self.plan.current, "backup_dir": str(self.backup)},
            )
            mutated = True
            self.install()
            self.publish(phase="verify")
            verify_installation(self.plan, self.plan.target, updated=True)
            # The existing TUI/user configuration is never read or rewritten.
            channel_written = True
            atomic_json(channel_path,
                        {"channel": self.plan.channel, "prefix": self.plan.prefix})
            self.publish(status="success", phase="complete")
            self._cleanup_backup_after_success()
            return True
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            error, kind = str(exc), failure_kind(exc)
            rollback = "not_needed"
            if mutated:
                try:
                    self.restore()
                    if channel_written:
                        if original_channel is None:
                            channel_path.unlink()
                        else:
                            temporary = channel_path.with_suffix(".restore")
                            temporary.write_bytes(original_channel)
                            os.replace(temporary, channel_path)
                    rollback = "restored"
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as recovery:
                    rollback = "failed"
                    error += f"\nRecovery failed: {recovery}"
            self.publish(status="failed", phase="complete", error=error, error_kind=kind, rollback=rollback)
            return False


def main() -> int:
    plan = UpdatePlan(**json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    lock_fds = tuple(int(fd) for fd in sys.argv[2].split(","))

    def interrupted(signum, frame):
        raise UpdateInterrupted(f"Updater interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    transaction = Transaction(plan, lock_fds=lock_fds)
    return 0 if transaction.run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
