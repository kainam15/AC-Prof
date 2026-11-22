"""Locate the host's uv and verify the CLI features used by Process preparation."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from acprof.capabilities import Capability
from acprof.system.command import run_command


def uv_install_paths() -> tuple[Path, ...]:
    directories = [os.environ.get("UV_INSTALL_DIR"), os.environ.get("XDG_BIN_HOME")]
    if data_home := os.environ.get("XDG_DATA_HOME"):
        directories.append(str(Path(data_home).parent / "bin"))
    directories.append(os.path.expanduser("~/.local/bin"))
    return tuple(dict.fromkeys(Path(directory) / "uv" for directory in directories
                               if directory and Path(directory).is_absolute()))


def locate_uv() -> Capability:
    if found := shutil.which("uv"):
        return Capability("available", found, "PATH", {"path": os.path.abspath(found)})
    denied = ""
    for path in (*uv_install_paths(), *(Path(d) / "uv" for d in os.get_exec_path())):
        try:
            if not path.is_file():
                continue
            if os.access(path, os.X_OK):
                return Capability("available", f"{path} (outside PATH)", "user executable",
                                  {"path": str(path.absolute()), "reason_code": "cli_not_on_path"})
            denied = denied or str(path)
        except PermissionError:
            denied = denied or str(path)
    if denied:
        return Capability("permission_denied", f"Cannot execute {denied}", "host executable",
                          {"reason_code": "permission_denied"})
    return Capability("tool_missing", "uv was not found in PATH or its installation directories",
                      "host executable", {"reason_code": "not_installed"})


def probe_process_uv(command_runner=run_command) -> Capability:
    located = locate_uv()
    if located.status != "available":
        return located
    path = located.evidence["path"]
    try:
        version = command_runner([path, "--version"], timeout=5)
        if version.returncode:
            raise RuntimeError((version.stderr or version.stdout).strip() or f"exit={version.returncode}")
        # Exercise argument parsing only. None of these commands creates an
        # environment, fetches an interpreter or installs a package.
        for args in (("venv", "--relocatable"), ("python", "find", "--no-python-downloads", "--offline"),
                     ("pip", "sync", "--require-hashes", "--link-mode", "copy")):
            result = command_runner([path, *args, "--help"], timeout=5)
            if result.returncode:
                error = (result.stderr or result.stdout).strip()
                detail = f"{located.detail}: {version.stdout.strip()}; uv {' '.join(args)} --help: {error or result.returncode}"
                # clap uses exit 2 for an unrecognized subcommand or argument.
                if result.returncode == 2:
                    return Capability("unsupported", detail, "uv CLI", {"reason_code": "version_incompatible"})
                raise RuntimeError(detail)
        return Capability("available", f"{located.detail}: {version.stdout.strip()}", "uv CLI", located.evidence)
    except PermissionError as exc:
        return Capability("permission_denied", str(exc), "uv CLI", {"reason_code": "permission_denied"})
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        return Capability("error", f"{path}: {exc}", "uv CLI", {"reason_code": "error"})


def require_process_uv() -> str:
    result = probe_process_uv()
    if result.status != "available":
        raise RuntimeError(f"Process preparation requires uv: {result.detail}")
    return result.evidence["path"]
