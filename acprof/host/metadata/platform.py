"""Bounded provenance probes, only during preparation outside measurement windows."""
from __future__ import annotations

import json
import platform
from pathlib import Path

from acprof import __version__
from acprof.platform import Environment, detect_environment
from acprof.system.command import run_command


def collect_platform_metadata(environment: Environment, project_dir: str | Path, *, runtime_kind='docker') -> dict:
    errors = {}

    def command(name, args):
        try:
            result = run_command(args, cwd=project_dir, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=5, check=False)
            if result.returncode:
                errors[name] = (result.stderr or result.stdout).strip() or f"exit={result.returncode}"
                return None
            return result.stdout.strip() or None
        except (OSError, TimeoutError) as exc:
            errors[name] = str(exc)
            return None
        except Exception as exc:
            # Optional provenance must preserve a reason, never invent a version.
            errors[name] = str(exc)
            return None

    docker = command("docker", ["docker", "info", "--format", "{{json .}}"]) if runtime_kind == 'docker' else None
    try:
        info = json.loads(docker) if docker else {}
        if not isinstance(info, dict):
            raise ValueError("docker info is not an object")
    except ValueError as exc:
        errors["docker"] = str(exc)
        info = {}
    gpu = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            for name, probe in (("driver_version", pynvml.nvmlSystemGetDriverVersion),
                                ("nvml_version", pynvml.nvmlSystemGetNVMLVersion),
                                ("cuda_driver_version", pynvml.nvmlSystemGetCudaDriverVersion)):
                try:
                    value = probe()
                    gpu[name] = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
                except Exception as exc:
                    gpu[name] = None
                    errors[name] = str(exc)
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        errors["nvml"] = str(exc)
    return {"acprof_version": __version__,
            "git_commit": command("git", ["git", "rev-parse", "HEAD"]),
            "docker": {key: info.get(key) for key in (
                "ServerVersion", "KernelVersion", "OperatingSystem", "OSType", "Architecture",
                "CgroupVersion", "CgroupDriver", "Driver", "DefaultRuntime", "Runtimes",
                "ContainerdCommit", "RuncCommit")},
            "gpu": gpu, "errors": errors}


def _linux_environment_label() -> str:
    try:
        os_release = platform.freedesktop_os_release()
    except Exception:
        return "linux"

    distro_id = str(os_release.get("ID", "")).strip().lower()
    version_id = str(os_release.get("VERSION_ID", "")).strip().strip('"')
    if not distro_id:
        return "linux"
    if version_id:
        return f"{distro_id}{version_id}"
    return distro_id


def _windows_environment_label() -> str:
    release = str(platform.release()).strip().lower()
    if release in {"10", "11"}:
        return f"windows{release}"
    return "windows"


def _macos_environment_label() -> str:
    release = platform.mac_ver()[0].strip()
    if not release:
        return "macos"
    major = release.split(".", 1)[0]
    if major.isdigit():
        return f"macos{major}"
    return "macos"


def operating_system_label() -> str:
    """Return the display OS release, distinct from measurement policy identity."""
    try:
        system = platform.system()
    except Exception:
        return "unknown"

    if system == "Windows":
        label = _windows_environment_label()
    elif system == "Linux":
        label = _linux_environment_label()
    elif system == "Darwin":
        label = _macos_environment_label()
    else:
        label = str(system).strip().lower() or "unknown"

    if label != "unknown" and detect_environment().environment == "wsl2":
        return f"{label}+wsl"
    return label
