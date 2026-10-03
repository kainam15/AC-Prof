"""Bounded provenance probes, only during preparation outside measurement windows."""
from __future__ import annotations

import json
from pathlib import Path

from acprof import __version__
from acprof.host.command import run_command
from acprof.platform import Environment


def collect_platform_metadata(environment: Environment, project_dir: str | Path) -> dict:
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

    docker = command("docker", ["docker", "info", "--format", "{{json .}}"])
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
