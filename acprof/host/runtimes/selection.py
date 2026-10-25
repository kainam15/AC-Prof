"""Resolve execution once, before preparation; never retry a failed model elsewhere."""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

from acprof.capabilities import Capability
from acprof.platform import Environment, detect_environment


@dataclass(frozen=True)
class RuntimeSelection:
    requested: str
    selected: str
    reason: str
    docker: Capability | None = None

    def to_dict(self):
        return {"requested": self.requested, "selected": self.selected, "reason": self.reason,
                "docker": self.docker.to_dict() if self.docker is not None else None}


def probe_docker() -> Capability:
    from acprof.host.preflight import require_cgroup_prerequisites, require_native_docker
    diagnostic = io.StringIO()
    try:
        with contextlib.redirect_stderr(diagnostic):
            require_native_docker()
            require_cgroup_prerequisites()
    except SystemExit:
        detail = diagnostic.getvalue().strip()
        status = "permission_denied" if "permission denied" in detail.lower() else "unavailable"
        return Capability(status, detail, "docker_preflight")
    except OSError as exc:
        from acprof.capabilities import capability_from_error
        return capability_from_error(exc, source="docker_preflight")
    return Capability("available", "local Docker daemon verified", "docker_preflight")


def select_runtime(requested: str, *, profiling_mode: str, environment: Environment | None = None,
                   docker_probe=None) -> RuntimeSelection:
    if requested not in {"auto", "docker", "process"}:
        raise ValueError("runtime must be auto, docker or process")
    environment = environment or detect_environment()
    if requested == "process":
        if environment.system != "Linux":
            raise ValueError("Process Runtime requires Linux")
        if profiling_mode != "basic":
            raise ValueError("Process Runtime supports basic; full is never silently downgraded")
        return RuntimeSelection(requested, "process", "explicit Process Runtime")
    docker = (docker_probe or probe_docker)()
    if docker.status in {"available", "verified"}:
        return RuntimeSelection(requested, "docker", "local Docker prerequisites available", docker)
    if requested == "docker" or profiling_mode != "basic":
        raise ValueError(f"Docker Runtime unavailable: {docker.detail}; select basic Process explicitly")
    if environment.system != "Linux":
        raise ValueError("no usable execution runtime; Process Runtime requires Linux")
    return RuntimeSelection(requested, "process", f"Docker unavailable: {docker.detail}", docker)


def require_runtime_requests(args) -> None:
    """Validate the resolved runtime, including an automatic Process selection."""
    if args.runtime != "process":
        return
    if (args.profiling_mode != "basic" or args.compute_profile_tool != "none"
            or args.execution_profile_tool != "none" or args.dram_energy == "required"):
        raise ValueError("Process Runtime supports basic collection without independent profilers; requested metrics cannot be downgraded")
    if args.rebuild_image:
        raise ValueError("--rebuild-image applies to Docker; Process environments are immutable")
