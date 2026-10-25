"""Serializable process artifacts and the common service lifecycle contract."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class RuntimeCleanupError(RuntimeError):
    """Cleanup cannot be proven; another case must not start."""

    run_error: BaseException | None = None

    def to_dict(self):
        return {'schema_version': 1, 'status': 'incomplete', 'detail': str(self),
                'run_error': str(self.run_error) if self.run_error else None}


@dataclass(frozen=True)
class ProcessInfo:
    python: str
    code_root: str
    model_path: str
    environment_id: str
    runtime_environment: dict[str, Any] = field(default_factory=dict)


def runtime_kind(artifact) -> str:
    return "process" if isinstance(artifact, ProcessInfo) else "docker"


def runtime_identity(artifact) -> str:
    return artifact.environment_id if isinstance(artifact, ProcessInfo) else artifact.tag


class ExecutionRuntime(Protocol):
    """One owner for preparation, validation, service lifetime and recovery.

    start owns launch and readiness as one exception-safe operation; stop owns
    termination and verified cleanup. A failed cleanup prevents the next case.
    """

    kind: str

    def prepare(self, task_info, project_dir, output_dir, *, rebuild=False, timeout_seconds=300): ...
    def validate(self, **kwargs) -> dict: ...
    def start(self, **kwargs): ...
    def stop(self, session, *, log_prefix=None) -> None: ...
    def failure(self, session, memory, returncode) -> str: ...
    def restore(self, snapshot: dict): ...


class ServiceSession(Protocol):
    name: str
    base_url: str
    host_port: int
    gpu_device: dict[str, Any]
