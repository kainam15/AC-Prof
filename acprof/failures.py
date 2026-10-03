"""Typed failures shared by preflight, containers and result consumers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

REASON_CODES = frozenset({
    "runtime_task_unsupported", "runtime_dependency_missing", "runtime_dependency_incompatible",
    "runtime_dependency_unknown", "precision_mismatch", "request_timeout", "resource_limit",
    "model_contract_required", "artifact_layout_unsupported", "processor_incompatible",
    "access_denied", "runtime_initialization_failed", "remote_code_disallowed",
    "compatibility_budget_exhausted", "inference_failed",
})
FAILURE_PREFIX = "ACPROF_FAILURE="


@dataclass(frozen=True)
class Failure:
    stage: str
    reason_code: str
    detail: str
    device: str = "unknown"
    runtime_profile: str = ""
    retryability: str = "not_retryable"
    evidence: dict[str, Any] = field(default_factory=dict)
    exception_type: str = ""

    def __post_init__(self):
        if self.reason_code not in REASON_CODES:
            raise ValueError(f"unregistered failure reason: {self.reason_code}")
        if self.retryability not in {"not_retryable", "after_configuration", "higher_budget", "transient"}:
            raise ValueError(f"invalid retryability: {self.retryability}")

    def to_dict(self) -> dict:
        return asdict(self)


class RuntimeFailure(ValueError):
    def __init__(self, failure: Failure):
        self.failure = failure
        super().__init__(f"[{failure.reason_code}] {failure.detail}")

    def __str__(self):
        return self.failure.detail + "\n" + FAILURE_PREFIX + json.dumps(self.failure.to_dict(), ensure_ascii=False)


def failure_from_exception(exc: BaseException, *, stage: str, device: str = "unknown",
                           runtime_profile: str = "", evidence: dict | None = None) -> Failure:
    """Keep the original typed cause; never classify natural-language messages."""
    cause, seen = exc, set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, RuntimeFailure):
            value = cause.failure
            return replace(value, device=device if value.device == "unknown" else value.device,
                           runtime_profile=value.runtime_profile or runtime_profile,
                           evidence={**value.evidence, **(evidence or {})})
        if isinstance(cause, ModuleNotFoundError):
            return Failure(stage, "runtime_dependency_missing", str(exc), device, runtime_profile,
                           "after_configuration", {"module": cause.name, **(evidence or {})}, type(cause).__name__)
        http_status = getattr(getattr(cause, "response", None), "status_code", None)
        if http_status in (401, 403):
            return Failure(stage, "access_denied", str(exc), device, runtime_profile,
                           "after_configuration", {**(evidence or {}), "http_status": http_status}, type(cause).__name__)
        if isinstance(cause, (TimeoutError, MemoryError, PermissionError)):
            code, retry = ("request_timeout", "higher_budget") if isinstance(cause, TimeoutError) else (
                ("resource_limit", "higher_budget") if isinstance(cause, MemoryError) else ("access_denied", "after_configuration"))
            return Failure(stage, code, str(exc), device, runtime_profile, retry, evidence or {}, type(cause).__name__)
        cause = cause.__cause__
    code = "inference_failed" if stage in {"predict", "preprocess", "completion", "postprocess", "validate_output"} else "runtime_initialization_failed"
    return Failure(stage, code, str(exc), device, runtime_profile,
                   evidence=evidence or {}, exception_type=type(exc).__name__)


def compatibility_status(failure: Failure | dict) -> str:
    code = failure.reason_code if isinstance(failure, Failure) else failure["reason_code"]
    if code == "resource_limit":
        return "unverified"
    if code in {"request_timeout", "compatibility_budget_exhausted", "runtime_dependency_unknown"}:
        return "inconclusive"
    if code == "model_contract_required":
        return "needs_configuration"
    if code == "access_denied":
        return "access_denied"
    return "failed"


def collect_failures(root: Path, case_csvs=()):
    from acprof.artifact_layout import ArtifactLayout, case_sidecar
    from acprof.artifacts import atomic_write_json
    failures = []
    for csv in case_csvs:
        path = case_sidecar(csv, "runtime_failures")
        if path.exists():
            failures.extend(json.loads(path.read_text())["failures"])
    path = ArtifactLayout.discover(root).path("runtime_failures.json")
    if failures or path.exists():
        # Resume archives previous case attempts. Refresh the current evidence
        # even if every retried case now succeeds; do not retain stale failures.
        atomic_write_json(path, {"schema_version": 1, "failures": failures})
