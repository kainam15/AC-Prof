"""Bounded Docker preparation diagnostics and explicit recovery capabilities."""
from __future__ import annotations

import os
import re
from contextvars import ContextVar
from urllib.parse import urlsplit

from acprof.host.preparation.preparation_log import redact_text
from acprof.network_policy import oci_source_url

preparation_log_path: ContextVar[str] = ContextVar("docker_preparation_log", default="")
rebuild_target: ContextVar[str] = ContextVar("docker_rebuild_target", default="")


def network_failure_kind(detail: str) -> str:
    text = detail.lower()
    if re.search(r"\b(?:401|403)\b", text):
        return "auth"
    for words, kind in (
        (("digest mismatch", "checksum", "failed verification"), "integrity"),
        (("unauthorized", "denied", "forbidden", "authentication required"), "auth"),
        (("certificate", "x509"), "tls"),
        (("proxyconnect", "proxy connection", "407 proxy"), "proxy"),
        (("no such host", "lookup", "name resolution"), "dns"),
        (("timeout", "timed out", "deadline exceeded"), "timeout"),
        (("no matching manifest", "does not match the specified platform", "does not provide the specified platform"), "platform"),
        (("manifest unknown", "manifest_unknown", "name unknown", "not found", "404"), "not-found"),
        (("tls",), "tls"),
    ):
        if any(word in text for word in words):
            return kind
    return "connection"


def frontend_failed(detail: str, step: str, reference: str) -> bool:
    if not reference:
        return False
    # Use the failed vertex or final error, never a successful frontend line in
    # the retained build log (a later wheel download can fail independently).
    if "resolve image config for docker-image://" in step or step.startswith("docker-image://" + reference):
        return True
    return any(reference in line and any(word in line.lower() for word in ("error", "failed"))
               for line in detail.splitlines())


def frontend_error(detail: str, reference: str, *, exit_code: int | None = None) -> DockerPreparationError:
    registry = urlsplit(oci_source_url(reference)).hostname or "unknown"
    reason = network_failure_kind(detail)
    error = DockerPreparationError(
        "docker_frontend_failed",
        f"BuildKit frontend download failed: registry={registry}; reason={reason}. "
        "Check ACPROF_DOCKER_HUB_MIRROR or the builder's network/proxy configuration, then retry preparation "
        "to rerun download preflight. Dependency mirrors and RUN --network do not configure frontend downloads.\n" + detail,
        image=reference,
        step=f"BuildKit frontend · {registry}", exit_code=exit_code,
    )
    error.network_context = "buildkit"
    error.registry = registry
    error.network_reason = reason
    error.actions = ["retry"]
    return error


def reason_code(detail: str, operation: str) -> str:
    text = detail.lower()
    network_reason = network_failure_kind(detail)
    if network_reason == "integrity":
        return "docker_integrity_failed"
    if network_reason == "tls":
        return "docker_tls_failed"
    if network_reason == "auth":
        return "docker_auth_failed"
    if (any(word in text for word in ("blob", "content digest", "content sha256", "layer"))
            and any(word in text for word in ("not found", "does not exist", "missing", "no such file"))):
        return "docker_content_incomplete"
    if "no space left" in text:
        return "docker_disk_full"
    if operation == "pull":
        if network_reason == "platform":
            return "docker_platform_mismatch"
        if network_reason == "not-found":
            return "docker_manifest_missing"
    if (network_reason in {"timeout", "dns", "proxy"}
            or any(word in text for word in ("connection reset", "connection refused", "network is unreachable",
                                             "registry unavailable", "service unavailable", "unexpected eof"))
            or re.search(r"\b(?:429|50[0-9])\b", text)):
        return "docker_network_failed"
    return f"docker_{operation}_failed"


class DockerPreparationError(RuntimeError):
    stage = "environment"

    def __init__(self, code: str, detail: str, *, step: str = "", image: str = "",
                 exit_code: int | None = None, content: dict | None = None, log_path: str = ""):
        self.reason_code = code
        self.detail = redact_text(detail, limit=12000)
        self.step = redact_text(step, limit=512)
        self.image = image
        self.exit_code = exit_code
        self.content = content or {}
        self.log_path = log_path or preparation_log_path.get()
        self.network_context = ""
        self.registry = ""
        self.network_reason = ""
        self.fallback_error: dict | None = None
        self.actions = ["retry", "rebuild"]
        if (self.content.get("repair_reference") and self.content.get("status") in {"incomplete", "unavailable"}
                and os.environ.get("ACPROF_REQUIRE_PREPARED_IMAGE") != "1"
                and not os.environ.get("ACPROF_MAX_DOWNLOAD", "").strip()):
            self.actions.insert(1, "repair")
        super().__init__(f"{self.step or image}: {self.detail}" + (f" (exit={exit_code})" if exit_code is not None else ""))

    def to_dict(self) -> dict:
        return {"reason_code": self.reason_code, "stage": self.stage, "step": self.step,
                "image": self.image, "exit_code": self.exit_code, "detail": self.detail,
                "content": self.content, "log_path": self.log_path, "actions": self.actions,
                "network_context": self.network_context, "registry": self.registry,
                "network_reason": self.network_reason, "fallback_error": self.fallback_error}
