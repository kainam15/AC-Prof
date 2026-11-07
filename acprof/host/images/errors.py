"""Bounded Docker preparation diagnostics and explicit recovery capabilities."""
from __future__ import annotations

import os
from contextvars import ContextVar

from acprof.host.preparation.preparation_log import redact_text

preparation_log_path: ContextVar[str] = ContextVar("docker_preparation_log", default="")
rebuild_target: ContextVar[str] = ContextVar("docker_rebuild_target", default="")


def reason_code(detail: str, operation: str) -> str:
    text = detail.lower()
    if any(word in text for word in ("unauthorized", "authentication required", "denied:", "pull access denied")):
        return "docker_auth_failed"
    if (any(word in text for word in ("blob", "content digest", "content sha256", "layer"))
            and any(word in text for word in ("not found", "does not exist", "missing", "no such file"))):
        return "docker_content_incomplete"
    if any(word in text for word in ("timeout", "timed out", "no such host", "connection reset", "tls handshake", "connection refused")):
        return "docker_network_failed"
    if "no space left" in text:
        return "docker_disk_full"
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
        self.actions = ["retry", "rebuild"]
        if (self.content.get("repair_reference") and self.content.get("status") in {"incomplete", "unavailable"}
                and os.environ.get("ACPROF_REQUIRE_PREPARED_IMAGE") != "1"
                and not os.environ.get("ACPROF_MAX_DOWNLOAD", "").strip()):
            self.actions.insert(1, "repair")
        super().__init__(f"{self.step or image}: {self.detail}" + (f" (exit={exit_code})" if exit_code is not None else ""))

    def to_dict(self) -> dict:
        return {"reason_code": self.reason_code, "stage": self.stage, "step": self.step,
                "image": self.image, "exit_code": self.exit_code, "detail": self.detail,
                "content": self.content, "log_path": self.log_path, "actions": self.actions}
