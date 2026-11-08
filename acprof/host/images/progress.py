"""Translate Docker JSON streams without inferring byte counts from console prose."""
from __future__ import annotations

import base64
import binascii
import json
import os
import time
from collections import deque
from typing import TextIO, cast
from uuid import uuid4

from acprof.events.image import ImageProgress
from acprof.events.preparation import encode_event
from acprof.host.images.transfers import TransferAccounting, byte_count
from acprof.host.preparation.preparation_log import PreparationLog, redact_text


def emit_image_progress(component: str, image: str, phase: str, **fields) -> None:
    progress = ImageProgress(component, image, phase, **fields)
    progress.validate()
    if os.environ.get("ACPROF_INTERACTIVE_PREPARATION") == "1" or os.environ.get("ACPROF_TUI") == "1":
        print(encode_event("environment", "running", image_progress=progress.to_dict()), flush=True)


class DockerOutput:
    def __init__(self, component: str, image: str, phase: str):
        self.component, self.image, self.phase = component, image, phase
        self.tail: deque[str] = deque(maxlen=80)
        self.step = ""
        self.error = ""
        self.last_update = 0.0
        self.seen = False
        self.operation_id = uuid4().hex
        self.accounting = TransferAccounting(build=phase == "building")
        self.activity = "work"
        self.current: int | None = None
        self.total: int | None = None
        self.last_detail = ""
        self._logs: dict[str, PreparationLog] = {}

    def write(self, text: str) -> int:
        if text.startswith("ACPROF_DEPENDENCY_PROGRESS "):
            try:
                value = json.loads(text.split(" ", 1)[1])
                activity, current, total = self.accounting.dependency(value)
                self.report(str(value.get("name", "")), current, total, activity=activity,
                            force=value.get("state") in {"cached", "complete", "failed"})
            except (ValueError, TypeError, AttributeError):
                self.accounting.partial = True
                self.log("[Docker dependency progress invalid]")
        self.log(text.rstrip("\n"))
        return len(text)

    def flush(self) -> None:
        pass  # The outer preparation evidence owns disk flushing.

    def finish(self) -> None:
        for log in self._logs.values():
            log.finish()
        self.report(self.last_detail, self.current, self.total, activity=self.activity, force=True)

    def transition(self, phase: str, *, detail: str = "") -> None:
        emit_image_progress(self.component, self.image, phase, detail=redact_text(detail),
                            operation_id=self.operation_id, **self.accounting.snapshot())

    def report(self, detail: str, current=None, total=None, *, activity="work", force=False) -> None:
        current, total = byte_count(current), byte_count(total)
        if current is None or total is None or total <= 0 or current > total:
            total = None
        self.last_detail, self.current, self.total, self.activity = detail, current, total, activity
        now = time.monotonic()
        if force or now - self.last_update >= 0.25:
            emit_image_progress(self.component, self.image, self.phase,
                                detail=redact_text(detail), current=current, total=total, activity=activity,
                                operation_id=self.operation_id, **self.accounting.snapshot())
            self.last_update = now

    def log(self, text: str) -> None:
        # Forward through PreparationTee so the existing bounded log owns persistence.
        safe = redact_text(text, limit=65536 - 64 if self.phase == "pulling" else 4096)
        self.tail.append(safe)
        # Retain pull diagnostics in the preparation log and optional TUI details.
        # Only the policy owner can decide whether this is a terminal failure.
        for line in safe.splitlines():
            print(("[docker][DEBUG] " if self.phase == "pulling" else "") + line, flush=True)

    def feed(self, line: str) -> None:
        self.seen = True
        try:
            value = json.loads(line)
        except ValueError:
            self.log(line.rstrip())
            return
        if not isinstance(value, dict):
            self.log(line.rstrip())
            return
        if "vertex" in value and "id" in value and "current" in value:
            self._status(value)
            return
        if "vertex" in value and "data" in value:
            self._log_record(value)
            return
        # buildx rawjson may emit a single vertex rather than a SolveStatus batch.
        if "id" in value and "name" in value and (value.get("started") or value.get("error")):
            self._vertex(value)
            return
        if "status" in value or "errorDetail" in value or "error" in value:
            error_detail = value.get("errorDetail") or {}
            error = (error_detail.get("message") if isinstance(error_detail, dict) else None) or value.get("error")
            detail = str(error or value.get("status", ""))
            if error:
                self.error = detail
            identifier = value.get("id", "")
            activity, current, total = self.accounting.engine(value)
            if not error or self.phase != "pulling":
                self.report(f"{identifier} {detail}".strip(), current, total, activity=activity, force=bool(error))
            self.log(f"{identifier} {detail}".strip())
            return
        if any(key in value for key in ("vertexes", "statuses", "logs", "warnings")):
            for vertex in value.get("vertexes", []):
                self._vertex(vertex)
            for status in value.get("statuses", []):
                self._status(status)
            for log in value.get("logs", []):
                self._log_record(log)
            return
        self.log(line.rstrip())

    def _vertex(self, vertex: dict) -> None:
        key = vertex.get("digest") or vertex.get("id")
        if vertex.get("cached") and isinstance(key, str):
            self.accounting.cached_vertices.add(key)
        if (vertex.get("started") and not vertex.get("completed") and not self.error) or vertex.get("error"):
            self.step = str(vertex.get("name", ""))
            self.report(self.step, force=bool(vertex.get("error")))
        if vertex.get("name"):
            self.log(str(vertex["name"]) + (" [cached]" if vertex.get("cached") else ""))
        if vertex.get("error"):
            self.error = str(vertex["error"])
            self.log(self.error)

    def _status(self, status: dict) -> None:
        activity, current, total = self.accounting.buildkit(status)
        self.report(f"{status.get('id', '')} {status.get('name', '')}".strip(), current, total, activity=activity)

    def _log_record(self, log: dict) -> None:
        try:
            vertex = str(log.get("vertex", ""))
            if vertex not in self._logs:
                self._logs[vertex] = PreparationLog(cast(TextIO, self), lambda exc: self.log(f"[Docker log error] {exc}"))
            channel = "stderr" if log.get("stream") == 2 else "stdout"
            self._logs[vertex].feed(channel, base64.b64decode(log.get("data", ""), validate=True).decode("utf-8", errors="replace"))
        except (ValueError, binascii.Error):
            self.log("[Docker log data could not be decoded]")

    @property
    def detail(self) -> str:
        return redact_text(self.error or "\n".join(self.tail), limit=12000)
