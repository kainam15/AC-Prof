"""Translate Docker JSON streams without inferring byte counts from console prose."""
from __future__ import annotations

import base64
import binascii
import json
import os
import time
from collections import deque
from typing import TextIO, cast

from acprof.events.image import ImageProgress
from acprof.events.preparation import encode_event
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
        self._logs = PreparationLog(cast(TextIO, self), lambda exc: self.log(f"[Docker log error] {exc}"))

    def write(self, text: str) -> int:
        self.log(text.rstrip("\n"))
        return len(text)

    def flush(self) -> None:
        pass  # The outer preparation evidence owns disk flushing.

    def finish(self) -> None:
        self._logs.finish()

    def report(self, detail: str, current=None, total=None, *, force=False) -> None:
        if not (type(current) is int and type(total) is int and 0 <= current <= total and total > 0):
            current = total = None
        now = time.monotonic()
        if force or now - self.last_update >= 0.25:
            emit_image_progress(self.component, self.image, self.phase,
                                detail=redact_text(detail), current=current, total=total)
            self.last_update = now

    def log(self, text: str) -> None:
        # Forward through PreparationTee so the existing bounded log owns persistence.
        safe = redact_text(text, limit=4096)
        self.tail.append(safe)
        print(safe, flush=True)

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
        # buildx rawjson may emit a single vertex rather than a SolveStatus batch.
        if "id" in value and "name" in value and (value.get("started") or value.get("error")):
            if not self.error or value.get("error"):
                self.step = str(value["name"])
            self.log(str(value["name"]) + (" [cached]" if value.get("cached") else ""))
        if "status" in value or "errorDetail" in value or "error" in value:
            error = value.get("errorDetail", {}).get("message") or value.get("error")
            detail = str(error or value.get("status", ""))
            if error:
                self.error = detail
            identifier = value.get("id", "")
            counts = value.get("progressDetail") or {}
            self.report(f"{identifier} {detail}".strip(), counts.get("current"), counts.get("total"), force=bool(error))
            self.log(f"{identifier} {detail}".strip())
            return
        if any(key in value for key in ("vertexes", "statuses", "logs", "warnings")):
            for vertex in value.get("vertexes", []):
                if (vertex.get("started") and not vertex.get("completed") and not self.error) or vertex.get("error"):
                    self.step = str(vertex.get("name", ""))
                    self.report(self.step, force=bool(vertex.get("error")))
                if vertex.get("name"):
                    self.log(str(vertex["name"]) + (" [cached]" if vertex.get("cached") else ""))
                if vertex.get("error"):
                    self.error = str(vertex["error"])
                    self.log(self.error)
            for status in value.get("statuses", []):
                # VertexStatus can also count arbitrary work units. Only known
                # byte-transfer statuses may drive the byte bar.
                byte_status = status.get("name", "").lower() in {"downloading", "extracting", "transferring", "sending tarball"}
                self.report(f"{status.get('id', '')} {status.get('name', '')}".strip(),
                            status.get("current") if byte_status else None,
                            status.get("total") if byte_status else None)
            for log in value.get("logs", []):
                try:
                    channel = "stderr" if log.get("stream") == 2 else "stdout"
                    self._logs.feed(channel, base64.b64decode(log.get("data", ""), validate=True).decode("utf-8", errors="replace"))
                except (ValueError, binascii.Error):
                    self.log("[Docker log data could not be decoded]")
            return
        self.log(line.rstrip())

    @property
    def detail(self) -> str:
        return redact_text(self.error or "\n".join(self.tail), limit=12000)
