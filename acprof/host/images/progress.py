"""Translate Docker JSON streams without inferring byte counts from console prose."""
from __future__ import annotations

import base64
import binascii
import json
import math
import os
import re
import time
from collections import deque
from typing import TextIO, cast
from urllib.parse import urlsplit
from uuid import uuid4

from acprof.events.image import ImageProgress
from acprof.events.preparation import encode_event
from acprof.host.images.transfers import TransferAccounting, byte_count
from acprof.host.preparation.preparation_log import PreparationLog, redact_text
from acprof.network_policy import configured_oci_sources, oci_source_url


def emit_image_progress(component: str, image: str, phase: str, **fields) -> None:
    progress = ImageProgress(component, image, phase, **fields)
    progress.validate()
    if os.environ.get("ACPROF_INTERACTIVE_PREPARATION") == "1" or os.environ.get("ACPROF_TUI") == "1":
        print(encode_event("environment", "running", image_progress=progress.to_dict()), flush=True)


def _stall_threshold() -> float:
    try:
        seconds = float(os.environ.get("ACPROF_DOCKER_PROGRESS_STALL_SECONDS", "30"))
    except ValueError as exc:
        raise ValueError("ACPROF_DOCKER_PROGRESS_STALL_SECONDS must be finite and positive") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("ACPROF_DOCKER_PROGRESS_STALL_SECONDS must be finite and positive")
    return seconds


def _image_source(reference: str, owner: str) -> dict:
    reference = reference.removeprefix("docker-image://")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+-]*(?:@sha256:[a-f0-9]{64})?", reference):
        return {"network_owner": owner, "source_kind": "unknown"}
    url = urlsplit(oci_source_url(reference))
    domain = url.hostname or ""
    registry = "docker.io" if domain in {"docker.io", "index.docker.io", "registry-1.docker.io"} else url.netloc
    mirror = configured_oci_sources()["docker_hub_mirror"]
    is_mirror = bool(mirror and reference.startswith(mirror + "/"))
    return {"source_image": redact_text(reference, limit=512), "source_registry": registry,
            "source_domain": "registry-1.docker.io" if registry == "docker.io" else domain,
            "source_kind": "mirror" if is_mirror else "official" if registry in {"docker.io", "ghcr.io"} else "unknown",
            "source_category": "docker_hub" if registry == "docker.io" or is_mirror else "ghcr" if registry == "ghcr.io" else "custom",
            "network_owner": owner}


def _step_source(step: str) -> tuple[str, str]:
    frontend = re.search(r"docker-image://([^\s]+)", step)
    if frontend:
        return ("frontend" if "resolve image config" in step else "content"), frontend[1]
    metadata = re.search(r"load metadata for ([^\s]+)", step)
    if metadata:
        return "metadata", metadata[1]
    parent = re.search(r"(?:^|\] )FROM ([^\s]+)", step)
    if parent:
        return "content", parent[1]
    if "load build context" in step or "load build definition" in step or "load .dockerignore" in step:
        return "upload", ""
    return "build", ""


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
        self.activity = "build" if phase == "building" else "work"
        self.current: int | None = None
        self.total: int | None = None
        self.last_detail = ""
        self.started_at = self.last_progress_at = time.monotonic()
        self.stall_after_seconds = _stall_threshold()
        self.source = _image_source(image, "daemon") if phase == "pulling" else {"network_owner": "buildkit"}
        self._signature: tuple | None = None
        self._logs: dict[str, PreparationLog] = {}
        self._vertices: dict[str, str] = {}
        self._active_vertices: dict[str, str] = {}
        self._json_pending = ""

    def start(self, *, detail: str = "", frontend: str = "", cli: bool = False) -> None:
        if frontend:
            self.source = _image_source(frontend, "buildkit")
        self.transition(self.phase, detail=detail)
        self.accounting.started = True
        if cli:
            self.accounting.partial = True
            self.accounting.uncovered_sources.add("cli_output")
        self.report(detail, activity=self.activity, force=True)

    def write(self, text: str) -> int:
        if text.startswith("ACPROF_DEPENDENCY_PROGRESS "):
            try:
                value = json.loads(text.split(" ", 1)[1])
                activity, current, total = self.accounting.dependency(value)
                source = {"network_owner": "buildkit", **{
                    name: value[name] for name in ("source_domain", "source_kind", "source_category")
                    if isinstance(value.get(name), str)}}
                ImageProgress(self.component, self.image, self.phase, **source).validate()
                if source.get("source_domain") and not re.fullmatch(r"[A-Za-z0-9.:-]+", source["source_domain"]):
                    raise ValueError("invalid dependency source domain")
                self.source = source
                self.report(str(value.get("name", "")), current, total, activity=activity,
                            force=value.get("state") in {"cached", "complete", "failed"})
            except (ValueError, TypeError, AttributeError):
                self.accounting.incomplete()
                self.log("[Docker dependency progress invalid]")
        self.log(text.rstrip("\n"))
        return len(text)

    def flush(self) -> None:
        pass  # The outer preparation evidence owns disk flushing.

    def finish(self, *, succeeded: bool = False) -> None:
        for log in self._logs.values():
            log.finish()
        if self._json_pending:
            self.accounting.incomplete()
            self.log("[Docker progress JSON incomplete]")
            self._json_pending = ""
        self.accounting.completed = succeeded
        if not succeeded:
            self.accounting.incomplete()
        self.report(self.last_detail, self.current, self.total, activity=self.activity, force=True)

    def _fields(self) -> dict:
        now = time.monotonic()
        return {"operation_id": self.operation_id, **self.accounting.snapshot(), **self.source,
                "elapsed_seconds": max(0.0, now - self.started_at),
                "idle_seconds": max(0.0, now - self.last_progress_at),
                "stall_after_seconds": self.stall_after_seconds}

    def transition(self, phase: str, *, detail: str = "") -> None:
        emit_image_progress(self.component, self.image, phase, detail=redact_text(detail),
                            activity=self.activity, **self._fields())

    def report(self, detail: str, current=None, total=None, *, activity="work", force=False) -> None:
        current, total = byte_count(current), byte_count(total)
        if current is None or total is None or total <= 0 or current > total:
            total = None
        detail = redact_text(detail)
        snapshot = self.accounting.snapshot()
        signature = (detail, current, total, activity, tuple(snapshot.items()), tuple(self.source.items()))
        now = time.monotonic()
        changed = signature != self._signature
        if changed:
            self.last_progress_at = now
            self._signature = signature
        self.last_detail, self.current, self.total, self.activity = detail, current, total, activity
        if force or changed and now - self.last_update >= 0.25:
            emit_image_progress(self.component, self.image, self.phase, detail=detail,
                                current=current, total=total, activity=activity, **self._fields())
            self.last_update = now

    def log(self, text: str) -> None:
        # Forward through PreparationTee so the existing bounded log owns persistence.
        safe = redact_text(text, limit=65536 - 64 if self.phase == "pulling" else 4096)
        self.tail.append(safe)
        for line in safe.splitlines():
            print(("[docker][DEBUG] " if self.phase == "pulling" else "") + line, flush=True)

    def feed(self, line: str) -> None:
        self.seen = self.accounting.started = True
        # Older BuildKit rawjson used indented JSON; retain a bounded message.
        if self._json_pending or line.strip() == "{":
            self._json_pending += line if line.endswith("\n") else line + "\n"
            if len(self._json_pending.encode("utf-8")) > 65536:
                self.accounting.incomplete()
                self.log("[Docker progress JSON omitted: exceeds 64 KiB]")
                self._json_pending = ""
                return
            line = self._json_pending
        try:
            value = json.loads(line)
        except ValueError:
            if not self._json_pending:
                self.accounting.incomplete()
                self.log(line.rstrip())
                if self.phase == "building" and not line.lstrip().startswith("ERROR"):
                    self._select_step(line.rstrip())
                    self.report(line.rstrip(), activity=self.activity)
            return
        self._json_pending = ""
        if not isinstance(value, dict):
            self.accounting.incomplete()
            self.log(line.rstrip())
            return
        self._record(value)

    def _record(self, value: dict) -> None:
        if any(key in value for key in ("vertexes", "statuses", "logs", "warnings")):
            for name, consume in (("vertexes", self._vertex), ("statuses", self._status),
                                  ("logs", self._log_record), ("warnings", self._warning)):
                records = value.get(name)
                if records is None:
                    continue
                if not isinstance(records, list):
                    self.accounting.incomplete()
                    continue
                for record in records:
                    if isinstance(record, dict):
                        consume(record)
                    else:
                        self.accounting.incomplete()
            return
        if "data" in value:
            self._log_record(value)
        elif "id" in value and any(key in value for key in ("vertex", "current", "total")):
            self._status(value)
        elif "digest" in value or "id" in value and any(key in value for key in ("name", "cached", "started", "completed")):
            self._vertex(value)
        elif "status" in value or "errorDetail" in value or "error" in value:
            self._engine_record(value)
        else:
            self.accounting.incomplete()

    def _engine_record(self, value: dict) -> None:
        error_detail = value.get("errorDetail") or {}
        error = (error_detail.get("message") if isinstance(error_detail, dict) else None) or value.get("error")
        detail = str(error or value.get("status", ""))
        if error:
            self.error = detail
            self.accounting.incomplete()
        identifier = value.get("id", "")
        activity, current, total = self.accounting.engine(value)
        if not error or self.phase != "pulling":
            self.report(f"{identifier} {detail}".strip(), current, total, activity=activity, force=bool(error))
        self.log(f"{identifier} {detail}".strip())

    def _select_step(self, step: str) -> None:
        self.step = step
        self.activity, reference = _step_source(step)
        self.source = _image_source(reference, "buildkit") if reference else {"network_owner": "buildkit"}

    def _vertex(self, vertex: dict) -> None:
        key = vertex.get("digest") or vertex.get("id")
        name = str(vertex.get("name", ""))
        if isinstance(key, str):
            if name:
                self._vertices[key] = name
            else:
                name = self._vertices.get(key, "")
            if vertex.get("cached"):
                self.accounting.cached_vertices.add(key)
            if vertex.get("started") and not vertex.get("completed"):
                self._active_vertices[key] = name
            if vertex.get("completed") or vertex.get("error"):
                self._active_vertices.pop(key, None)
        active = vertex.get("started") and not vertex.get("completed") and not self.error
        if active or vertex.get("error"):
            self._select_step(name)
            self.report(name, activity=self.activity, force=True)
        elif name == self.step and vertex.get("completed"):
            next_step = next(reversed(self._active_vertices.values()), name)
            self._select_step(next_step)
            self.report(next_step, activity=self.activity)
        if name:
            self.log(name + (" [cached]" if vertex.get("cached") else ""))
        if vertex.get("error"):
            self.error = str(vertex["error"])
            self.accounting.incomplete()
            self.log(self.error)

    def _status(self, status: dict) -> None:
        vertex = self._vertices.get(str(status.get("vertex", "")), "")
        if vertex:
            self._select_step(vertex)
        else:
            self.step = ""
            self.source = {"network_owner": "buildkit"}
        parent_activity = _step_source(vertex)[0] if vertex else "work"
        if str(status.get("id", "")).startswith("resolve "):
            parent_activity = "frontend" if "docker-image://" in vertex else "metadata"
        metadata = parent_activity in {"frontend", "metadata"}
        activity, current, total = self.accounting.buildkit(status, metadata=metadata)
        if metadata or activity == "work":
            activity = parent_activity
        self.report(vertex or f"{status.get('id', '')} {status.get('name', '')}".strip(),
                    current, total, activity=activity)

    def _log_record(self, log: dict) -> None:
        try:
            vertex = str(log.get("vertex", ""))
            if vertex not in self._logs:
                self._logs[vertex] = PreparationLog(cast(TextIO, self), lambda exc: self.log(f"[Docker log error] {exc}"))
            channel = "stderr" if log.get("stream") == 2 else "stdout"
            self._logs[vertex].feed(channel, base64.b64decode(log.get("data", ""), validate=True).decode("utf-8", errors="replace"))
        except (ValueError, TypeError, binascii.Error):
            self.accounting.incomplete()
            self.log("[Docker log data could not be decoded]")

    def _warning(self, warning: dict) -> None:
        data = warning.get("short")
        if isinstance(data, str):
            try:
                self.log(base64.b64decode(data, validate=True).decode("utf-8", errors="replace"))
            except (ValueError, binascii.Error):
                self.log("[Docker warning could not be decoded]")

    @property
    def detail(self) -> str:
        return redact_text(self.error or "\n".join(self.tail), limit=12000)
