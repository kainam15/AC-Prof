"""Accumulate reported payload without treating content-store offsets as traffic."""
from __future__ import annotations

import re
from dataclasses import dataclass


def byte_count(value) -> int | None:
    return value if type(value) is int and value >= 0 else None


@dataclass
class Resource:
    total: int | None = None
    cached: int = 0
    hit: bool = False
    payload_complete: bool = False


class TransferAccounting:
    """One Docker operation. Object sizes deduplicate; retry payload is additive."""

    def __init__(self, *, build: bool, engine_offsets: bool = False):
        self.resources: dict[str, Resource] = {}
        self.counters: dict[str, int] = {}
        self.retrying: set[str] = set()
        self.cached_vertices: set[str] = set()
        self.transferred: int | None = None
        self.partial = build
        self.engine_offsets = engine_offsets
        self.started = False
        self.completed = False
        self.uncovered_sources = {"metadata", "protocol_overhead"}
        if build:
            self.uncovered_sources.update(("buildkit_layers", "unreported_build"))

    def incomplete(self) -> None:
        self.partial = True
        self.uncovered_sources.add("incomplete_stream")

    def resource(self, key: str, total=None) -> Resource:
        resource = self.resources.setdefault(key, Resource())
        size = byte_count(total)
        if size is not None and size > 0:
            resource.total = max(resource.total or 0, size)
        return resource

    def payload(self, key: str, current) -> None:
        current = byte_count(current)
        if current is None:
            return
        previous = self.counters.get(key, 0)
        if current < previous:
            # A regression without an explicit retry is ambiguous, not new bytes.
            self.incomplete()
            return
        self.transferred = (self.transferred or 0) + current - previous
        self.counters[key] = current

    def offset_payload(self, key: str, current) -> None:
        """Only observed growth is new evidence; the initial ingest may be old."""
        self.partial = True
        self.uncovered_sources.add("resume_baseline")
        current = byte_count(current)
        if current is None:
            return
        previous = self.counters.get(key)
        self.counters[key] = current
        if previous is not None and current > previous:
            self.transferred = (self.transferred or 0) + current - previous

    def engine(self, value: dict) -> tuple[str, int | None, int | None]:
        self.started = True
        key, status = value.get("id"), str(value.get("status", "")).lower()
        counts = value.get("progressDetail")
        counts = counts if isinstance(counts, dict) else {}
        current, total = byte_count(counts.get("current")), byte_count(counts.get("total"))
        activity = "extract" if status == "extracting" else "download" if status == "downloading" else "work"
        # Engine JSONProgress can carry arbitrary units, not only bytes.
        if counts.get("units") not in (None, "", "B", "bytes"):
            if activity == "download":
                self.incomplete()
            return "extract" if activity == "extract" else "work", None, None
        if not isinstance(key, str) or not key:
            if current is not None:
                self.incomplete()
            return activity, None, None
        if status.startswith("retrying"):
            self.retrying.add(key)
        elif status == "downloading":
            resource = self.resource(key, total)
            resource.hit = False
            if key in self.retrying:
                # Moby counts newly read bytes per attempt; Range shrinks total.
                self.counters.pop(key, None)
                self.retrying.remove(key)
            if self.engine_offsets:
                self.offset_payload(key, current)
                activity = "content"
            else:
                self.payload(key, current)
            resource.payload_complete = current is not None and total is not None and current == total
        elif status == "already exists":
            resource = self.resource(key, total)
            resource.hit = True
            resource.cached = resource.total or 0
        elif status in {"pulling fs layer", "waiting", "download complete", "pull complete"}:
            self.resource(key)
        if activity == "work":
            return activity, None, None
        return activity, current, total

    def buildkit(self, value: dict, *, metadata: bool = False) -> tuple[str, int | None, int | None]:
        self.started = True
        key, name = value.get("id"), str(value.get("name", "")).lower()
        current, total = byte_count(value.get("current")), byte_count(value.get("total"))
        if metadata:
            return "metadata", None, None
        if not isinstance(key, str) or not key:
            return "work", None, None
        if name == "downloading":
            self.resource(key, total)
            self.payload(key, current)
            return "download", current, total
        if not name and re.fullmatch(r"sha256:[a-f0-9]{64}", key):
            # Offsets include cached/shared content. Never infer a GET from them.
            self.resource(key, total)
            return "content", current, total
        if name in {"extracting", "transferring", "sending tarball"} or key in {
                "transferring context", "transferring dockerfile", "reading from stdin"}:
            return "extract" if name == "extracting" else "upload", current, total
        return "work", None, None

    def dependency(self, value: dict) -> tuple[str, int | None, int | None]:
        self.started = True
        key, transfer = value.get("id"), value.get("transfer_id")
        if (type(value.get("version")) is not int or value.get("version") != 1 or not isinstance(key, str) or not key
                or not isinstance(transfer, str) or not transfer):
            raise ValueError("invalid dependency progress identity")
        current, total = byte_count(value.get("current")), byte_count(value.get("total"))
        if (current is None or byte_count(value.get("received")) is None
                or byte_count(value.get("cached")) is None
                or value.get("total") is not None and total is None):
            raise ValueError("invalid dependency progress counts")
        resource = self.resource("dependency:" + key, total)
        if total is not None:
            resource.total = total
        resource.hit |= value.get("state") == "cached"
        resource.cached = byte_count(value.get("cached")) or 0
        self.payload("dependency:" + transfer, value.get("received"))
        return "download", current, total

    def snapshot(self) -> dict:
        cached = (self.completed and not self.partial and bool(self.resources)
                  and all(r.hit for r in self.resources.values()) and not self.transferred)
        transferred = 0 if cached else self.transferred
        if cached:
            status = "cached"
        elif not self.started:
            status = "waiting"
        elif transferred is None:
            status = "unavailable"
        elif (self.partial or not self.completed or self.retrying
              or any(not r.hit and not r.payload_complete for r in self.resources.values())):
            status = "partial"
        else:
            status = "reported"
        return {"transferred_bytes": transferred,
                "known_bytes": sum(r.total or 0 for r in self.resources.values()),
                "unknown_items": sum(r.total is None for r in self.resources.values()),
                "cached_bytes": sum(r.cached for r in self.resources.values()),
                "cached_items": sum(r.hit for r in self.resources.values()),
                "reused_steps": len(self.cached_vertices), "transfer_status": status,
                "uncovered_sources": tuple(sorted(self.uncovered_sources))}
