"""Preparation-only Docker progress; payload and resource sizes have distinct scopes."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ImageProgress:
    component: str
    image: str
    phase: str
    detail: str = ""
    current: int | None = None
    total: int | None = None
    content: dict | None = None
    reason_code: str = ""
    activity: str = "download"
    operation_id: str = ""
    transferred_bytes: int | None = None
    known_bytes: int | None = None
    unknown_items: int | None = None
    cached_bytes: int | None = None
    cached_items: int | None = None
    reused_steps: int = 0
    transfer_status: str = "unavailable"
    source_image: str = ""
    source_registry: str = ""
    source_domain: str = ""
    source_kind: str = ""
    source_category: str = ""
    network_owner: str = ""
    elapsed_seconds: float = 0.0
    idle_seconds: float = 0.0
    stall_after_seconds: float = 30.0
    uncovered_sources: tuple[str, ...] = ()

    def validate(self) -> None:
        if (self.component not in {"platform", "environment", "model", "service"}
                or self.phase not in {"checking", "pulling", "fallback", "building", "verifying", "complete", "failed", "cancelled"}
                or not isinstance(self.image, str) or not 0 < len(self.image) <= 512
                or not isinstance(self.detail, str) or len(self.detail) > 2048
                or not isinstance(self.reason_code, str) or len(self.reason_code) > 128):
            raise ValueError("invalid Docker preparation progress")
        if self.total is not None and self.current is None:
            raise ValueError("partial Docker byte counts")
        if self.current is not None and (type(self.current) is not int or self.current < 0):
            raise ValueError("invalid Docker byte counts")
        if self.total is not None and (type(self.total) is not int or self.total <= 0
                or self.current is None or self.current > self.total):
            raise ValueError("invalid Docker byte counts")
        for count in (self.transferred_bytes, self.known_bytes, self.unknown_items,
                      self.cached_bytes, self.cached_items, self.reused_steps):
            if count is not None and (type(count) is not int or count < 0):
                raise ValueError("invalid Docker accounting count")
        if type(self.reused_steps) is not int:
            raise ValueError("invalid Docker reused step count")
        if (self.activity not in {"download", "extract", "content", "upload", "work", "frontend", "metadata", "build"}
                or self.transfer_status not in {"waiting", "unavailable", "reported", "partial", "cached"}
                or not isinstance(self.operation_id, str) or len(self.operation_id) > 128):
            raise ValueError("invalid Docker accounting scope")
        for value in (self.source_image, self.source_registry, self.source_domain):
            if not isinstance(value, str) or len(value) > 512 or any(ord(c) < 32 for c in value):
                raise ValueError("invalid Docker source")
        if (self.source_kind not in {"", "official", "mirror", "unknown"}
                or self.source_category not in {"", "docker_hub", "ghcr", "debian", "pypi", "pytorch", "model", "custom"}
                or self.network_owner not in {"", "daemon", "buildkit", "host"}):
            raise ValueError("invalid Docker source scope")
        for seconds in (self.elapsed_seconds, self.idle_seconds, self.stall_after_seconds):
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError("invalid Docker progress timing")
        if self.stall_after_seconds <= 0:
            raise ValueError("invalid Docker progress threshold")
        if (not isinstance(self.uncovered_sources, (tuple, list)) or len(self.uncovered_sources) > 7
                or any(not isinstance(item, str) or item not in {"metadata", "protocol_overhead", "buildkit_layers", "unreported_build",
                                    "cli_output", "incomplete_stream", "resume_baseline"} for item in self.uncovered_sources)):
            raise ValueError("invalid Docker accounting coverage")
        if self.transfer_status == "cached" and self.transferred_bytes != 0:
            raise ValueError("invalid Docker cache evidence")
        if self.content is not None and (not isinstance(self.content, dict)
                or self.content.get("status") not in {"complete", "incomplete", "unavailable", "unverified"}):
            raise ValueError("invalid Docker content status")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)
