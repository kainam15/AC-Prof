"""Ephemeral preparation-only download messages, independent of measurement events."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

PREFIX = "ACPROF_DOWNLOAD "
MAX_MESSAGE = 8192
STATES = frozenset({"downloading", "verifying", "complete", "failed", "cancelled"})


@dataclass(frozen=True)
class DownloadProgress:
    model: str
    source: str
    state: str
    total_bytes: int
    cached_bytes: int
    completed_bytes: int | None
    transferred_bytes: int | None
    rate_bytes_s: float | None
    files_completed: int
    files_total: int
    attempt: int = 1
    detail: str = ""
    version: int = 1

    def validate(self) -> None:
        if (self.version != 1 or type(self.version) is not int or not isinstance(self.state, str) or self.state not in STATES
                or not isinstance(self.model, str) or not 0 < len(self.model) <= 512
                or not isinstance(self.source, str) or self.source not in {"huggingface", "modelscope"}
                or not isinstance(self.detail, str) or len(self.detail) > 512):
            raise ValueError("invalid download progress identity")
        for value in (self.total_bytes, self.cached_bytes, self.files_completed, self.files_total, self.attempt):
            if type(value) is not int or value < 0:
                raise ValueError("invalid download progress count")
        for byte_count in (self.completed_bytes, self.transferred_bytes):
            if byte_count is not None and (type(byte_count) is not int or byte_count < 0):
                raise ValueError("invalid download progress bytes")
        if (self.attempt < 1 or self.cached_bytes > self.total_bytes or self.files_completed > self.files_total
                or self.completed_bytes is not None and self.completed_bytes > self.total_bytes):
            raise ValueError("inconsistent download progress")
        rate = self.rate_bytes_s
        if rate is not None and (type(rate) not in (int, float) or not math.isfinite(rate) or rate < 0):
            raise ValueError("invalid download progress rate")
        if rate is not None and self.transferred_bytes is None:
            raise ValueError("download speed without transfer evidence")


def encode_event(progress: DownloadProgress) -> str:
    progress.validate()
    line = PREFIX + json.dumps(asdict(progress), ensure_ascii=True, allow_nan=False)
    if len(line) > MAX_MESSAGE:
        raise ValueError("download message exceeds size limit")
    return line


def parse_event(line: str) -> DownloadProgress | None:
    if not line.startswith(PREFIX):
        return None
    if len(line) > MAX_MESSAGE:
        raise ValueError("download message exceeds size limit")
    try:
        progress = DownloadProgress(**json.loads(line[len(PREFIX):]))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid download progress message") from exc
    progress.validate()
    return progress
