"""Bounded original log records and reversible presentation projections."""
from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass

ANSI = re.compile(r"\x1b(?:\][^\x07\x1b]*(?:\x07|\x1b\\)|\[[0-?]*[ -/]*[@-~]|[@-Z\\-_])")
LEVEL = re.compile(r"(?:^|\[)(DEBUG|INFO|SUCCESS|WARNING|WARN|ERROR|CRITICAL)(?:\]|\s*:|\s+-\s+|\s)", re.I)
PREFIX = re.compile(r"^(?:\d{4}-\d\d-\d\d[ T][\d:.,+Z-]+\s+)?(?:\[[^\]]+\]\s*)*")
BUILD = re.compile(r"^#\d+\s+(.*)")
DETAIL = re.compile(r"(?:/[\w.~-]+){2,}|\b[0-9a-f]{32,64}\b", re.I)


@dataclass(frozen=True)
class LogRecord:
    sequence: int
    raw: str
    text: str
    level: str
    source: str
    scope: int
    detail: bool = False
    success: bool = False
    summary: str = ""


@dataclass(frozen=True)
class LogRow:
    record: LogRecord
    text: str
    count: int = 1


class LogRecords:
    """Memory is bounded by both original line count and UTF-8 byte count.

    No files, timers or global logging configuration are owned by this model.
    Unknown output remains visible. Only identified verbose output is folded.
    """

    def __init__(self, max_lines: int, max_bytes: int) -> None:
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        self.records: deque[LogRecord] = deque()
        self.byte_count = 0
        self.discarded = 0
        self.sequence = 0
        self.scope = 0
        self.continuation = ""

    def append(self, content: str) -> None:
        for raw in content.removesuffix("\n").split("\n"):
            record = self.classify(raw)
            size = len(raw.encode("utf-8")) + 1
            if size > self.max_bytes:
                self.discarded += 1
                continue
            self.records.append(record)
            self.byte_count += size
        self.trim()

    def trim(self) -> None:
        while self.records and (len(self.records) > self.max_lines or self.byte_count > self.max_bytes):
            self.byte_count -= len(self.records.popleft().raw.encode("utf-8")) + 1
            self.discarded += 1

    def classify(self, raw: str) -> LogRecord:
        self.sequence += 1
        text = ANSI.sub("", raw)
        prefix = PREFIX.match(text)
        prefix_end = prefix.end() if prefix else 0
        # Restrict inference to the log header, never e.g. "0 errors" or a JSON value.
        match = LEVEL.search(text[:prefix_end]) or LEVEL.match(text[prefix_end:])
        build = BUILD.match(text)
        source_match = re.match(r"\[([^\]]+)\]", text)
        source = "docker" if build or text.startswith("ERROR: failed to solve") else source_match[1] if source_match else ""
        level = match[1].upper() if match else "INFO"
        level = {"WARNING": "WARN", "CRITICAL": "ERROR", "SUCCESS": "INFO"}.get(level, level)
        success = bool(match and match[1].upper() == "SUCCESS") or text.startswith("✓ ")
        detail = level == "DEBUG"
        summary = ""
        if prefix_end or match or build:
            self.continuation = level if level in {"WARN", "ERROR"} else ""
        elif text.startswith("Traceback (most recent call last)"):
            level = self.continuation = "ERROR"
        elif self.continuation:
            level = self.continuation
        if text.startswith("ACPROF_FAILURE "):
            level = self.continuation = "ERROR"
        elif re.match(r"^.+:\d+:\s*(?:UserWarning|FutureWarning|DeprecationWarning|RuntimeWarning):", text):
            level = self.continuation = "WARN"
        if source in {"phase", "cmd"} or source in {"case", "run", "preparation"} and level == "INFO":
            self.scope += 1
        if build:
            body = build[1]
            if body.startswith("ERROR") or re.match(r"[\d.]+\s+(?:ERROR|error:|fatal:|Traceback)", body):
                level = "ERROR"
            success = body.startswith(("CACHED", "DONE"))
            detail = not (success or level == "ERROR" or body.startswith("["))
        json_start = text.find("{")
        if json_start >= 0 and (json_start == 0 or source in {"network-preflight", "network-download"}):
            try:
                value = json.loads(text[json_start:])
            except (ValueError, RecursionError):
                pass  # Unrecognized text stays visible, including malformed diagnostics.
            else:
                if isinstance(value, dict):
                    explicit = str(value.get("level", "")).upper()
                    if explicit in {"WARN", "WARNING", "ERROR", "CRITICAL"}:
                        level = "WARN" if explicit in {"WARN", "WARNING"} else "ERROR"
                    if value.get("status") in ("error", "failed", "failure"):
                        level = "ERROR"
                    if level not in {"WARN", "ERROR"}:
                        summary = text[:json_start] + "{ " + ", ".join(list(value)[:8]) + " … }"
        return LogRecord(self.sequence, raw, text, level, source, self.scope, detail, success, summary)

    @property
    def raw_text(self) -> str:
        return "\n".join(record.raw for record in self.records)

    def project(self, *, level: str, details: bool, query: str) -> list[LogRow]:
        rows: list[LogRow] = []
        pattern = re.compile(re.escape(query), re.I) if query else None
        previous: LogRecord | None = None
        failed_builds = {record.scope for record in self.records if record.source == "docker" and record.level == "ERROR"}
        for record in self.records:
            build_context = record.source == "docker" and record.scope in failed_builds
            if level != "ALL" and record.level != level and not (level == "ERROR" and build_context):
                previous = None
                continue
            if pattern:
                if pattern.search(record.text) is None:
                    continue
            elif not details and record.detail and not (record.source == "docker" and record.scope in failed_builds):
                previous = None
                continue
            text = record.text if details or pattern else record.summary or record.text
            if (not details and not pattern and previous and record.level == "WARN"
                    and (previous.raw, previous.source, previous.scope) == (record.raw, record.source, record.scope)):
                first = rows[-1]
                rows[-1] = LogRow(first.record, f"{record.text}  [×{first.count + 1}]", first.count + 1)
            else:
                rows.append(LogRow(record, text))
            previous = record
        return rows
