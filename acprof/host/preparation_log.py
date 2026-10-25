"""Bounded, redacted preparation text; no threads or file-descriptor capture."""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from threading import RLock
from typing import TextIO
from urllib.parse import urlsplit, urlunsplit

_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s<>\"']+", re.IGNORECASE)
_AUTH = re.compile(r"\b(Bearer|Basic)[ \t]+[^\s,;\"'<>]+", re.IGNORECASE)
_KEY_NAME = (
    r"(?:[A-Za-z][A-Za-z0-9]*[_-])*"
    r"(?:authorization|proxy[_-]?authorization|api[_-]?key|access[_-]?key|"
    r"secret[_-]?key|private[_-]?key|signing[_-]?key|session[_-]?key|"
    r"access[_-]?token|refresh[_-]?token|id[_-]?token|"
    r"client[_-]?secret|token|secret|password|passwd|credentials?)"
)
_KEY_VALUE = re.compile(
    rf"(?P<prefix>(?<![\w.-])[\"']?{_KEY_NAME}[\"']?[ \t]*[:=][ \t]*)"
    r"(?P<value>\"(?:\\.|[^\"\\])*(?:\"|$)|"
    r"'(?:\\.|[^'\\])*(?:'|$)|[^,;\r\n}\]]*)", re.IGNORECASE,
)
_LINE_BREAK = re.compile(r"\r\n|\r|\n")
_TRUNCATED = "... [truncated]"


def _clean_url(match: re.Match[str]) -> str:
    try:
        value = urlsplit(match.group())
        return urlunsplit((value.scheme, value.netloc.rsplit("@", 1)[-1], value.path, "", ""))
    except ValueError:
        return "[REDACTED_URL]"


def _clean_value(match: re.Match[str]) -> str:
    value = match["value"]
    quote = value[:1] if value[:1] in {'"', "'"} else ""
    return f"{match['prefix']}{quote}[REDACTED]{quote}"


def _utf8_size(text: str) -> int:
    return len(text.encode("utf-8", errors="replace"))


def redact_text(text: str, limit: int | None = 2048) -> str:
    """Redact a complete logical line, then bound UTF-8 output without splitting characters."""
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("limit must be a nonnegative integer or None")
    safe = _URL.sub(_clean_url, text)
    safe = _KEY_VALUE.sub(_clean_value, safe)
    safe = _AUTH.sub(lambda match: f"{match[1]} [REDACTED]", safe)
    raw = safe.encode("utf-8", errors="replace")
    if limit is None or len(raw) <= limit:
        return raw.decode("utf-8")
    if limit < len(_TRUNCATED):
        return _TRUNCATED[:limit]
    return raw[:limit - len(_TRUNCATED)].decode("utf-8", errors="ignore") + _TRUNCATED


@dataclass
class _PendingLine:
    parts: list[str] = field(default_factory=list)
    byte_size: int = 0
    dropping: bool = False
    skip_lf: bool = False

    def reset(self) -> None:
        self.parts.clear()
        self.byte_size = 0
        self.dropping = False


class PreparationLog:
    """Separate bounded fragments per stream; the context owns the underlying file.

    Limits count UTF-8 bodies and output bytes including LF. Oversized lines are
    omitted whole. Flush never publishes partial lines, since secrets can span writes.
    """

    def __init__(self, stream: TextIO, on_error: Callable[[Exception], None],
                 max_bytes: int = 8 * 1024 * 1024, max_line_bytes: int = 64 * 1024) -> None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        if type(max_line_bytes) is not int or max_line_bytes < 1:
            raise ValueError("max_line_bytes must be a positive integer")
        self._stream, self._on_error = stream, on_error
        self._max_bytes, self._max_line_bytes = max_bytes, max_line_bytes
        self._lock = RLock()
        self._pending = {"stdout": _PendingLine(), "stderr": _PendingLine()}
        self._full = self._failed = self._finished = False
        self.byte_size = self.omitted_lines = 0

    @property
    def status(self) -> str:
        if self._failed:
            return "failed"
        if self.omitted_lines:
            return "truncated"
        return "saved" if self._finished else "recording"

    def _fail(self, exc: Exception) -> None:
        if self._failed:
            return
        self._failed = True
        for line in self._pending.values():
            line.reset()
        try:
            self._on_error(exc)
        except Exception:
            pass  # Diagnostic reporting must not replace the observed operation.

    def _append(self, line: _PendingLine, text: str, start: int, end: int) -> None:
        if start == end or line.dropping:
            return
        remaining = self._max_line_bytes - line.byte_size
        if self._full or end - start > remaining:
            line.reset()
            line.dropping = True
            return
        part = text[start:end]
        size = _utf8_size(part)
        if size > remaining:
            line.reset()
            line.dropping = True
            return
        line.parts.append(part)
        line.byte_size += size

    def _publish_line(self, line: _PendingLine) -> None:
        dropping = line.dropping or self._full
        body = "" if dropping else "".join(line.parts)
        line.reset()
        if dropping:
            self.omitted_lines += 1
            return
        body = redact_text(body, limit=None)
        if _utf8_size(body) > self._max_line_bytes:
            self.omitted_lines += 1
            return
        record = body + "\n"
        size = _utf8_size(record)
        if self.byte_size + size > self._max_bytes:
            self._full = True
            self.omitted_lines += 1
            return
        try:
            written = self._stream.write(record)
            if written != len(record):
                if type(written) is int and 0 <= written < len(record):
                    self.byte_size += _utf8_size(record[:written])
                raise OSError("short write while saving preparation log")
            self.byte_size += size
        except Exception as exc:
            self._fail(exc)

    def feed(self, channel: str, text: str) -> None:
        with self._lock:
            if channel not in self._pending:
                raise ValueError("channel must be stdout or stderr")
            if self._finished or self._failed or not text:
                return
            line, start = self._pending[channel], 0
            if line.skip_lf:
                start = 1 if text.startswith("\n") else 0
                line.skip_lf = False
            for ending in _LINE_BREAK.finditer(text, start):
                self._append(line, text, start, ending.start())
                self._publish_line(line)
                if self._failed:
                    return
                line.skip_lf = ending.group() == "\r" and ending.end() == len(text)
                start = ending.end()
            self._append(line, text, start, len(text))

    def flush(self) -> None:
        with self._lock:
            if self._failed or self._finished:
                return
            try:
                self._stream.flush()
            except Exception as exc:
                self._fail(exc)

    def finish(self) -> None:
        """Complete fragments and flush once; the context closes the owned file."""
        with self._lock:
            if self._finished:
                return
            if not self._failed:
                for line in self._pending.values():
                    if line.parts or line.dropping:
                        self._publish_line(line)
                        if self._failed:
                            break
                self.flush()
            self._finished = True


class PreparationTee:
    """Non-owning text proxy; binary buffer and inherited-FD writes bypass it."""

    def __init__(self, original: TextIO, log: PreparationLog, channel: str) -> None:
        if channel not in {"stdout", "stderr"}:
            raise ValueError("channel must be stdout or stderr")
        self._original, self._log, self._channel = original, log, channel

    def __getattr__(self, name: str):
        return getattr(self._original, name)

    def write(self, text: str) -> int:
        written = self._original.write(text)
        if written is None:
            self._log.feed(self._channel, text)
        elif written > 0:
            self._log.feed(self._channel, text[:written])
        return written

    def writelines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self._original.flush()
        self._log.flush()

    def close(self) -> None:
        self.flush()
