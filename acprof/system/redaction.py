"""Redact credential-bearing diagnostic text before publishing it."""
from __future__ import annotations

import re
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
    r"'(?:\\.|[^'\\])*(?:'|$)|(?:\[REDACTED\])?[^,;\r\n}\]]*)", re.IGNORECASE,
)
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
