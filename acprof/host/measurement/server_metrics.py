"""Opt-in server metrics observation outside the HTTP load duration."""
from __future__ import annotations

import hashlib
import http.client
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlsplit

from acprof.analysis.server_metrics import (
    SERVER_METRICS_SCOPE,
    parse_server_metrics,
    summarize_server_metrics,
)
from acprof.system.cleanup import cleanup_on_exit


def _empty_snapshot() -> dict[str, Any]:
    return {"status": "not_collected", **dict.fromkeys(
        ("error", "started_s", "finished_s", "sha256", "bytes", "series", "process_start_time_s"),
    )}


def _read_metrics(origin: str, headers: Mapping[str, str], declaration: dict[str, Any]) -> bytes:
    target = urlsplit(origin)
    factory = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
    connection = factory(str(target.hostname), target.port, timeout=declaration["timeout_seconds"])
    with cleanup_on_exit(connection.close, description="server metrics connection"):
        connection.request("GET", declaration["path"], headers={
            **headers, "Connection": "close", "Accept": "text/plain; version=0.0.4",
        })
        response = connection.getresponse()
        with cleanup_on_exit(response.close, description="server metrics response"):
            if response.status != 200:
                raise ValueError(f"GET /metrics returned HTTP {response.status}")
            content_type = (response.getheader("Content-Type") or "").split(";", 1)[0].strip().lower()
            if content_type != "text/plain":
                raise ValueError("GET /metrics must return Prometheus text/plain")
            raw = response.read(declaration["max_response_bytes"] + 1)
            if len(raw) > declaration["max_response_bytes"]:
                raise ValueError("GET /metrics exceeded the 4 MiB response limit")
            if response.length not in (None, 0):
                raise ValueError("GET /metrics ended before Content-Length")
            return raw


def _capture(
    snapshot: dict[str, Any], origin: str, headers: Mapping[str, str], declaration: dict[str, Any], epoch: float,
) -> None:
    snapshot["started_s"] = time.perf_counter() - epoch
    try:
        raw = _read_metrics(origin, headers, declaration)
        snapshot.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        parsed = parse_server_metrics(raw.decode("utf-8"), declaration["model"])
        snapshot.update(parsed, status="ok")
    except BaseException as error:
        snapshot.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        snapshot["finished_s"] = time.perf_counter() - epoch


@contextmanager
def observe_server_metrics(
    report: dict[str, Any], origin: str, headers: Mapping[str, str],
) -> Iterator[None]:
    declaration = report["identity"].get("server_metrics")
    if declaration is None:
        yield
        return
    before, after = _empty_snapshot(), _empty_snapshot()
    observation = {
        "schema_version": 1, "kind": "vllm", "scope": SERVER_METRICS_SCOPE,
        "before": before, "after": after,
        "summary": summarize_server_metrics(before, after, declaration["model"]),
    }
    report["server_metrics"] = observation
    epoch = time.perf_counter()
    try:
        _capture(before, origin, headers, declaration, epoch)
        with cleanup_on_exit(lambda: _capture(after, origin, headers, declaration, epoch),
                             description="after-load server metrics"):
            yield
    finally:
        observation["summary"] = summarize_server_metrics(before, after, declaration["model"])
