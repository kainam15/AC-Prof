"""Bounded local request inputs prepared before HTTP load scheduling."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from acprof.artifacts import MAX_JSON_ARTIFACT_BYTES, loads_finite_json, reject_isolated_surrogates
from acprof.experiments.streaming import request_digest, validate_request_credentials

__all__ = [
    "MAX_REQUEST_DATASET_BYTES", "MAX_REQUEST_DATASET_ROWS",
    "read_request_dataset", "request_digest", "validate_request_credentials",
]

MAX_REQUEST_DATASET_BYTES = 64 * 1024 * 1024
MAX_REQUEST_DATASET_ROWS = 100_000


def read_request_dataset(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read once, retaining exact text, source lines and the raw file digest."""
    source = path.resolve(strict=True)
    if not source.is_file():
        raise ValueError("request JSONL must be a regular file")
    bodies: list[dict[str, Any]] = []
    lines: list[int] = []
    digest = hashlib.sha256()
    total = 0
    with source.open("rb") as stream:
        for number, raw in enumerate(iter(lambda: stream.readline(MAX_JSON_ARTIFACT_BYTES + 1), b""), 1):
            total += len(raw)
            if total > MAX_REQUEST_DATASET_BYTES:
                raise ValueError("request JSONL exceeds the 64 MiB input limit")
            if len(raw) > MAX_JSON_ARTIFACT_BYTES:
                raise ValueError(f"request JSONL line {number} exceeds the 4 MiB request limit")
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                value = loads_finite_json(raw.decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("request must be a JSON object")
                reject_isolated_surrogates(value)
                validate_request_credentials(value)
            except (ValueError, UnicodeError) as error:
                raise ValueError(f"request JSONL line {number}: {error}") from error
            if len(bodies) >= MAX_REQUEST_DATASET_ROWS:
                raise ValueError("request JSONL exceeds the 100000 request limit")
            bodies.append(value)
            lines.append(number)
    if not bodies:
        raise ValueError("request JSONL must contain at least one request object")
    return bodies, {"path": str(source), "sha256": digest.hexdigest(), "rows": len(bodies), "lines": lines}
