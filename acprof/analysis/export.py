"""Explicit, exclusive exports of computed analysis; never modify source results."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TextIO

from acprof.artifacts import atomic_write, atomic_write_json


def _flatten(record: dict, prefix: str = "") -> dict:
    flattened = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flattened.update(_flatten(value, name + "."))
        elif isinstance(value, list):
            flattened[name] = json.dumps(value, ensure_ascii=False, allow_nan=False)
        else:
            flattened[name] = value
    return flattened


def statistics_rows(payload: dict) -> list[dict]:
    """Flatten groups, retaining units, parameters, provenance and unavailable states."""
    groups = payload.get("groups")
    if not isinstance(groups, list) or any(not isinstance(group, dict) for group in groups):
        raise ValueError("CSV export requires window statistics or experiment comparison groups")
    parameters = {
        name: payload.get(name) for name in (
            "schema_version", "kind", "confidence", "resamples", "seed", "block_size",
            "method", "resampling_unit", "filter", "assumption", "direction", "purpose", "precision",
            "result_csv", "result_sha256", "source_sha256", "status",
            "run_status", "measurement_status", "quality_status", "quality_reasons",
        ) if name in payload
    }
    metadata = _flatten(parameters, "analysis.")
    return [{**metadata, **_flatten(group)} for group in groups]


def export_analysis(payload: dict, destination: Path, *, format: str) -> None:
    """Write a new JSON/CSV only on explicit request, with atomic no-clobber publication."""
    if format not in {"json", "csv"}:
        raise ValueError("Analysis export format must be csv or json")
    if destination.suffix.lower() != "." + format:
        raise ValueError(f"Choose a new .{format} destination")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Export destination already exists: {destination}")
    if format == "json":
        atomic_write_json(destination, payload, overwrite=False)
        return
    rows = statistics_rows(payload)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        raise ValueError("There are no statistics rows to export")

    def write(stream: TextIO) -> None:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    atomic_write(destination, write, overwrite=False)
