"""Explicit, exclusive exports of computed analysis; never modify source results."""
from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from acprof.artifacts import atomic_write, atomic_write_json


def _leaves(record: dict, prefix: str = "") -> Iterator[tuple[str, object]]:
    """Visit columns in insertion order without encoding list-valued cells."""
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _leaves(value, name + ".")
        else:
            yield name, value


def _flatten(record: dict, prefix: str = "") -> dict:
    return {name: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, list) else value
            for name, value in _leaves(record, prefix)}


def _prepare_statistics_csv(payload: dict) -> tuple[list[str], Iterator[dict]]:
    """Discover the full header, then flatten each group only as it is written."""
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
    # Later groups can introduce columns. Scan their keys first, without keeping
    # rows or serializing list cells that will only be needed during writing.
    fields = dict.fromkeys(metadata)
    for group in groups:
        fields.update(dict.fromkeys(name for name, _ in _leaves(group)))
    if not groups or not fields:
        raise ValueError("There are no statistics rows to export")
    return list(fields), ({**metadata, **_flatten(group)} for group in groups)


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
    fields, rows = _prepare_statistics_csv(payload)

    def write(stream: TextIO) -> None:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    atomic_write(destination, write, overwrite=False)
