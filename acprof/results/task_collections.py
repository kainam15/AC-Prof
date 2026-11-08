"""Read frozen task-collection manifests without importing run orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from acprof.artifacts import read_json_object


def read_task_collection(root: Path, *, run_id: str = "") -> tuple[Path, dict[str, Any]]:
    """Select one recorded collection, preserving explicit resume requirements."""
    paths = root / "metadata" / "task_runs"
    candidates = [paths / f"{run_id}.json"] if run_id else sorted(paths.glob("*.json"))
    if len(candidates) != 1:
        raise ValueError("select an existing task collection with --task-run-id; no unique collection to resume")
    data = read_json_object(candidates[0], label="task collection")
    if data.get("schema_version") != 1:
        raise ValueError("unsupported task collection schema")
    return candidates[0], data
