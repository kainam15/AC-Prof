"""Task-collection readers retain selection, schema and frozen-file behavior."""
from __future__ import annotations

import json

import pytest

from acprof.results.task_collections import read_task_collection


def write_manifest(root, name, payload):
    path = root / "metadata" / "task_runs" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("run_id", ["", "frozen"])
def test_reads_exact_frozen_collection_without_rewriting(tmp_path, run_id):
    expected = {"schema_version": 1, "task_snapshot": {"model_id": "fixture/model"}, "status": "failed"}
    path = write_manifest(tmp_path, "frozen", expected)
    original = path.read_bytes()

    actual_path, actual = read_task_collection(tmp_path, run_id=run_id)

    assert actual_path == path
    assert actual == expected
    assert path.read_bytes() == original


@pytest.mark.parametrize("count", [0, 2])
def test_resume_requires_unambiguous_selection(tmp_path, count):
    for index in range(count):
        write_manifest(tmp_path, str(index), {"schema_version": 1})
    with pytest.raises(ValueError, match="no unique collection to resume"):
        read_task_collection(tmp_path)


def test_explicit_selection_retains_other_collections(tmp_path):
    selected = write_manifest(tmp_path, "chosen", {"schema_version": 1, "status": "failed"})
    other = write_manifest(tmp_path, "other", {"schema_version": 2})
    original = other.read_bytes()

    path, data = read_task_collection(tmp_path, run_id="chosen")

    assert path == selected
    assert data["status"] == "failed"
    assert other.read_bytes() == original


def test_missing_explicit_collection_and_unknown_schema_fail(tmp_path):
    with pytest.raises(OSError, match="missing"):
        read_task_collection(tmp_path, run_id="missing")
    write_manifest(tmp_path, "newer", {"schema_version": 2})
    with pytest.raises(ValueError, match="unsupported task collection schema"):
        read_task_collection(tmp_path, run_id="newer")
