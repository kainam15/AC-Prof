"""Layered readers release CSV payloads while retaining verified snapshot rows."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import tracemalloc

import pytest
from test_incremental_results import initialized, receipt
from test_projection_recovery import watch_projection

from acprof.artifacts import file_sha256
from acprof.config import CSV_FIELDS
from acprof.result_csv import ResultValidationError
from acprof.result_layers import LAYER_FILES
from acprof.result_store.initialization import initial_files, initialize_results
from acprof.result_store.protocol import publication_lock
from acprof.result_store.reader import read_snapshot
from acprof.result_store.writer import ResultWriter


def snapshot_store(root, *, count=8, repetitions=4096):
    layout = initialized(root)
    detail = '诊断,"quoted"\n' * repetitions
    with ResultWriter(root) as writer:
        for index in range(count):
            writer.append_window(receipt(root, index))
        rows = [dict(row) for row in writer.rows.values()]
        for row in rows:
            for name in ("process_resource_error", "gpu_energy_fallback_reason",
                         "compute_profile_error_torch_profiler_eager", "compute_profile_error_ncu",
                         "compute_profile_error_nsys", "compute_profile_error_massif"):
                row[name] = detail
            row["workload_contract"] = json.dumps({"文字": detail}, ensure_ascii=False)
        writer.update_rows(rows, reason="fixture diagnostic text")
    return layout, rows


@pytest.mark.parametrize("operation", ["snapshot", "initialize"])
def test_reader_does_not_retain_all_layer_payloads(tmp_path, operation):
    layout, expected_rows = snapshot_store(tmp_path)
    manifest_bytes = layout.result_csv.read_bytes()
    manifest = json.loads(manifest_bytes)
    payload_size = sum(info["committed_bytes"] for info in manifest["layers"].values())
    assert payload_size > 4 * 1024 * 1024
    tracemalloc.start()
    try:
        result = read_snapshot(tmp_path) if operation == "snapshot" else initialize_results(layout)
        retained, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if operation == "snapshot":
        assert result.fields == list(CSV_FIELDS)
        assert result.rows == expected_rows
        assert result.sha256 == hashlib.sha256(manifest_bytes).hexdigest()
        # The returned rows remain resident; temporary CSV buffers must not add
        # another complete copy of all nine layers to that retained result.
        assert peak - retained < payload_size, f"temporary snapshot memory: {peak - retained:,} bytes"
    else:
        assert result == manifest
        largest = max(info["committed_bytes"] for info in manifest["layers"].values())
        assert peak < 2 * largest, f"validation allocated {peak:,} bytes for {largest:,}-byte layers"
    assert layout.result_csv.read_bytes() == manifest_bytes


def artifact_hashes(layout):
    return {path.relative_to(layout.root).as_posix(): file_sha256(path)
            for path in layout.root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("layer", LAYER_FILES)
def test_snapshot_rejects_modified_bytes_before_decoding(tmp_path, layer):
    layout, _ = snapshot_store(tmp_path, count=1, repetitions=1)
    path = layout.layer_csv(layer)
    path.write_bytes(b"\xff" + path.read_bytes()[1:])
    before = artifact_hashes(layout)
    with pytest.raises(ResultValidationError, match="committed prefix"):
        read_snapshot(tmp_path)
    assert artifact_hashes(layout) == before


@pytest.mark.parametrize("operation", ["snapshot", "initialize", "prepare"])
@pytest.mark.parametrize("damage", ["truncated", "missing", "tail"])
def test_all_readers_validate_the_last_layer(tmp_path, operation, damage):
    layout = initialized(tmp_path)
    path = layout.layer_csv("massif")
    if damage == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    elif damage == "missing":
        path.unlink()
    else:
        with path.open("ab") as stream:
            stream.write(b'\xff,"uncommitted tail')
    before = artifact_hashes(layout)
    actions = {"snapshot": lambda: read_snapshot(tmp_path),
               "initialize": lambda: initialize_results(layout), "prepare": lambda: initial_files(layout)}
    if damage == "tail" and operation == "snapshot":
        assert actions[operation]().rows == []
    else:
        error = FileNotFoundError if damage == "missing" else ResultValidationError
        message = {"tail": "uncommitted tail", "truncated": "committed prefix", "missing": "massif"}[damage]
        with pytest.raises(error, match=message):
            actions[operation]()
    assert artifact_hashes(layout) == before


def test_snapshot_ignores_all_pending_tails_without_changing_files(tmp_path):
    layout, expected = snapshot_store(tmp_path, count=2, repetitions=1)
    manifest = layout.result_csv.read_bytes()
    for layer in LAYER_FILES:
        with layout.layer_csv(layer).open("ab") as stream:
            stream.write(b'\xff,"uncommitted tail')
    before = artifact_hashes(layout)
    snapshot = read_snapshot(tmp_path)
    assert snapshot.rows == expected
    assert snapshot.sha256 == hashlib.sha256(manifest).hexdigest()
    assert artifact_hashes(layout) == before


def republish_layer(layout, layer, contents):
    """Keep hashes coherent so malformed CSV reaches the semantic checks."""
    path = layout.layer_csv(layer)
    path.write_bytes(contents)
    manifest = json.loads(layout.result_csv.read_bytes())
    manifest["layers"][layer].update(committed_bytes=len(contents), sha256=hashlib.sha256(contents).hexdigest())
    commit_path = layout.root / manifest["commit"]["path"]
    commit = json.loads(commit_path.read_bytes())
    commit["after"] = {name: value for name, value in manifest.items() if name != "commit"}
    commit_bytes = json.dumps(commit, ensure_ascii=False).encode("utf-8")
    commit_path.write_bytes(commit_bytes)
    manifest["commit"]["sha256"] = hashlib.sha256(commit_bytes).hexdigest()
    layout.result_csv.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")


@pytest.mark.parametrize(("layer", "damage", "message"), [
    ("summary", "header", "header changed"),
    ("performance", "row_count", "invalid performance row count"),
    ("resources", "order", "measurement order mismatch"),
    ("energy", "identity", "measurement identity mismatch"),
    ("network", "duplicate", "duplicate measurement"),
    ("massif", "orphan", "orphan profiler row"),
])
def test_snapshot_preserves_layer_semantic_checks(tmp_path, layer, damage, message):
    layout, _ = snapshot_store(tmp_path, count=2, repetitions=1)
    with layout.layer_csv(layer).open(encoding="utf-8", newline="") as stream:
        header, *rows = list(csv.reader(stream))
    if damage == "header":
        for row in [header, *rows]:
            row[0], row[1] = row[1], row[0]
    elif damage == "row_count":
        rows.pop()
    elif damage == "order":
        rows.reverse()
    elif damage == "identity":
        rows[0][header.index("case_id")] = "different-case"
    elif damage == "duplicate":
        rows[1] = list(rows[0])
    else:
        rows[0][header.index("repeat_idx")] = "99"
    stream = io.StringIO(newline="")
    csv.writer(stream).writerows([header, *rows])
    republish_layer(layout, layer, stream.getvalue().encode("utf-8"))
    before = artifact_hashes(layout)
    with pytest.raises(ResultValidationError, match=message):
        read_snapshot(tmp_path)
    assert artifact_hashes(layout) == before


def test_snapshot_hashes_and_parses_one_read_despite_path_replacement(tmp_path, monkeypatch):
    layout, expected = snapshot_store(tmp_path, count=2, repetitions=1)
    target = layout.layer_csv("resources")
    replacement = tmp_path / "replacement.csv"
    replacement.write_bytes(b"replacement must not enter the verified snapshot")
    manifest = layout.result_csv.read_bytes()
    with monkeypatch.context() as guard:
        observed = watch_projection(guard, target, after_close=lambda _record, _index: replacement.replace(target))
        snapshot = read_snapshot(tmp_path)
    assert snapshot.rows == expected
    assert snapshot.sha256 == hashlib.sha256(manifest).hexdigest()
    assert len(observed.records) == 1
    assert observed.records[0].stream.closed
    assert target.read_bytes() == b"replacement must not enter the verified snapshot"


def test_late_read_failure_closes_stream_and_releases_publication_lock(tmp_path, monkeypatch):
    layout, _ = snapshot_store(tmp_path, count=1, repetitions=1)
    before = artifact_hashes(layout)
    with monkeypatch.context() as guard:
        observed = watch_projection(guard, layout.layer_csv("massif"), error_on="rb")
        with pytest.raises(OSError, match="fixture projection read failed"):
            read_snapshot(tmp_path)
    assert len(observed.records) == 1
    assert observed.records[0].stream.closed
    with publication_lock(layout, exclusive=True, blocking=False):
        assert artifact_hashes(layout) == before
