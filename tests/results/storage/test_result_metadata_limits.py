"""Result scans enforce limits at the read boundary, including journal metadata."""
import hashlib
import json
import shutil
from concurrent.futures import CancelledError

import pytest

from acprof.artifacts import MAX_JSON_ARTIFACT_BYTES
from acprof.result_store.protocol import publication_lock
from acprof.result_store.reader import committed_content_digest, read_snapshot
from acprof.results.csv import read_result_csv_snapshot
from acprof.results.layers import read_result_layers
from tests.results.storage.test_committed_digest import make_store
from tests.results.storage.test_json_artifact_reads import observe_reader
from tests.results.storage.test_result_row_assembly import row_store


def pad_file(path, size):
    with path.open("ab") as stream:
        stream.write(b" " * (size - stream.tell()))


def test_duplicate_manifest_read_stops_at_remaining_budget(tmp_path, monkeypatch):
    layout = make_store(tmp_path)
    pad_file(layout.result_csv, 8 * 1024 * 1024)
    budget = [127]
    observed = observe_reader(monkeypatch, layout.result_csv)
    with pytest.raises(ValueError, match="byte budget"):
        committed_content_digest(tmp_path, cancelled=lambda: False, budget=budget)
    assert sum(observed.sizes) == 128
    assert observed.offsets[-1] == 128
    assert budget == [-1]
    assert all(stream.closed for stream in observed.streams)


def test_duplicate_commit_read_shares_manifest_budget(tmp_path, monkeypatch):
    layout = make_store(tmp_path)
    manifest_raw = layout.result_csv.read_bytes()
    manifest = json.loads(manifest_raw)
    commit = layout.contained(manifest["commit"]["path"])
    pad_file(commit, 8 * 1024 * 1024)
    budget = [len(manifest_raw) + 127]
    observed = observe_reader(monkeypatch, commit)
    with pytest.raises(ValueError, match="byte budget"):
        committed_content_digest(tmp_path, cancelled=lambda: False, budget=budget)
    assert sum(observed.sizes) == 128
    assert observed.offsets[-1] == 128
    assert budget == [-1]
    assert all(stream.closed for stream in observed.streams)


@pytest.mark.parametrize("operation", ["snapshot", "duplicate"])
def test_result_manifest_uses_metadata_limit(tmp_path, monkeypatch, operation):
    layout = make_store(tmp_path)
    pad_file(layout.result_csv, 8 * 1024 * 1024)
    observed = observe_reader(monkeypatch, layout.result_csv)
    with pytest.raises(ValueError, match="4 MiB read limit"):
        if operation == "snapshot":
            read_snapshot(tmp_path)
        else:
            committed_content_digest(tmp_path, cancelled=lambda: False, budget=[16 * 1024 * 1024])
    assert sum(observed.sizes) == MAX_JSON_ARTIFACT_BYTES + 1
    assert observed.offsets[-1] == MAX_JSON_ARTIFACT_BYTES + 1
    assert all(stream.closed for stream in observed.streams)


@pytest.mark.parametrize("operation", ["snapshot", "duplicate"])
@pytest.mark.parametrize("target", ["manifest", "commit"])
def test_cancel_during_metadata_read_stops_before_next_chunk(tmp_path, monkeypatch, operation, target):
    layout = make_store(tmp_path)
    manifest = json.loads(layout.result_csv.read_bytes())
    path = layout.result_csv if target == "manifest" else layout.contained(manifest["commit"]["path"])
    pad_file(path, 2 * 1024 * 1024)
    if target == "commit":
        manifest["commit"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        layout.result_csv.write_text(json.dumps(manifest), encoding="utf-8")
    observed = observe_reader(monkeypatch, path)
    cancelled = lambda: bool(observed.sizes)
    with pytest.raises(CancelledError if operation == "snapshot" else InterruptedError):
        if operation == "snapshot":
            read_snapshot(tmp_path, cancelled=cancelled)
        else:
            committed_content_digest(tmp_path, cancelled=cancelled, budget=[8 * 1024 * 1024])
    assert sum(observed.sizes) == 65536
    assert observed.offsets[-1] == 65536
    assert all(stream.closed for stream in observed.streams)
    with publication_lock(layout, exclusive=True, blocking=False):
        pass


@pytest.mark.parametrize("read", [read_result_csv_snapshot, read_result_layers])
def test_public_result_read_cancels_manifest_probe(tmp_path, monkeypatch, read):
    layout = make_store(tmp_path)
    pad_file(layout.result_csv, 2 * 1024 * 1024)
    observed = observe_reader(monkeypatch, layout.result_csv)
    with pytest.raises(CancelledError):
        read(layout.result_csv if read is read_result_csv_snapshot else tmp_path,
             cancelled=lambda: bool(observed.sizes))
    assert sum(observed.sizes) == 65536
    assert observed.offsets[-1] == 65536
    assert all(stream.closed for stream in observed.streams)


def duplicate_catalog(root):
    layout = make_store(root / "original")
    manifest = json.loads(layout.result_csv.read_bytes())
    layout.path("run_state.json").write_text(json.dumps({
        "schema_version": 1, "run_id": manifest["run_id"], "status": "complete"}), encoding="utf-8")
    clone = root / "copy"
    shutil.copytree(layout.root, clone)
    return layout.result_csv, clone / "result_layers.json"


def test_catalog_manifest_probe_obeys_duplicate_budget(tmp_path, monkeypatch):
    from acprof.tui.experiments.catalog import scan_experiments

    paths = duplicate_catalog(tmp_path)
    observed = observe_reader(monkeypatch, paths)
    report = scan_experiments([tmp_path], max_duplicate_bytes=127)
    assert len(report.records) == 1
    assert report.records[0].issues == ("duplicate_evidence_unverified",)
    assert sum(observed.sizes) == 128
    assert all(stream.closed for stream in observed.streams)


def test_catalog_cancellation_during_manifest_probe_is_propagated(tmp_path, monkeypatch):
    from acprof.tui.experiments.catalog import scan_experiments

    paths = duplicate_catalog(tmp_path)
    for path in paths:
        pad_file(path, 2 * 1024 * 1024)
    observed = observe_reader(monkeypatch, paths)
    with pytest.raises(InterruptedError, match="scan cancelled"):
        scan_experiments([tmp_path], cancelled=lambda: bool(observed.sizes))
    assert sum(observed.sizes) == 65536
    assert all(stream.closed for stream in observed.streams)


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_duplicate_hashes_manifest_once_and_keeps_exact_budget(tmp_path, monkeypatch, version):
    from acprof.tui.experiments.catalog import _duplicate_digest, scan_experiments

    manifest, _ = row_store(tmp_path, version, count=1, width=1)
    record = scan_experiments([tmp_path]).records[0]
    path = tmp_path / "result_layers.json"
    raw = path.read_bytes()
    expected = hashlib.sha256(json.dumps({
        "state": record.state, "metadata": record.metadata, "validation": record.validation,
        "failures": record.failures,
    }, sort_keys=True).encode())
    expected.update(raw)
    size = len(raw)
    for layer, info in sorted(manifest["layers"].items()):
        payload = (tmp_path / info["path"]).read_bytes()
        size += len(payload)
        expected.update(layer.encode())
        expected.update(payload)
    budget = [size]
    observed = observe_reader(monkeypatch, path)
    assert _duplicate_digest(record, lambda: False, budget) == expected.hexdigest()
    assert budget == [0]
    assert len(observed.streams) == 1
    assert sum(observed.sizes) == len(raw)


def test_snapshot_hash_includes_utf8_bom(tmp_path):
    layout = make_store(tmp_path)
    before = read_snapshot(tmp_path)
    raw = b"\xef\xbb\xbf" + layout.result_csv.read_bytes()
    layout.result_csv.write_bytes(raw)
    snapshot = read_snapshot(tmp_path)
    assert snapshot.rows == before.rows
    assert snapshot.sha256 == hashlib.sha256(raw).hexdigest()
