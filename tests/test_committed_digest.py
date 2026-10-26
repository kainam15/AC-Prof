"""Duplicate detection verifies committed bytes without retaining whole CSV layers."""
import hashlib
import json
import os
import shutil
import tracemalloc
from pathlib import Path

import pytest

from acprof.artifact_layout import ArtifactLayout
from acprof.result_csv import ResultValidationError
from acprof.result_store.protocol import publication_lock
from acprof.result_store.reader import committed_content_digest
from acprof.result_store.writer import ResultWriter


def make_store(root, *, diagnostic_bytes=0):
    layout = ArtifactLayout.for_new_run(root)
    layout.initialize()
    with ResultWriter(root) as writer:
        writer.update_rows([dict(
            cpu_cores="1", mem_cap_gb="4", gpu_mode="off", input_scale="1", warmup="0", repeat_idx="0",
            run_id=writer.manifest["run_id"], case_id="1c_4g_off", case_attempt_id="a" * 32, window_id="fixture",
            status="error", error="fixture: " + "x" * diagnostic_bytes,
        )], reason="synthetic local digest fixture")
    return layout


def expected_digest(layout):
    raw = layout.result_csv.read_bytes()
    manifest = json.loads(raw)
    expected = hashlib.sha256(raw)
    size = len(raw)
    for layer, info in sorted(manifest["layers"].items()):
        payload = layout.layer_csv(layer).read_bytes()[:info["committed_bytes"]]
        expected.update(layer.encode())
        expected.update(payload)
        size += len(payload)
    return expected.hexdigest(), size


def observe_csv_reads(monkeypatch):
    original = Path.open
    reads = []
    streams = []

    class Reader:
        def __init__(self, path, stream):
            self.path, self.stream = path, stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size=-1):
            result = self.stream.read(size)
            reads.append((self.path, len(result), os.lseek(self.stream.fileno(), 0, os.SEEK_CUR)))
            return result

    def open_file(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        if path.suffix == ".csv" and mode == "rb":
            streams.append(stream)
            return Reader(path, stream)
        return stream

    monkeypatch.setattr(Path, "open", open_file)
    return reads, streams


def test_large_committed_csv_does_not_become_a_large_python_buffer(tmp_path):
    layout = make_store(tmp_path, diagnostic_bytes=16 * 1024 * 1024)
    expected, size = expected_digest(layout)
    budget = [size]
    tracemalloc.start()
    try:
        actual = committed_content_digest(tmp_path, cancelled=lambda: False, budget=budget)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert actual == expected
    assert budget == [0]
    # Includes the existing fixed 4 MiB layout metadata read allocation.
    assert peak < 6 * 1024 * 1024, f"digest allocated {peak:,} bytes"


def test_cancelled_scan_does_not_open_any_artifact(tmp_path, monkeypatch):
    make_store(tmp_path)
    opened = []
    original = Path.open

    def open_file(path, *args, **kwargs):
        opened.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(InterruptedError, match="scan cancelled"):
        committed_content_digest(tmp_path, cancelled=lambda: True, budget=[32 * 1024 * 1024])
    assert opened == []


def test_cancel_during_large_layer_stops_before_next_chunk(tmp_path, monkeypatch):
    layout = make_store(tmp_path, diagnostic_bytes=8 * 1024 * 1024)
    reads, streams = observe_csv_reads(monkeypatch)
    summary = layout.layer_csv("summary")
    with pytest.raises(InterruptedError, match="scan cancelled"):
        committed_content_digest(tmp_path, cancelled=lambda: any(path == summary for path, _, _ in reads),
                                 budget=[32 * 1024 * 1024])
    assert 0 < sum(size for path, size, _ in reads if path == summary) <= 1024 * 1024
    assert all(stream.closed for stream in streams)
    # A cancelled reader must also release the shared publication lock.
    with publication_lock(layout, exclusive=True, blocking=False):
        pass


def test_uncommitted_tail_is_neither_read_nor_hashed(tmp_path, monkeypatch):
    layout = make_store(tmp_path)
    expected, size = expected_digest(layout)
    manifest = json.loads(layout.result_csv.read_bytes())
    for layer in manifest["layers"]:
        with layout.layer_csv(layer).open("ab") as stream:
            stream.write(b"uncommitted tail" * 1000)
    reads, streams = observe_csv_reads(monkeypatch)
    budget = [size]
    assert committed_content_digest(tmp_path, cancelled=lambda: False, budget=budget) == expected
    assert budget == [0]
    assert sum(size for _, size, _ in reads) == sum(info["committed_bytes"] for info in manifest["layers"].values())
    # Check the underlying descriptor, including any hidden buffered prefetch.
    assert {path: offset for path, _, offset in reads} == {
        layout.layer_csv(layer): info["committed_bytes"] for layer, info in manifest["layers"].items()}
    assert all(stream.closed for stream in streams)


@pytest.mark.parametrize("remaining", [0, -1])
def test_exhausted_budget_does_not_read_artifacts(tmp_path, monkeypatch, remaining):
    make_store(tmp_path)
    opened = []
    original = Path.open

    def open_file(path, *args, **kwargs):
        opened.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(ValueError, match="byte budget"):
        committed_content_digest(tmp_path, cancelled=lambda: False, budget=[remaining])
    assert opened == []


@pytest.mark.parametrize("corruption", ["truncate", "change", "missing", "commit"])
def test_corrupt_committed_evidence_still_fails(tmp_path, corruption):
    layout = make_store(tmp_path)
    path = layout.layer_csv("summary")
    if corruption == "truncate":
        path.write_bytes(path.read_bytes()[:-1])
    elif corruption == "change":
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b"fixture:", b"changed:", 1))
    elif corruption == "missing":
        path.unlink()
    else:
        manifest = json.loads(layout.result_csv.read_bytes())
        layout.contained(manifest["commit"]["path"]).write_bytes(b"{}")
    with pytest.raises((ResultValidationError, FileNotFoundError, ValueError)):
        committed_content_digest(tmp_path, cancelled=lambda: False, budget=[32 * 1024 * 1024])


def test_busy_writer_lock_is_not_waited_on(tmp_path):
    layout = make_store(tmp_path)
    with publication_lock(layout, exclusive=True):
        with pytest.raises(BlockingIOError):
            committed_content_digest(tmp_path, cancelled=lambda: False, budget=[32 * 1024 * 1024])


def test_insufficient_prefix_budget_does_not_read_csv(tmp_path, monkeypatch):
    layout = make_store(tmp_path)
    _, size = expected_digest(layout)
    reads, streams = observe_csv_reads(monkeypatch)
    with pytest.raises(ValueError, match="byte budget"):
        committed_content_digest(tmp_path, cancelled=lambda: False, budget=[size - 1])
    assert reads == []
    assert streams == []


def test_invalid_negative_size_cannot_replenish_shared_budget(tmp_path):
    layout = make_store(tmp_path)
    manifest = json.loads(layout.result_csv.read_bytes())
    manifest["layers"]["summary"]["committed_bytes"] = -32 * 1024 * 1024
    layout.result_csv.write_text(json.dumps(manifest))
    budget = [1024 * 1024]
    with pytest.raises(ResultValidationError, match="invalid incremental summary layer"):
        committed_content_digest(tmp_path, cancelled=lambda: False, budget=budget)
    assert budget == [1024 * 1024]


def test_catalog_collapses_incremental_copies_but_rejects_modified_prefix(tmp_path):
    from acprof.tui.experiment_catalog import scan_experiments

    layout = make_store(tmp_path / "original")
    manifest = json.loads(layout.result_csv.read_bytes())
    layout.path("run_state.json").write_text(json.dumps({
        "schema_version": 1, "run_id": manifest["run_id"], "status": "complete"}))
    clone = tmp_path / "copy"
    shutil.copytree(layout.root, clone)
    summary = clone / manifest["layers"]["summary"]["path"]
    with summary.open("ab") as stream:
        stream.write(b"uncommitted tail")
    report = scan_experiments([tmp_path])
    assert len(report.records) == 1
    assert set(report.records[0].aliases) == {layout.root, clone}
    assert report.records[0].issues == ()
    with summary.open("r+b") as stream:
        stream.seek(manifest["layers"]["summary"]["committed_bytes"] - 3)
        stream.write(b"X")
    report = scan_experiments([tmp_path])
    assert report.records[0].issues == ("duplicate_evidence_unverified",)
