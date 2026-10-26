"""Repair committed CSV projections without retaining or parsing pending tails."""
from __future__ import annotations

import json
import os
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_incremental_results import initialized, receipt
from test_window_sources import encoded_record

from acprof.result_store.writer import ResultWriter


def committed_store(root, *, large_row=False):
    layout = initialized(root)
    path = receipt(root)
    if large_row:
        record = json.loads(path.read_bytes())
        record["row"]["workload_contract"] = json.dumps({"文字": "甲乙丙\n" * 20000}, ensure_ascii=False)
        path.write_bytes(encoded_record(record))
    with ResultWriter(root) as writer:
        writer.append_window(path)
    return layout


def append_sparse_tail(path, size):
    with path.open("ab") as stream:
        stream.truncate(path.stat().st_size + size)


def watch_projection(monkeypatch, target, *, fragment_size=None, after_close=None,
                     error_on=None, deny_write=False):
    observed = SimpleNamespace(records=[], truncations=[])

    class Stream:
        def __init__(self, stream, mode):
            self.record = SimpleNamespace(stream=stream, mode=mode, requests=[], received=0, offsets=[])
            observed.records.append(self.record)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.record.stream.close()
            if after_close is not None:
                after_close(self.record, len(observed.records))

        def read(self, size=-1):
            self.record.requests.append(size)
            if error_on == self.record.mode:
                raise OSError("fixture projection read failed")
            raw = self.record.stream.read(min(size, fragment_size) if fragment_size and size >= 0 else size)
            self.record.received += len(raw)
            self.record.offsets.append(os.lseek(self.fileno(), 0, os.SEEK_CUR))
            return raw

        def truncate(self, size):
            observed.truncations.append(size)
            if error_on == "truncate":
                raise OSError("fixture projection truncate failed")
            return self.record.stream.truncate(size)

        def flush(self):
            return self.record.stream.flush()

        def fileno(self):
            return self.record.stream.fileno()

    def open_file(path: Path, mode: str = "r", *args, **kwargs):
        if path == target and mode == "r+b" and deny_write:
            raise PermissionError("unchanged projection must stay read-only")
        stream = open(path, mode, *args, **kwargs)
        return Stream(stream, mode) if path == target and mode in {"rb", "r+b"} else stream

    monkeypatch.setattr(Path, "open", open_file)
    return observed


def test_recovery_memory_does_not_grow_with_an_uncommitted_tail(tmp_path):
    layout = committed_store(tmp_path)
    target = layout.layer_csv("energy")
    expected = target.read_bytes()
    manifest = layout.result_csv.read_bytes()
    inode = target.stat().st_ino
    append_sparse_tail(target, 32 * 1024 * 1024)
    tracemalloc.start()
    try:
        with ResultWriter(tmp_path) as recovered:
            assert len(recovered.rows) == 1
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024, f"recovery allocated {peak:,} bytes for an uncommitted tail"
    assert target.read_bytes() == expected
    assert target.stat().st_ino == inode
    assert layout.result_csv.read_bytes() == manifest


def test_recovery_reads_only_the_prefix_and_one_tail_byte(tmp_path, monkeypatch):
    layout = committed_store(tmp_path, large_row=True)
    target = layout.layer_csv("summary")
    expected = target.read_bytes()
    assert len(expected) > 2 * 65536
    append_sparse_tail(target, 8 * 1024 * 1024)
    with monkeypatch.context() as guard:
        observed = watch_projection(guard, target, fragment_size=4093)
        with ResultWriter(tmp_path):
            pass
    assert observed.truncations == [len(expected)]
    assert target.read_bytes() == expected
    for record in observed.records:
        assert record.stream.closed
        assert all(0 < size <= 65536 for size in record.requests)
        assert record.received <= len(expected) + 1
        assert max(record.offsets) <= len(expected) + 1, "buffered reads prefetched the uncommitted tail"


@pytest.mark.parametrize("replacement_kind", ["corrupt", "short", "already-correct"])
def test_recovery_rechecks_the_file_opened_for_truncation(tmp_path, monkeypatch, replacement_kind):
    layout = committed_store(tmp_path)
    target = layout.layer_csv("energy")
    expected = target.read_bytes()
    append_sparse_tail(target, 4096)
    replacement = tmp_path / "replacement.csv"
    replacement.write_bytes({"corrupt": b"!" * (len(expected) + 17),
                             "short": expected[:10], "already-correct": expected}[replacement_kind])

    def replace_after_validation(record, count):
        if record.mode == "rb" and count == 1:
            replacement.replace(target)

    with monkeypatch.context() as guard:
        observed = watch_projection(guard, target, after_close=replace_after_validation)
        with ResultWriter(tmp_path):
            pass
    assert not replacement.exists()
    assert target.read_bytes() == expected
    assert observed.truncations == []
    assert all(record.stream.closed for record in observed.records)


def test_complete_read_only_projection_keeps_its_inode_and_permissions(tmp_path, monkeypatch):
    layout = committed_store(tmp_path)
    target = layout.layer_csv("energy")
    expected = target.read_bytes()
    target.chmod(0o444)
    before = target.stat()
    with monkeypatch.context() as guard:
        observed = watch_projection(guard, target, deny_write=True)
        with ResultWriter(tmp_path):
            pass
    after = target.stat()
    assert (after.st_ino, after.st_mode, after.st_mtime_ns) == (before.st_ino, before.st_mode, before.st_mtime_ns)
    assert target.read_bytes() == expected
    assert [record.mode for record in observed.records] == ["rb"]
    assert observed.truncations == []
    assert all(record.stream.closed for record in observed.records)


@pytest.mark.parametrize("damage", ["missing", "empty", "short", "corrupt", "corrupt-tail"])
def test_recovery_rebuilds_damaged_projections_from_unchanged_authority(tmp_path, monkeypatch, damage):
    layout = committed_store(tmp_path)
    target = layout.layer_csv("energy")
    expected = target.read_bytes()
    authority_paths = [layout.result_csv, *layout.path("raw/windows").rglob("*.json"),
                       *layout.path("raw/telemetry").rglob("*.json"),
                       *layout.result_store_path("commits").glob("*.json")]
    authority = {path: path.read_bytes() for path in authority_paths}
    if damage == "missing":
        target.unlink()
    elif damage == "empty":
        target.write_bytes(b"")
    elif damage == "short":
        target.write_bytes(expected[:10])
    else:
        target.write_bytes(b"!" * len(expected))
        if damage == "corrupt-tail":
            append_sparse_tail(target, 8 * 1024 * 1024)
    with monkeypatch.context() as guard:
        observed = watch_projection(guard, target)
        with ResultWriter(tmp_path) as recovered:
            assert next(iter(recovered.rows.values()))["latency_app_s"] == "0.1"
    assert target.read_bytes() == expected
    assert observed.truncations == []
    assert all(path.read_bytes() == raw for path, raw in authority.items())
    for record in observed.records:
        assert record.stream.closed
        assert record.received <= len(expected) + 1
        assert all(offset <= len(expected) + 1 for offset in record.offsets)


@pytest.mark.parametrize("failure", ["rb", "r+b", "truncate", "fsync"])
def test_recovery_errors_keep_pending_journal_and_allow_retry(tmp_path, monkeypatch, failure):
    layout = committed_store(tmp_path)
    target = layout.layer_csv("energy")
    second = receipt(tmp_path, 1)
    manifest = layout.result_csv.read_bytes()
    with ResultWriter(tmp_path) as writer:
        before = {layout.layer_csv(layer): layout.layer_csv(layer).read_bytes()
                  for layer in writer.manifest["layers"]}
        append = writer._append_bytes

        def interrupted_append(layer, raw):
            append(layer, raw)
            if layer == "energy":
                raise OSError("fixture interrupted append")

        with monkeypatch.context() as guard:
            guard.setattr(writer, "_append_bytes", interrupted_append)
            with pytest.raises(OSError, match="fixture interrupted append"):
                writer.append_window(second)
    pending = layout.result_store_path("pending.json")
    journal = pending.read_bytes()
    original_stat = target.stat()
    original_fsync = os.fsync

    def failing_sync(fd):
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) == (original_stat.st_dev, original_stat.st_ino):
            raise OSError("fixture projection fsync failed")
        return original_fsync(fd)

    recovering = ResultWriter(tmp_path)
    with monkeypatch.context() as guard:
        observed = watch_projection(guard, target, error_on=failure)
        if failure == "fsync":
            guard.setattr(os, "fsync", failing_sync)
        with pytest.raises(OSError, match="fixture projection"):
            with recovering:
                pytest.fail("recovery silently accepted an I/O failure")
    assert recovering.lease is None
    assert all(record.stream.closed for record in observed.records)
    assert pending.read_bytes() == journal
    assert layout.result_csv.read_bytes() == manifest
    with ResultWriter(tmp_path) as recovered:
        assert not pending.exists()
        assert all(path.read_bytes() == raw for path, raw in before.items())
        assert recovered.append_window(second)
        assert not recovered.append_window(second)


def test_corrupt_source_is_rejected_before_touching_projection_tail(tmp_path, monkeypatch):
    layout = committed_store(tmp_path)
    target = layout.layer_csv("energy")
    append_sparse_tail(target, 1024 * 1024)
    original_size = target.stat().st_size
    next(layout.path("raw/windows").rglob("*.json")).write_bytes(b"{}")
    with monkeypatch.context() as guard:
        observed = watch_projection(guard, target)
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            with ResultWriter(tmp_path):
                pytest.fail("corrupt immutable authority was accepted")
    assert observed.records == []
    assert observed.truncations == []
    assert target.stat().st_size == original_size
