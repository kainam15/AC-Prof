"""Cancelled layered reads stop actual I/O and row work without changing artifacts."""
import hashlib
import io
import json
from concurrent.futures import CancelledError
from functools import partial
from pathlib import Path
from threading import Event

import pytest
from test_result_row_assembly import row_store

from acprof import result_csv, result_layers
from acprof.artifact_layout import ArtifactLayout
from acprof.result_store import reader
from acprof.result_store.protocol import publication_lock


def source_hashes(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def watch_cancelled_read(monkeypatch, target, cancelled):
    original_open = Path.open
    opened, received = [], []

    class CancellingReader(io.RawIOBase):
        def __init__(self, raw):
            super().__init__()
            self.raw = raw

        def readable(self):
            return True

        def readinto(self, buffer):
            size = self.raw.readinto(buffer)
            received.append(size)
            cancelled.set()
            return size

        def read(self, size=-1):
            value = self.raw.read(size)
            received.append(len(value))
            cancelled.set()
            return value

        def close(self):
            self.raw.close()
            super().close()

    def open_source(path, *args, **kwargs):
        raw = original_open(path, *args, **kwargs)
        if path == target:
            stream = CancellingReader(raw)
            opened.append(stream)
            return stream
        return raw

    monkeypatch.setattr(Path, "open", open_source)
    return opened, received


@pytest.mark.parametrize(("version", "target"), [(1, "summary"), (1, "source"), (2, "summary"), (3, "summary")])
@pytest.mark.parametrize("operation", ["read", "verify"])
def test_layer_cancellation_stops_bytes_and_closes_streams(tmp_path, monkeypatch, version, target, operation):
    manifest, _ = row_store(tmp_path, version, count=1, width=1, cell="x" * (2 * 1024 * 1024))
    path = tmp_path / "result_layers.json"
    watched = tmp_path / manifest["layers"]["summary"]["path"]
    if target == "source":
        watched = tmp_path / "imported.csv"
        watched.write_bytes(b"original imported CSV\n" * 100_000)
        manifest.update(source_name=watched.name, source_sha256=hashlib.sha256(watched.read_bytes()).hexdigest())
        path.write_text(json.dumps(manifest))
    before = source_hashes(tmp_path)
    cancelled = Event()
    operation_call = (partial(result_csv.read_result_csv_snapshot, path) if operation == "read" else
                      partial(result_csv.verify_result_source, path, before["result_layers.json"]))
    with monkeypatch.context() as scoped:
        opened, received = watch_cancelled_read(scoped, watched, cancelled)
        with pytest.raises(CancelledError):
            operation_call(cancelled=cancelled.is_set)
    assert 0 < sum(received) <= 256 * 1024
    assert len(opened) == 1
    assert opened[0].closed
    assert opened[0].raw.closed
    if version == 3:
        with publication_lock(ArtifactLayout(tmp_path), exclusive=True, blocking=False):
            pass
    assert source_hashes(tmp_path) == before


@pytest.mark.parametrize("version", [1, 2, 3])
def test_layer_cancellation_stops_buffered_csv_rows(tmp_path, monkeypatch, version):
    row_store(tmp_path, version, count=100, width=1)
    before = source_hashes(tmp_path)
    cancelled = Event()
    original = result_csv.validate_result_rows
    consumed = []

    def observe(fields, rows, source, **kwargs):
        def cancel_after_first():
            for row in rows:
                consumed.append(row)
                cancelled.set()
                yield row

        return original(fields, cancel_after_first(), source, **kwargs)

    monkeypatch.setattr(result_csv, "validate_result_rows", observe)
    with pytest.raises(CancelledError):
        result_csv.read_result_csv_snapshot(tmp_path / "result_layers.json", cancelled=cancelled.is_set)
    assert len(consumed) == 1
    assert source_hashes(tmp_path) == before


@pytest.mark.parametrize("version", [1, 2, 3])
def test_layer_cancellation_stops_final_row_assembly(tmp_path, monkeypatch, version):
    row_store(tmp_path, version, count=100, width=1)
    cancelled = Event()
    popped = []
    original = result_layers.consume_layer_rows

    class CancelAfterPop(dict):
        def pop(self, key):
            value = super().pop(key)
            popped.append(key)
            cancelled.set()
            return value

    def assemble(fields, keys, combined, **kwargs):
        return original(fields, keys, CancelAfterPop(combined), **kwargs)

    monkeypatch.setattr(reader if version == 3 else result_layers, "consume_layer_rows", assemble)
    with pytest.raises(CancelledError):
        result_csv.read_result_csv_snapshot(tmp_path / "result_layers.json", cancelled=cancelled.is_set)
    assert len(popped) == 1


@pytest.mark.parametrize("version", [1, 2, 3])
def test_uncancelled_layer_read_preserves_rows_hashes_and_integrity_checks(tmp_path, version):
    manifest, expected = row_store(tmp_path, version, count=3, width=2, cell='诊断,"quoted"\r\nnext')
    path = tmp_path / "result_layers.json"
    before = source_hashes(tmp_path)
    summary = tmp_path / manifest["layers"]["summary"]["path"]
    if version == 3:
        with summary.open("ab") as stream:
            stream.write(b'\xff,"uncommitted tail')
    snapshot = result_csv.read_result_csv_snapshot(path, cancelled=lambda: False)
    assert snapshot.rows == expected
    assert snapshot.sha256 == before["result_layers.json"]
    result_csv.verify_result_source(path, snapshot.sha256, cancelled=lambda: False)
    # A real corruption remains an integrity failure with cancellation enabled.
    summary.write_bytes(b"X" + summary.read_bytes()[1:])
    with pytest.raises(ValueError, match="modified|prefix"):
        result_csv.verify_result_source(path, snapshot.sha256, cancelled=lambda: False)
    with pytest.raises(ValueError, match="modified|prefix"):
        result_csv.read_result_csv_snapshot(path, cancelled=lambda: False)
