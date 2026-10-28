"""Cancellation stops snapshot reads and verification without leaking descriptors."""
import io
from concurrent.futures import CancelledError
from functools import partial
from pathlib import Path
from threading import Event

import pytest
from test_sequential_result_reads import artifact_hashes, snapshot_store

from acprof import result_csv


@pytest.mark.parametrize("operation", ["read", "verify"])
def test_cancelled_raw_reads_are_bounded_and_closed(tmp_path, monkeypatch, operation):
    path = tmp_path / "result.csv"
    content = b"status,note\nok," + b"x" * (2 * 1024 * 1024) + b"\n"
    path.write_bytes(content)
    snapshot = result_csv.read_result_csv_snapshot(path, validate_measurements=False)
    cancelled = Event()
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

        def close(self):
            self.raw.close()
            super().close()

    def open_source(source, *args, **kwargs):
        raw = original_open(source, *args, **kwargs)
        if source == path:
            stream = CancellingReader(raw)
            opened.append(stream)
            return stream
        return raw

    operation_call = (partial(result_csv.read_result_csv_snapshot, path, validate_measurements=False)
                      if operation == "read" else partial(result_csv.verify_result_source, path, snapshot.sha256))
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", open_source)
        with pytest.raises(CancelledError):
            operation_call(cancelled=cancelled.is_set)
    assert 0 < sum(received) <= 256 * 1024
    assert len(opened) == 1
    assert opened[0].closed
    assert opened[0].raw.closed
    assert path.read_bytes() == content


def test_cancellation_stops_rows_already_buffered_by_the_text_reader(tmp_path, monkeypatch):
    path = tmp_path / "result.csv"
    path.write_text("status,note\n" + "ok,small row\n" * 100)
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
        result_csv.read_result_csv_snapshot(path, validate_measurements=False, cancelled=cancelled.is_set)
    assert len(consumed) == 1


def test_cancellation_after_verified_layer_read_discards_snapshot(tmp_path, monkeypatch):
    from acprof.result_store import reader

    layout, _ = snapshot_store(tmp_path, count=1, repetitions=1)
    before = artifact_hashes(layout)
    cancelled = Event()
    original = reader.read_snapshot

    def cancel_after(*args, **kwargs):
        result = original(*args, **kwargs)
        cancelled.set()
        return result

    monkeypatch.setattr(reader, "read_snapshot", cancel_after)
    with pytest.raises(CancelledError):
        result_csv.read_result_csv_snapshot(layout.result_csv, cancelled=cancelled.is_set)
    assert artifact_hashes(layout) == before
