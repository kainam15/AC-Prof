"""Cancellation stops snapshot reads and verification without leaking descriptors."""
import csv
import io
from concurrent.futures import CancelledError
from functools import partial
from pathlib import Path
from threading import Event

import pytest

from acprof.results import csv as result_csv
from acprof.results.measurement_plan import RecordedMeasurementOptions, RecordedMeasurementPlan
from tests.results.storage.test_sequential_result_reads import artifact_hashes, snapshot_store


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


@pytest.mark.parametrize("storage", ["flat", "layered"])
def test_cancellation_during_plan_matching_stops_before_next_key(tmp_path, monkeypatch, storage):
    layout, rows = snapshot_store(tmp_path, count=4, repetitions=1)
    path = layout.result_csv
    if storage == "flat":
        path = tmp_path / "result.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    plan = RecordedMeasurementPlan(
        RecordedMeasurementOptions.parse({"cpus": "1", "mems": "4", "gpus": "off", "warmup": 0, "repeat": 4}),
        (64,),
    )
    stop = Event()
    match = RecordedMeasurementPlan.coverage

    class InterruptingKeys(tuple):
        def __iter__(self):
            yield self[0]
            stop.set()
            yield self[1]
            pytest.fail("continued matching windows after cancellation")

    def cancel_during_match(self, actual, *, cancelled=None):
        return match(self, InterruptingKeys(actual), cancelled=cancelled)

    monkeypatch.setattr(RecordedMeasurementPlan, "coverage", cancel_during_match)
    with pytest.raises(CancelledError):
        result_csv.read_result_csv_snapshot(path, expected=plan, cancelled=stop.is_set)
    assert stop.is_set()
