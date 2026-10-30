"""Committed layer readers bound binary reads without weakening validation."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest
from tests.results.storage.test_projection_recovery import watch_projection
from tests.results.storage.test_sequential_result_reads import snapshot_store

from acprof.results.csv import KEY_FIELDS, ResultValidationError
from acprof.result_store import protocol
from acprof.result_store.initialization import initialize_results
from acprof.result_store.reader import read_snapshot


@pytest.mark.parametrize("operation", ["snapshot", "initialize"])
@pytest.mark.parametrize("fragment_size", [None, 4093])
def test_large_committed_prefix_is_read_once_in_bounded_chunks(tmp_path, monkeypatch, operation, fragment_size):
    layout, expected_rows = snapshot_store(tmp_path, count=2, repetitions=8192)
    original_manifest = layout.result_csv.read_bytes()
    manifest = json.loads(original_manifest)
    target = layout.layer_csv("summary")
    prefix_size = manifest["layers"]["summary"]["committed_bytes"]
    assert prefix_size > 256 * 1024
    if operation == "snapshot":
        with target.open("ab") as stream:
            stream.write(b'\xff,"uncommitted tail')
    with monkeypatch.context() as guard:
        observed = watch_projection(guard, target, fragment_size=fragment_size)
        if operation == "snapshot":
            snapshot = read_snapshot(tmp_path)
            assert snapshot.rows == expected_rows
        else:
            assert initialize_results(layout) == manifest
    assert len(observed.records) == 1
    record = observed.records[0]
    assert record.stream.closed
    assert record.received == prefix_size
    assert all(0 < size <= 256 * 1024 for size in record.requests), record.requests
    assert max(record.offsets) <= prefix_size, "binary buffering read beyond the committed prefix"
    assert layout.result_csv.read_bytes() == original_manifest


def layer_fixture(tmp_path, *, detail="text"):
    fields = [*KEY_FIELDS, "environment_class", "error"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    rows = []
    for index in range(2):
        row = dict(zip(KEY_FIELDS, ("2", "8", "off", "1", "0", str(index)), strict=True))
        row.update(environment_class="native_linux", error=detail)
        writer.writerow(row)
        rows.append(row)
    raw = stream.getvalue().encode("utf-8")
    path = tmp_path / "layer.csv"
    path.write_bytes(raw)
    return path, {"fields": fields, "committed_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}, rows


def test_premature_eof_cannot_accept_rows_before_a_later_hash_drain(tmp_path, monkeypatch):
    path, info, _ = layer_fixture(tmp_path)
    payload = path.read_bytes()
    first_row_end = payload.index(b"\n", payload.index(b"\n") + 1) + 1

    class PrematureEOF(io.BytesIO):
        calls = 0

        def readinto(self, buffer):
            self.calls += 1
            if self.calls == 2:
                return 0
            view = memoryview(buffer)
            limit = first_row_end if self.calls == 1 else len(view)
            return super().readinto(view[:limit])

    raw = PrematureEOF(payload)
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: raw)
    with pytest.raises(ResultValidationError, match="committed prefix"):
        protocol.read_layer_rows(path, info)
    assert raw.closed


@pytest.mark.parametrize("damage", ["utf8", "header", "csv"])
@pytest.mark.parametrize("matching_digest", [False, True])
def test_hash_integrity_precedes_parsing_errors(tmp_path, monkeypatch, damage, matching_digest):
    path, info, _ = layer_fixture(tmp_path, detail='诊断,"quoted"\r\n' * 16384)
    payload = path.read_bytes()
    if damage == "utf8":
        damaged = b"\xff" + payload[1:]
        parse_error, message = UnicodeDecodeError, "decode"
    elif damage == "header":
        damaged = payload.replace(b"mem_cap_gb", b"cpu_cores", 1)
        parse_error, message = ResultValidationError, "duplicate CSV columns"
    else:
        header_end = payload.index(b"\n") + 1
        damaged = payload[:header_end] + b'"unterminated\n' + b"x" * len(payload)
        parse_error, message = csv.Error, "unexpected end"
    path.write_bytes(damaged)
    info["committed_bytes"] = len(damaged)
    if matching_digest:
        info["sha256"] = hashlib.sha256(damaged).hexdigest()
    else:
        parse_error, message = ResultValidationError, "committed prefix"
    with monkeypatch.context() as guard:
        observed = watch_projection(guard, path, fragment_size=4093)
        with pytest.raises(parse_error, match=message):
            protocol.read_layer_rows(path, info)
    assert len(observed.records) == 1
    record = observed.records[0]
    assert record.received == len(damaged)
    assert record.stream.closed
    assert all(0 < size <= 256 * 1024 for size in record.requests)
    assert path.read_bytes() == damaged


@pytest.mark.parametrize("failure", [None, OSError("read failed"), KeyboardInterrupt("cancelled read")])
def test_read_failure_propagates_and_closes_the_same_stream(tmp_path, monkeypatch, failure):
    path, info, _ = layer_fixture(tmp_path)

    class FailingRaw(io.BytesIO):
        calls = 0

        def readinto(self, buffer):
            self.calls += 1
            if self.calls == 2:
                if failure is None:
                    return None
                raise failure
            return super().readinto(memoryview(buffer)[:7])

    raw = FailingRaw(path.read_bytes())
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: raw)
    expected = BlockingIOError if failure is None else type(failure)
    with pytest.raises(expected) as caught:
        protocol.read_layer_rows(path, info)
    if failure is not None:
        assert caught.value is failure
    assert raw.closed


def test_exact_validation_reads_only_one_tail_byte(tmp_path, monkeypatch):
    layout, _ = snapshot_store(tmp_path, count=1, repetitions=1)
    target = layout.layer_csv("summary")
    original = target.read_bytes()
    pending = b'\xff,"uncommitted tail' * 4096
    target.write_bytes(original + pending)
    with monkeypatch.context() as guard:
        observed = watch_projection(guard, target, fragment_size=17)
        with pytest.raises(ResultValidationError, match="uncommitted tail"):
            initialize_results(layout)
    assert len(observed.records) == 1
    record = observed.records[0]
    assert record.received == len(original) + 1
    assert max(record.offsets) == len(original) + 1
    assert record.stream.closed
    assert target.read_bytes() == original + pending


def test_exact_validation_does_not_treat_a_blocked_tail_check_as_eof(tmp_path, monkeypatch):
    layout, _ = snapshot_store(tmp_path, count=1, repetitions=1)
    target = layout.layer_csv("summary")

    class BlockedTail(io.BytesIO):
        def read(self, size=-1):
            return None

    raw = BlockedTail(target.read_bytes())

    def open_file(path: Path, *args, **kwargs):
        return raw if path == target else open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(BlockingIOError, match="tail check would block"):
        initialize_results(layout)
    assert raw.closed
