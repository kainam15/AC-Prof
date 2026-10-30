"""Logical CSV views retain parsing semantics without building a complete text copy."""
from __future__ import annotations

import csv
import gc
import io
import traceback
import tracemalloc
import weakref
from concurrent.futures import CancelledError

import pandas as pd
import pytest
from tests.results.storage.test_incremental_results import initialized, receipt
from tests.results.storage.test_result_row_assembly import row_store
from tests.results.storage.test_sequential_result_reads import artifact_hashes, snapshot_store

from acprof.artifacts import file_sha256
from acprof.results.csv import ResultValidationError, open_result_text, read_result_csv
from acprof.results.layers import publish_result_rows
from acprof.result_store.writer import ResultWriter


def test_opening_layered_text_does_not_allocate_a_complete_csv_copy(tmp_path):
    layout, _ = snapshot_store(tmp_path, count=8, repetitions=4096)
    before = artifact_hashes(layout)
    gc.collect()
    tracemalloc.start()
    try:
        parsed = read_result_csv(layout.result_csv)
        _, parse_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del parsed
    gc.collect()
    tracemalloc.start()
    try:
        with open_result_text(layout.result_csv) as stream:
            assert stream.read(0) == ""
            _, view_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Opening a view may retain parsed rows, but must not build another full CSV.
    assert view_peak < parse_peak + 1024 * 1024, (
        f"opening text view allocated {view_peak:,} bytes; parsing alone used {parse_peak:,}")
    assert artifact_hashes(layout) == before


def csv_text(fields, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("mode", ["one-character", "chunks", "lines", "short-lines", "all"])
def test_layered_text_preserves_bytes_across_read_boundaries(tmp_path, version, mode):
    row_store(tmp_path, version, count=3, width=2, cell='文字🧪,"quoted"\r\nnext\rline\x00' * 32)
    fields, rows = read_result_csv(tmp_path / "result_layers.json")
    expected = csv_text(fields, rows)
    before = {path: file_sha256(path) for path in tmp_path.rglob("*") if path.is_file()}
    with open_result_text(tmp_path) as stream:
        assert stream.read(0) == ""
        if mode in {"one-character", "chunks"}:
            size = 1 if mode == "one-character" else 13
            actual = "".join(iter(lambda: stream.read(size), ""))
        elif mode == "short-lines":
            actual = "".join(iter(lambda: stream.readline(5), ""))
        elif mode == "lines":
            actual = "".join(stream)
        else:
            actual = stream.read()
        assert actual.encode("utf-8") == expected.encode("utf-8")
        assert stream.read() == stream.readline() == ""
        assert list(stream) == []
        assert not stream.seekable()
        with pytest.raises(io.UnsupportedOperation):
            stream.seek(0)
    assert stream.closed
    with pytest.raises(ValueError, match="closed"):
        stream.read(0)
    assert {path: file_sha256(path) for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("version", [1, 2, 3])
def test_layered_text_remains_a_strict_dict_reader_input(tmp_path, version):
    row_store(tmp_path, version, count=3, width=2, cell='文字🧪,"quoted"\r\nnext\x00' * 4096)
    fields, expected = read_result_csv(tmp_path / "result_layers.json")
    with open_result_text(tmp_path) as stream:
        reader = csv.DictReader(stream, strict=True)
        assert reader.fieldnames == fields
        assert list(reader) == expected


def test_pandas_keeps_round_trip_numbers_missing_values_and_inference(tmp_path):
    _, rows = row_store(tmp_path, 2, count=3, width=0)
    extra = ["number", "big_integer", "missing", "flag", "leading_zero", "diagnostic"]
    for index, row in enumerate(rows):
        row.update(number="0.10000000000000002", big_integer=str(9007199254740993 + index),
                   missing=["", "N/A", "nan"][index], flag=["True", "False", "True"][index],
                   leading_zero=" 001 ", diagnostic='文字,"quoted"\r\nnext')
    fields = list(rows[0])
    assert all(name in fields for name in extra)
    publish_result_rows(fields, rows, tmp_path)
    fields, rows = read_result_csv(tmp_path / "result_layers.json")
    expected = pd.read_csv(io.StringIO(csv_text(fields, rows), newline=""),
                           skipinitialspace=True, float_precision="round_trip", iterator=False, chunksize=None)
    with open_result_text(tmp_path) as stream:
        actual = pd.read_csv(stream, skipinitialspace=True, float_precision="round_trip", iterator=False, chunksize=None)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual["number"].iloc[0] == float("0.10000000000000002")
    assert actual["big_integer"].tolist() == [9007199254740993, 9007199254740994, 9007199254740995]
    assert actual["missing"].isna().all()


def test_initial_snapshot_streams_only_its_header(tmp_path):
    layout = initialized(tmp_path)
    fields, rows = read_result_csv(layout.result_csv)
    assert rows == []
    with open_result_text(layout.result_csv) as stream:
        assert list(csv.reader(stream)) == [fields]


def test_view_remains_bound_to_verified_rows_after_next_commit(tmp_path):
    layout = initialized(tmp_path)
    with ResultWriter(tmp_path) as writer:
        writer.append_window(receipt(tmp_path, 0))
        _, first = read_result_csv(layout.result_csv)
        with open_result_text(layout.result_csv) as stream:
            writer.append_window(receipt(tmp_path, 1))
            assert list(csv.DictReader(stream, strict=True)) == first
    assert len(read_result_csv(layout.result_csv)[1]) == 2


def test_corrupt_final_layer_is_rejected_before_returning_stream(tmp_path):
    layout = initialized(tmp_path)
    layout.layer_csv("massif").write_bytes(b"corrupted layer")
    before = artifact_hashes(layout)
    with pytest.raises(ResultValidationError, match="committed prefix"):
        open_result_text(layout.result_csv)
    assert artifact_hashes(layout) == before


def tracked_reader(references):
    class TrackedRow(dict):
        pass

    def read(_path):
        rows = [TrackedRow(note=f"{index}:" + "文字" * 8192) for index in range(16)]
        references.extend(weakref.ref(row) for row in rows)
        return ["note"], rows
    return read


def test_consumed_and_cancelled_views_release_private_rows(tmp_path, monkeypatch):
    references = []
    monkeypatch.setattr("acprof.results.csv.read_result_csv", tracked_reader(references))
    stream = open_result_text(tmp_path / "result_layers.json")
    assert all(reference() is not None for reference in references)
    def cancel_while_reading():
        with stream:
            assert stream.read(65536)
            assert 0 < sum(reference() is not None for reference in references) < len(references)
            raise CancelledError()

    with pytest.raises(CancelledError):
        cancel_while_reading()
    assert stream.closed
    gc.collect()
    assert all(reference() is None for reference in references)
    stream.close()


@pytest.mark.parametrize("failure", [OSError("injected encoding failure"), KeyboardInterrupt("cancelled")])
def test_serialization_failure_releases_rows_and_rejects_later_reads(tmp_path, monkeypatch, failure):
    references = []
    monkeypatch.setattr("acprof.results.csv.read_result_csv", tracked_reader(references))
    calls = 0

    def encode(text):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise failure
        return text.encode("utf-8")

    monkeypatch.setattr("acprof.results.csv._CsvRowEncoder.write", staticmethod(encode))
    stream = open_result_text(tmp_path / "result_layers.json")
    with pytest.raises(type(failure)) as caught, stream:
        stream.read()
    assert caught.value is failure
    assert stream.closed
    # csv.DictWriter's traceback retains the failed row, but no queued rows.
    assert references[0]() is not None
    assert all(reference() is None for reference in references[1:])
    traceback.clear_frames(caught.value.__traceback__)
    assert all(reference() is None for reference in references)
    with pytest.raises(ValueError, match="closed"):
        stream.read()


def test_flat_csv_keeps_bom_newline_and_seek_behavior(tmp_path):
    path = tmp_path / "historical.csv"
    content = 'note,value\r\n"文字\r\nline",001\r\n'
    path.write_bytes(content.encode("utf-8-sig"))
    before = path.read_bytes()
    with open_result_text(path) as stream:
        assert stream.read() == content
        stream.seek(0)
        assert list(csv.DictReader(stream)) == [{"note": "文字\r\nline", "value": "001"}]
    assert path.read_bytes() == before


def test_cancelled_summary_does_not_open_a_logical_stream(tmp_path, monkeypatch):
    from acprof.tui.run.diagnostics import summarize_result_csv

    layout, _ = snapshot_store(tmp_path, count=2, repetitions=1)
    before = artifact_hashes(layout)
    def forbidden(_path):
        pytest.fail("cancelled summary opened a logical CSV stream")

    monkeypatch.setattr("acprof.results.csv.open_result_text", forbidden)
    with pytest.raises(CancelledError):
        summarize_result_csv(layout.result_csv, cancelled=lambda: True)
    assert artifact_hashes(layout) == before
