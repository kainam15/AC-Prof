"""Result readers accept complete diagnostic fields emitted by their writers."""
from __future__ import annotations

import csv
import hashlib
import io
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from acprof.config import CSV_FIELDS
from acprof.result_csv import (
    ResultValidationError,
    open_result_text,
    parse_result_stream,
    read_result_csv,
    read_result_csv_snapshot,
)
from acprof.result_layers import export_result_layers, publish_result_layers, read_result_layers
from acprof.result_store.writer import ResultWriter
from tests.results.storage.test_recovery_hash_state import diagnostic_store


@pytest.fixture(autouse=True)
def default_csv_limit():
    previous = csv.field_size_limit(131072)
    try:
        yield
    finally:
        csv.field_size_limit(previous)


def result_file(root: Path, *, size=131073, encoding="utf-8-sig", **changes):
    fragment = '诊断\x00,"quoted"\r\n'
    row = dict.fromkeys(CSV_FIELDS, "nan")
    row.update(cpu_cores="1", mem_cap_gb="4", gpu_mode="off", input_scale="1", warmup="0",
               repeat_idx="0", status="ok", error="", environment_class="native_linux",
               latency_s="0.25", cpu_idle_power_w="10", gpu_idle_power_w="20",
               note=(fragment * (size // len(fragment) + 1))[:size])
    row.update(changes)
    path = root / "source.csv"
    with path.open("w", encoding=encoding, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return path, row


@pytest.mark.parametrize("export", [False, True], ids=["layered", "wide-export"])
def test_writer_diagnostics_round_trip_without_truncation(tmp_path, export):
    layout = diagnostic_store(tmp_path)
    with ResultWriter(tmp_path) as writer:
        expected = list(writer.rows.values())
    path = tmp_path / "export.csv" if export else layout.result_csv
    if export:
        assert export_result_layers(tmp_path, path) == 1
        csv.field_size_limit(131072)
    raw = path.read_bytes()
    snapshot = read_result_csv_snapshot(path)
    assert snapshot.rows == expected
    assert snapshot.sha256 == hashlib.sha256(raw).hexdigest()
    assert len(snapshot.rows[0]["process_resource_error"]) == 196608
    assert len(snapshot.rows[0]["workload_contract"]) == 245770
    assert path.read_bytes() == raw


@pytest.mark.parametrize("size", [131071, 131072, 131073, 524288])
@pytest.mark.parametrize("reader", ["plain", "snapshot", "text-view"])
def test_result_readers_preserve_large_quoted_unicode_fields(tmp_path, size, reader):
    path, expected = result_file(tmp_path, size=size)
    raw = path.read_bytes()
    if reader == "plain":
        fields, rows = read_result_csv(path)
        assert fields == list(expected)
        assert rows == [expected]
    elif reader == "snapshot":
        snapshot = read_result_csv_snapshot(path)
        assert snapshot.sha256 == hashlib.sha256(raw).hexdigest()
        assert snapshot.rows == [expected]
    else:
        with open_result_text(path) as stream:
            assert list(csv.DictReader(stream, strict=True)) == [expected]
    assert path.read_bytes() == raw


def test_historical_layer_publish_and_read_preserve_large_fields(tmp_path):
    path, expected = result_file(tmp_path)
    raw = path.read_bytes()
    manifest = publish_result_layers(path)
    assert manifest["schema_version"] == 2
    csv.field_size_limit(131072)
    _, rows = read_result_layers(tmp_path)
    assert rows == [expected]
    assert path.read_bytes() == raw


@pytest.mark.parametrize("damage", ["unclosed-quote", "duplicate", "invalid-key"])
def test_large_fields_keep_strict_csv_and_measurement_validation(tmp_path, damage):
    path, row = result_file(tmp_path)
    if damage == "unclosed-quote":
        path.write_text('cpu_cores,mem_cap_gb,gpu_mode,input_scale,warmup,repeat_idx,note\n'
                        '1,4,off,1,0,0,"' + 'x' * 200000, encoding="utf-8")
        error, message = csv.Error, "unexpected end of data"
    elif damage == "duplicate":
        with path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=list(row)).writerow(row)
        error, message = ResultValidationError, "duplicate measurement"
    else:
        path, _ = result_file(tmp_path, repeat_idx="invalid")
        error, message = ResultValidationError, "invalid repeat_idx"
    raw = path.read_bytes()
    with pytest.raises(error, match=message):
        read_result_csv_snapshot(path)
    assert path.read_bytes() == raw


def test_concurrent_readers_never_restore_a_smaller_limit(tmp_path):
    path, expected = result_file(tmp_path)
    # Preserve CR/LF inside the field, just as the production file reader does.
    with path.open(encoding="utf-8-sig", newline="") as source:
        text = source.read()
    started = [Event(), Event()]
    proceed = [Event(), Event()]

    def lines(index):
        with io.StringIO(text, newline="") as stream:
            yield next(stream)
            started[index].set()
            assert proceed[index].wait(5), "reader was not released"
            yield from stream

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(parse_result_stream, lines(0), path, None)
        try:
            assert started[0].wait(5), "first reader did not start"
            second = pool.submit(parse_result_stream, lines(1), path, None)
            assert started[1].wait(5), "second reader did not start"
            proceed[0].set()
            assert first.result(timeout=5)[1].rows == [expected]
            proceed[1].set()
            assert second.result(timeout=5)[1].rows == [expected]
            assert csv.field_size_limit() >= len(expected["note"])
        finally:
            for event in proceed:
                event.set()


def test_importing_result_helpers_does_not_change_csv_policy():
    subprocess.run([sys.executable, "-c", "import csv; before = csv.field_size_limit(); "
                    "import acprof.result_csv; assert csv.field_size_limit() == before"],
                   check=True, capture_output=True, text=True, timeout=10)
