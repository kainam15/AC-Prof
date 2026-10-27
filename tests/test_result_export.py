"""Explicit exports stream complete CSVs without overwriting concurrent outputs."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from test_incremental_results import initialized
from test_result_row_assembly import row_store

from acprof.artifacts import atomic_write, file_sha256
from acprof.result_csv import ResultValidationError
from acprof.result_layers import export_result_layers, read_result_layers


@pytest.mark.parametrize("version", [1, 2, 3])
def test_export_streams_rows_and_preserves_exact_csv_bytes(tmp_path, monkeypatch, version):
    root = tmp_path / "source"
    _, rows = row_store(root, version, count=32, width=4, cell='诊断,"quoted"\r\n第二行')
    fields, expected_rows = read_result_layers(root)
    assert expected_rows == rows
    before = {path: file_sha256(path) for path in root.rglob("*") if path.is_file()}
    expected = io.StringIO(newline="")
    writer = csv.DictWriter(expected, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    expected_text = expected.getvalue()
    destination = tmp_path / "export.csv"
    writes = []

    def observe_write(path, write, **kwargs):
        def observed(stream):
            class ObservedWriter:
                @staticmethod
                def write(text):
                    assert not destination.exists(), "partial output became visible"
                    writes.append(len(text))
                    return stream.write(text)
            write(ObservedWriter())
        atomic_write(path, observed, **kwargs)

    monkeypatch.setattr("acprof.result_layers.atomic_write", observe_write)
    assert export_result_layers(root, destination) == len(rows)
    assert destination.read_bytes() == expected_text.encode("utf-8")
    assert sum(writes) == len(expected_text)
    assert max(writes) < len(expected_text) // 2, "export buffered the complete CSV"
    assert {path: file_sha256(path) for path in before} == before
    assert sorted(tmp_path.iterdir()) == sorted((root, destination))


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_export_rejects_target_created_after_initial_check(tmp_path, monkeypatch, kind):
    root = tmp_path / "source"
    row_store(root, 3, count=1, width=1)
    destination = tmp_path / "export.csv"

    def competing_reader(path):
        result = read_result_layers(path)
        if kind == "file":
            destination.write_bytes(b"concurrent export\n")
            destination.chmod(0o600)
        elif kind == "directory":
            destination.mkdir()
        else:
            destination.symlink_to("missing-target")
        return result

    monkeypatch.setattr("acprof.result_layers.read_result_layers", competing_reader)
    with pytest.raises(ResultValidationError, match="refusing to overwrite existing CSV"):
        export_result_layers(root, destination)
    if kind == "file":
        assert destination.read_bytes() == b"concurrent export\n"
        assert destination.stat().st_mode & 0o777 == 0o600
    elif kind == "directory":
        assert destination.is_dir()
        assert not list(destination.iterdir())
    else:
        assert destination.is_symlink()
        assert destination.readlink() == Path("missing-target")
    assert sorted(tmp_path.iterdir()) == sorted((root, destination))


def test_export_initial_snapshot_keeps_header(tmp_path):
    root = tmp_path / "source"
    initialized(root)
    fields, rows = read_result_layers(root)
    assert rows == []
    destination = tmp_path / "export.csv"
    assert export_result_layers(root, destination) == 0
    with destination.open(encoding="utf-8", newline="") as stream:
        assert list(csv.reader(stream)) == [fields]


def test_export_validation_failure_does_not_create_output(tmp_path):
    root = tmp_path / "source"
    manifest, _ = row_store(root, 3, count=1, width=1)
    last_layer = root / manifest["layers"]["massif"]["path"]
    last_layer.write_bytes(b"corrupt layer")
    destination = tmp_path / "export.csv"
    before = {path: file_sha256(path) for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ResultValidationError, match="committed prefix"):
        export_result_layers(root, destination)
    assert {path: file_sha256(path) for path in before} == before
    assert list(tmp_path.iterdir()) == [root]
