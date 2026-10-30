"""Layer snapshots parse once and recheck content without rebuilding row tables."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from acprof.results import csv as result_csv, layers as result_layers
from tests.results.storage.test_result_row_assembly import row_store


@pytest.mark.parametrize("version", [1, 2])
def test_layer_snapshot_parses_each_payload_once(tmp_path, monkeypatch, version):
    manifest, expected = row_store(tmp_path, version, count=3, width=2, cell='中文,"quoted"\r\nline')
    path = tmp_path / "result_layers.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    parses = Counter()
    original = result_csv.parse_result_stream

    def observe(stream, path, expected, **kwargs):
        parses[Path(path)] += 1
        return original(stream, path, expected, **kwargs)

    monkeypatch.setattr(result_csv, "parse_result_stream", observe)
    snapshot = result_csv.read_result_csv_snapshot(path)
    assert snapshot.rows == expected
    assert snapshot.sha256 == digest
    assert parses == Counter({tmp_path / info["path"]: 1 for info in manifest["layers"].values()})


@pytest.mark.parametrize("version", [1, 2, 3])
def test_final_content_verification_never_parses_csv(tmp_path, monkeypatch, version):
    row_store(tmp_path, version, count=2, width=1)
    path = tmp_path / "result_layers.json"
    snapshot = result_csv.read_result_csv_snapshot(path)

    def reject_parse(*_args, **_kwargs):
        pytest.fail("content verification must not rebuild parsed row tables")

    monkeypatch.setattr(result_csv, "parse_result_stream", reject_parse)
    result_csv.verify_result_source(path, snapshot.sha256)


def after_summary_read(monkeypatch, summary, change):
    original = result_layers.read_result_csv_snapshot
    changed = False

    def read(path, **kwargs):
        nonlocal changed
        snapshot = original(path, **kwargs)
        if Path(path) == summary and not changed:
            changed = True
            change()
        return snapshot

    monkeypatch.setattr(result_layers, "read_result_csv_snapshot", read)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("operation", ["corrupt", "delete"])
def test_snapshot_rechecks_already_parsed_layers(tmp_path, monkeypatch, version, operation):
    manifest, _ = row_store(tmp_path, version, count=2, width=1)
    summary = tmp_path / manifest["layers"]["summary"]["path"]

    def change():
        if operation == "delete":
            summary.unlink()
        else:
            summary.write_bytes(b"damaged after parsing\n")

    after_summary_read(monkeypatch, summary, change)
    with pytest.raises(ValueError, match="changed|modified"):
        result_csv.read_result_csv_snapshot(tmp_path / "result_layers.json")


@pytest.mark.parametrize("version", [1, 2])
def test_manifest_change_during_read_is_rejected_even_with_identical_rows(tmp_path, monkeypatch, version):
    manifest, _ = row_store(tmp_path, version, count=2, width=1)
    path = tmp_path / "result_layers.json"
    summary = tmp_path / manifest["layers"]["summary"]["path"]

    def change_manifest():
        path.write_text(json.dumps({**manifest, "source_note": "changed after parsing began"}))

    after_summary_read(monkeypatch, summary, change_manifest)
    with pytest.raises(ValueError, match="changed|hash"):
        result_csv.read_result_csv_snapshot(path)


@pytest.mark.parametrize("version", [1, 2])
def test_manifest_parse_remains_bound_if_path_is_restored(tmp_path, monkeypatch, version):
    manifest, _ = row_store(tmp_path, version, count=2, width=1)
    path = tmp_path / "result_layers.json"
    before = path.read_bytes()
    original = result_layers.read_json_object

    def read_replacement(candidate, **kwargs):
        assert candidate == path
        path.write_text(json.dumps({**manifest, "source_note": "transient replacement"}))
        try:
            return original(candidate, **kwargs)
        finally:
            path.write_bytes(before)

    monkeypatch.setattr(result_layers, "read_json_object", read_replacement)
    with pytest.raises(ValueError, match="hash"):
        result_csv.read_result_csv_snapshot(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("version", [1, 2, 3])
def test_layer_snapshot_keeps_expected_measurement_validation(tmp_path, version):
    _, rows = row_store(tmp_path, version, count=2, width=1)
    path = tmp_path / "result_layers.json"
    expected = [result_csv.measurement_key(row) for row in rows]
    assert result_csv.read_result_csv_snapshot(path, expected=iter(expected)).rows == rows
    with pytest.raises(ValueError, match="plan mismatch"):
        result_csv.read_result_csv_snapshot(path, expected=expected[:1])


@pytest.mark.parametrize("version", [[], {}, True, "2", 9])
def test_content_verification_rejects_invalid_manifest_versions(tmp_path, version):
    manifest, _ = row_store(tmp_path, 2, count=1, width=1)
    path = tmp_path / "result_layers.json"
    path.write_text(json.dumps({**manifest, "schema_version": version}))
    with pytest.raises(ValueError, match="schema|manifest|snapshot"):
        result_csv.verify_result_source(path, hashlib.sha256(path.read_bytes()).hexdigest())
