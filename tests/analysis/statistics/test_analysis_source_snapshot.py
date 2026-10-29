"""Report consumers share one read while retaining strict audit and change detection."""
import copy
import csv
import hashlib
from pathlib import Path

import pytest

from acprof.analysis import model
from acprof.analysis.audit import audit_result
from acprof.analysis.comparison import load_comparison_snapshot
from acprof.result_csv import ResultValidationError, read_result_csv_snapshot
from tests.results.storage.test_result_row_assembly import row_store


@pytest.mark.parametrize("legacy", ["current", "mode-alias", "missing-identity", "spaced-header"])
def test_analysis_reads_csv_twice_for_shared_parse_and_final_content_check(tmp_path, monkeypatch, legacy):
    row = {"cpu_cores": "2", "mem_cap_gb": "4", "gpu_mode": "off", "input_scale": "32",
           "repeat_idx": "0", "warmup": "0", "status": "ok", "latency_app_p95_s": "0.04",
           "note": ' preserve 中文,"quoted"\r\nsecond line '}
    if legacy == "mode-alias":
        row["gpu_mode"] = " CPU "
    elif legacy == "missing-identity":
        del row["gpu_mode"]
        row["device"] = "CPU"
    elif legacy == "spaced-header":
        row = {" " + name + " ": value for name, value in row.items()}
    path = tmp_path / "result_all.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    before = path.read_bytes()
    modes = []
    original_open = Path.open

    def observe_read(candidate, *args, **kwargs):
        if candidate == path:
            modes.append(args[0] if args else kwargs.get("mode", "r"))
        return original_open(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observe_read)
    data = model.load_analysis([path])
    assert modes == ["rb", "rb"]
    assert data.sources[0]["sha256"] == hashlib.sha256(before).hexdigest()
    assert data.raw_rows[0]["row"] == {key.strip(): value for key, value in row.items()}
    assert data.configs[0]["metrics"]["latency_app_p95_s"]["value"] == .04
    assert data.configs[0]["device"] == "CPU"
    assert path.read_bytes() == before


@pytest.mark.parametrize("version", [1, 2, 3])
def test_layered_analysis_does_not_serialize_a_second_csv(tmp_path, monkeypatch, version):
    _, rows = row_store(tmp_path, version, count=2, width=2, cell='中文,"quoted"\nline')

    def reject_csv_encoding(_text):
        pytest.fail("analysis must consume parsed layer rows without CSV re-encoding")

    monkeypatch.setattr("acprof.result_csv._CsvRowEncoder.write", staticmethod(reject_csv_encoding))
    data = model.load_analysis([tmp_path])
    assert [entry["row"] for entry in data.raw_rows] == rows


@pytest.mark.parametrize("version", [2, 3])
def test_layer_content_recheck_does_not_build_another_complete_row_table(tmp_path, monkeypatch, version):
    _, rows = row_store(tmp_path, version, count=2, width=2)
    identity = model._identity

    def reject_reparse(*_args, **_kwargs):
        pytest.fail("final content verification must hash layers without rebuilding rows")

    def after_parsing(*args, **kwargs):
        result = identity(*args, **kwargs)
        monkeypatch.setattr("acprof.result_layers.read_result_layers", reject_reparse)
        return result

    monkeypatch.setattr(model, "_identity", after_parsing)
    data = model.load_analysis([tmp_path])
    assert [entry["row"] for entry in data.raw_rows] == rows


@pytest.mark.parametrize("version", [1, 2, 3])
def test_layer_change_during_analysis_is_rejected_even_with_unchanged_manifest(tmp_path, monkeypatch, version):
    manifest, _ = row_store(tmp_path, version, count=1, width=1)
    manifest_path = tmp_path / "result_layers.json"
    before = manifest_path.read_bytes()
    layer = tmp_path / manifest["layers"]["summary"]["path"]
    identity = model._identity

    def change_layer(*args, **kwargs):
        result = identity(*args, **kwargs)
        layer.write_bytes(b"corrupted after the source snapshot was parsed\n")
        return result

    monkeypatch.setattr(model, "_identity", change_layer)
    with pytest.raises(ValueError, match="changed during analysis"):
        model.load_analysis([tmp_path])
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize("version", [1, 2, 3])
def test_content_verification_only_ignores_v3_uncommitted_tails(tmp_path, monkeypatch, version):
    manifest, rows = row_store(tmp_path, version, count=1, width=1)
    layer = tmp_path / manifest["layers"]["summary"]["path"]
    identity = model._identity

    def append_pending_tail(*args, **kwargs):
        result = identity(*args, **kwargs)
        with layer.open("ab") as stream:
            stream.write(b"uncommitted partial row")
        return result

    monkeypatch.setattr(model, "_identity", append_pending_tail)
    if version == 3:
        data = model.load_analysis([tmp_path])
        assert [entry["row"] for entry in data.raw_rows] == rows
    else:
        with pytest.raises(ValueError, match="changed during analysis"):
            model.load_analysis([tmp_path])
    assert layer.read_bytes().endswith(b"uncommitted partial row")


@pytest.mark.parametrize("invalid", ["mode-alias", "retired-field", "missing-diagnostic", "duplicate"])
def test_shared_historical_rows_never_bypass_strict_auditing(tmp_path, invalid):
    path = tmp_path / "result_all.csv"
    header = "cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,error"
    row = "2,4,off,32,0,0,ok,"
    if invalid == "mode-alias":
        row = row.replace("off", "CPU")
    elif invalid == "retired-field":
        header += ",energy_total_j"
        row += ",999"
    elif invalid == "missing-diagnostic":
        row = row.replace("ok", "error")
    path.write_text(header + "\n" + row + "\n" + (row + "\n" if invalid == "duplicate" else ""))
    snapshot = read_result_csv_snapshot(path, validate_measurements=False)
    original = copy.deepcopy(snapshot)
    with pytest.raises(ResultValidationError):
        read_result_csv_snapshot(path)
    report = audit_result(path, result_snapshot=snapshot)
    assert not report["valid"]
    assert any(issue["code"] == "invalid_csv" for issue in report["issues"])
    comparison = load_comparison_snapshot(path, result_snapshot=snapshot)
    assert not comparison["valid"]
    assert comparison["conditions"]["actual_workload"] is None
    assert snapshot == original


def test_audit_normalization_does_not_change_shared_source_rows(tmp_path):
    path = tmp_path / "result_all.csv"
    path.write_text("cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,environment_class\n"
                    "2,4,off,32,0,0,ok,historical-environment\n")
    snapshot = read_result_csv_snapshot(path, validate_measurements=False)
    original = copy.deepcopy(snapshot)
    report = audit_result(path, result_snapshot=snapshot)
    assert report["row_environment_classes"] == ["unknown"]
    assert snapshot == original


def test_structural_snapshot_cannot_silently_ignore_expected_measurements(tmp_path):
    with pytest.raises(ValueError, match="expected measurements require"):
        read_result_csv_snapshot(tmp_path / "missing.csv", expected=[], validate_measurements=False)
