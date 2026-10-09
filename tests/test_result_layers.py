"""Layered CSV projections preserve every measurement and isolate profilers."""
import csv
import json

import pytest

from acprof.config import CSV_FIELDS
from acprof.result_csv import ResultValidationError
from acprof.result_layers import (
    LAYER_FILES,
    export_result_layers,
    field_layer,
    layer_fields,
    publish_result_layers,
    read_result_layers,
)


@pytest.fixture
def wide_csv(tmp_path):
    source = tmp_path / "result_all.csv"
    rows = []
    for idx in range(2):
        row = dict.fromkeys(CSV_FIELDS, "nan")
        row.update(
            cpu_cores="2", mem_cap_gb="4", gpu_mode="on",
            input_scale="64", warmup="0", repeat_idx=str(idx),
            environment_class="native_linux", latency_p50_s="0.03",
            gpu_energy_total_j="0.8", packet_total_wire_bytes_per_request="2048",
            container_mem_usage_peak_bytes="4096", status="ok", error="",
            compute_profile_error_ncu="nan",
        )
        if idx == 0:
            row["gpu_kernel_launch_count_per_request_ncu"] = "42"
            row["compute_profile_error_ncu"] = ""
            row["model_logical_mflop_per_request_torch_profiler_eager"] = "15.5"
        rows.append(row)
    fields = [*CSV_FIELDS, "historical_note"]
    rows[0]["historical_note"] = "line one\nline two"
    rows[1]["historical_note"] = ""
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return source, fields, rows


def test_field_ownership_is_total_and_profiler_specific():
    fields = layer_fields(CSV_FIELDS)
    owned = [field for names in fields.values() for field in names if field not in
             {"cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "warmup", "repeat_idx"}]
    assert len(owned) == len(set(owned))
    assert set(owned) == set(CSV_FIELDS) - {
        "cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "warmup", "repeat_idx"}
    assert field_layer("gpu_kernel_launch_count_per_request_ncu") == "ncu"
    assert field_layer("compute_profile_error_nsys") == "nsys"
    assert field_layer("cpu_heap_peak_total_bytes_massif") == "massif"
    assert field_layer("model_logical_mflops_app_torch_profiler_eager") == "torch"


def test_split_roundtrip_and_sparse_profiler_rows(wide_csv, tmp_path):
    source, fields, expected = wide_csv
    manifest = publish_result_layers(source)
    assert set(manifest["layers"]) == {
        "summary", "performance", "resources", "energy", "network", "torch", "ncu"}
    assert not (tmp_path / LAYER_FILES["massif"]).exists()
    assert not (tmp_path / LAYER_FILES["nsys"]).exists()
    with (tmp_path / LAYER_FILES["ncu"]).open(newline="") as f:
        assert len(list(csv.DictReader(f))) == 1
    full_fields, full_rows = read_result_layers(tmp_path)
    assert set(full_fields) == set(fields)
    assert full_rows == [{field: row.get(field, "nan") for field in full_fields} for row in expected]
    export = tmp_path / "export.csv"
    assert export_result_layers(tmp_path, export) == 2
    with export.open(newline="") as f:
        assert list(csv.DictReader(f)) == full_rows
    assert (tmp_path / "result_all.csv").read_bytes() == source.read_bytes()


def test_manifest_hash_detects_modified_layer(wide_csv, tmp_path):
    publish_result_layers(wide_csv[0])
    (tmp_path / LAYER_FILES["energy"]).write_text("malicious\n")
    with pytest.raises(ResultValidationError, match="modified energy"):
        read_result_layers(tmp_path)


def test_manifest_schema_rejects_tampered_field_ownership(wide_csv, tmp_path):
    publish_result_layers(wide_csv[0])
    manifest_path = tmp_path / "result_layers.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["layers"]["energy"]["fields"].append("latency_p50_s")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ResultValidationError, match="energy layer schema"):
        read_result_layers(tmp_path)


def test_migrate_to_separate_directory_preserves_source(wide_csv, tmp_path):
    source, _, _ = wide_csv
    original = source.read_bytes()
    migrated = tmp_path / "migrated"
    publish_result_layers(source, output_dir=migrated)
    assert source.read_bytes() == original
    assert read_result_layers(migrated)[1][0]["historical_note"] == "line one\nline two"


def test_no_profiler_files_without_profiler_evidence(wide_csv, tmp_path):
    source, _, rows = wide_csv
    for row in rows:
        row["gpu_kernel_launch_count_per_request_ncu"] = "nan"
        row["model_logical_mflop_per_request_torch_profiler_eager"] = "nan"
        row["compute_profile_error_ncu"] = "nan"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*CSV_FIELDS, "historical_note"])
        writer.writeheader()
        writer.writerows(rows)
    manifest = publish_result_layers(source)
    assert all(layer not in manifest["layers"] for layer in ("torch", "ncu", "nsys", "massif"))


def test_export_refuses_to_overwrite_a_layer(wide_csv, tmp_path):
    publish_result_layers(wide_csv[0])
    with pytest.raises(ResultValidationError, match="overwrite"):
        export_result_layers(tmp_path, tmp_path / LAYER_FILES["summary"])

def _posthoc_context(wide_csv, tmp_path):
    from types import SimpleNamespace

    from acprof.host.collection_history import empty_collection_history
    from acprof.host.posthoc.storage import create_backup

    source, fields, original_rows = wide_csv
    metadata_path = tmp_path / "static_meta.json"
    metadata_path.write_text('{"revision":"original"}')
    history_path = tmp_path / "collection_history.json"
    context = SimpleNamespace(
        result_dir=tmp_path, result_csv=source, static_meta_path=metadata_path,
        collection_history_path=history_path, collection_history_existed=False,
        csv_encoding="utf-8",
    )
    backup = create_backup(context)
    rows = [dict(row) for row in original_rows]
    rows[0]["gpu_kernel_launch_count_per_request_ncu"] = "84"
    changes = dict(
        fieldnames=fields, rows=rows,
        static_meta={"revision": "new"},
        collection_history=empty_collection_history(),
        backup_dir=backup,
    )
    return context, changes


def test_posthoc_updates_only_changed_layer(wide_csv, tmp_path):
    from acprof.host.posthoc.storage import commit_result_files

    publish_result_layers(wide_csv[0])
    before = {name: (tmp_path / relative).read_bytes()
              for name, relative in LAYER_FILES.items() if (tmp_path / relative).is_file()}
    context, changes = _posthoc_context(wide_csv, tmp_path)
    commit_result_files(context, **changes)
    _, rows = read_result_layers(tmp_path)
    assert rows[0]["gpu_kernel_launch_count_per_request_ncu"] == "84"
    assert (tmp_path / LAYER_FILES["ncu"]).read_bytes() != before["ncu"]
    for name in ("summary", "performance", "resources", "energy", "network", "torch"):
        assert (tmp_path / LAYER_FILES[name]).read_bytes() == before[name]


def test_posthoc_layer_publish_failure_restores_wide_and_layers(wide_csv, tmp_path, monkeypatch):
    import acprof.result_layers as layer_module
    from acprof.host.posthoc.storage import commit_result_files

    source = wide_csv[0]
    publish_result_layers(source)
    original = source.read_bytes()
    original_layers = read_result_layers(tmp_path)
    context, changes = _posthoc_context(wide_csv, tmp_path)
    original_atomic_write = layer_module.atomic_write
    failing_path = tmp_path / LAYER_FILES["ncu"]

    def fail_ncu(destination, write, **kwargs):
        if destination == failing_path:
            raise OSError("simulated layer storage failure")
        return original_atomic_write(destination, write, **kwargs)

    monkeypatch.setattr(layer_module, "atomic_write", fail_ncu)
    with pytest.raises(OSError, match="simulated layer storage failure"):
        commit_result_files(context, **changes)
    assert source.read_bytes() == original
    assert read_result_layers(tmp_path) == original_layers


def test_cli_migration_requires_separate_directory(wide_csv, tmp_path, capsys):
    from acprof.cli.main import main

    source = wide_csv[0]
    original = source.read_bytes()
    with pytest.raises(SystemExit) as failure:
        main(["results", "split", str(source)])
    assert failure.value.code == 2
    migrated = tmp_path / "converted"
    assert main(["results", "split", str(source), "--output-dir", str(migrated)]) == 0
    assert main(["results", "verify", str(migrated)]) == 0
    exported = tmp_path / "reconstructed.csv"
    assert main(["results", "export", str(migrated), str(exported)]) == 0
    assert main(["results", "export", str(migrated), str(exported)]) == 1
    assert source.read_bytes() == original
    assert exported.is_file()


def test_verify_rejects_stale_source_hash(wide_csv, tmp_path):
    publish_result_layers(wide_csv[0])
    with wide_csv[0].open("a") as f:
        f.write("\n")
    with pytest.raises(ResultValidationError, match="source CSV has changed"):
        read_result_layers(tmp_path)


def test_audit_reports_corrupt_layer(wide_csv, tmp_path):
    from acprof.analysis.audit import audit_result

    publish_result_layers(wide_csv[0])
    report = audit_result(tmp_path)
    assert "invalid_layers" not in {item["code"] for item in report["issues"]}
    (tmp_path / LAYER_FILES["network"]).write_text("broken\n")
    report = audit_result(tmp_path)
    assert "invalid_layers" in {item["code"] for item in report["issues"]}
    assert not report["valid"]
