"""Durable, directly inspectable results between measurement windows."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from acprof.artifact_layout import ArtifactLayout
from acprof.config import CSV_FIELDS
from acprof.result_layers import BASE_LAYERS, LAYER_FILES, layer_fields, read_result_layers
from acprof.result_store.writer import ResultWriter
from acprof.telemetry import window_evidence
from acprof.window_records import (
    WindowPublicationContext,
    commit_window_record,
    prepare_window_telemetry,
)


def test_initialization_publishes_headers_without_measurements(tmp_path):
    layout = ArtifactLayout.for_new_run(tmp_path)
    layout.initialize()
    expected = layer_fields(CSV_FIELDS)
    for layer in BASE_LAYERS:
        path = layout.path(LAYER_FILES[layer])
        assert path.is_file(), f"missing initial {layer} CSV"
        with path.open(newline="") as stream:
            reader = csv.reader(stream)
            assert next(reader) == expected[layer]
            assert list(reader) == []
    for name in ("profiling", "metadata", "raw", "plots", "logs", ".acprof"):
        assert layout.path(name).is_dir()
    fields, rows = read_result_layers(tmp_path)
    assert fields == list(CSV_FIELDS)
    assert rows == []
    manifest = json.loads(layout.result_csv.read_text())
    assert manifest["state"] == "initialized"
    assert manifest["row_count"] == 0


def test_empty_initialization_can_be_retried_without_changing_identity(tmp_path):
    layout = ArtifactLayout.for_new_run(tmp_path)
    layout.initialize()
    before = layout.result_csv.read_bytes()
    layout.initialize()
    assert layout.result_csv.read_bytes() == before


def test_plot_directory_uses_its_own_metadata_and_output_directory(tmp_path, monkeypatch):
    from acprof.cli import plot

    layout = initialized(tmp_path)
    layout.path("static_meta.json").write_text(json.dumps({"schema_version": 7,
        "platform": {"environment": "native_linux", "native": True},
        "collection_tier": "full", "comparability_class": "native_linux"}))
    with ResultWriter(tmp_path) as writer:
        writer.append_window(receipt(tmp_path))
    captured = []
    monkeypatch.setattr(plot.plotting_data, "build_plot_groups", lambda _frame: [])
    monkeypatch.setattr(plot.analysis_latency_report, "write_latency_model_report",
                        lambda frame, meta, path: captured.append((frame, meta, path)))
    plot.main([str(tmp_path)])
    frame, meta, output = captured[0]
    assert list(frame["environment_class"]) == ["native_linux"]
    assert meta["platform"]["environment"] == "native_linux"
    assert output == str(layout.plots_dir)


def receipt(root, index=0, *, attempt="a" * 32, cpu=1):
    layout = ArtifactLayout.discover(root)
    manifest = json.loads(layout.result_csv.read_text())
    case = layout.case("fixture/model", cpu, 4, "off")
    context = WindowPublicationContext(manifest["run_id"], case.case_id, attempt,
                                       str(case.window_records(attempt)), str(case.telemetry(attempt)))
    window = f"case_scale1_r{index}"
    row = {name: "nan" for name in CSV_FIELDS}
    row.update(cpu_cores=str(cpu), mem_cap_gb="4", gpu_mode="off", input_scale="1", warmup="0",
               repeat_idx=str(index), status="ok", error="", environment_class="native_linux",
               latency_app_s=str(0.1 + index),
               workload_contract=json.dumps({"文字": "第一行\n第二行"}, ensure_ascii=False, indent=2))
    row.update(context.identity(window))
    evidence = window_evidence(identity=context.identity(window), context={},
                               collectors={"resource": {"status": "unavailable", "reason": "fixture"}})
    telemetry = prepare_window_telemetry(context, window, evidence)
    commit_window_record(context, window_id=window, row=row,
                         request_record={"record": {"sniff_group_id": window, "latency_app_s": [0.1 + index]}},
                         telemetry_reference=telemetry, projections={})
    return context.record_path(window)


def initialized(root):
    layout = ArtifactLayout.for_new_run(root)
    layout.initialize()
    return layout


def test_windows_append_without_reading_previous_csv_and_are_idempotent(tmp_path, monkeypatch):
    layout = initialized(tmp_path)
    first, second = receipt(tmp_path), receipt(tmp_path, 1)
    with ResultWriter(tmp_path) as writer:
        assert writer.append_window(first)
        before = {name: layout.layer_csv(name).read_bytes() for name in BASE_LAYERS}
        manifest_before = layout.result_csv.read_bytes()
        assert not writer.append_window(first)
        assert layout.result_csv.read_bytes() == manifest_before
        original = Path.read_bytes
        def deny_old_csv_read(path):
            assert path.suffix != ".csv", "append reread an old CSV"
            return original(path)
        with monkeypatch.context() as guard:
            guard.setattr(Path, "read_bytes", deny_old_csv_read)
            writer.append_window(second)
        for name in BASE_LAYERS:
            assert layout.layer_csv(name).read_bytes().startswith(before[name])
    _, rows = read_result_layers(tmp_path)
    assert [row["latency_app_s"] for row in rows] == ["0.1", "1.1"]
    assert [row["window_id"] for row in rows] == ["case_scale1_r0", "case_scale1_r1"]


def test_legacy_snapshot_import_cannot_replace_a_live_commit(tmp_path):
    from acprof.result_csv import ResultValidationError
    from acprof.result_layers import publish_result_rows

    layout = initialized(tmp_path)
    with ResultWriter(tmp_path) as writer:
        writer.append_window(receipt(tmp_path))
    before = layout.result_csv.read_bytes()
    fields, rows = read_result_layers(tmp_path)
    with pytest.raises(ResultValidationError, match="incremental"):
        publish_result_rows(fields, rows, tmp_path)
    assert layout.result_csv.read_bytes() == before


def test_snapshot_version_change_with_identical_rows_is_detected(tmp_path, monkeypatch):
    from acprof import result_csv

    layout = initialized(tmp_path)
    with ResultWriter(tmp_path) as writer:
        writer.append_window(receipt(tmp_path))
    snapshot = result_csv.read_result_csv_snapshot(layout.result_csv)
    original = result_csv._file_sha256
    def commit_after_hash(path):
        checksum = original(path)
        with ResultWriter(tmp_path) as writer:
            writer.freeze()
        return checksum
    monkeypatch.setattr(result_csv, "_file_sha256", commit_after_hash)
    assert not result_csv.result_csv_snapshot_unchanged(snapshot)


@pytest.mark.parametrize("failed_layer", BASE_LAYERS)
def test_crash_at_each_module_exposes_only_commit_and_replays_once(tmp_path, monkeypatch, failed_layer):
    layout = initialized(tmp_path)
    path = receipt(tmp_path)
    old = {name: layout.layer_csv(name).read_bytes() for name in BASE_LAYERS}
    with ResultWriter(tmp_path) as writer:
        original = writer._append_bytes
        def crash(layer, raw):
            original(layer, raw)
            if layer == failed_layer:
                raise OSError("simulated disk failure")
        monkeypatch.setattr(writer, "_append_bytes", crash)
        with pytest.raises(OSError, match="simulated disk failure"):
            writer.append_window(path)
        with pytest.raises(ValueError, match="requires recovery"):
            writer.append_window(path)
    assert read_result_layers(tmp_path)[1] == []
    with ResultWriter(tmp_path) as recovered:
        assert all(layout.layer_csv(name).read_bytes() == raw for name, raw in old.items())
        assert recovered.append_window(path)
        assert not recovered.append_window(path)
    assert len(read_result_layers(tmp_path)[1]) == 1


def test_recovery_repairs_projection_but_rejects_corrupt_authority(tmp_path):
    layout = initialized(tmp_path)
    path = receipt(tmp_path)
    with ResultWriter(tmp_path) as writer:
        writer.append_window(path)
    expected = layout.layer_csv("energy").read_bytes()
    layout.layer_csv("energy").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="committed prefix"):
        read_result_layers(tmp_path)
    with ResultWriter(tmp_path):
        assert layout.layer_csv("energy").read_bytes() == expected
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        with ResultWriter(tmp_path):
            pass


def test_postprocessing_replaces_only_affected_modules(tmp_path):
    layout = initialized(tmp_path)
    with ResultWriter(tmp_path) as writer:
        writer.append_window(receipt(tmp_path))
        before = {name: (layout.layer_csv(name).stat().st_ino, layout.layer_csv(name).read_bytes()) for name in BASE_LAYERS}
        row = dict(next(iter(writer.rows.values())))
        row.update(latency_s="0.2", packet_total_wire_bytes_per_request="1200")
        writer.update_rows([row], reason="packet parser fixture")
        for name in ("summary", "resources", "energy"):
            assert (layout.layer_csv(name).stat().st_ino, layout.layer_csv(name).read_bytes()) == before[name]
    _, rows = read_result_layers(tmp_path)
    assert rows[0]["latency_s"] == "0.2"
    assert rows[0]["packet_total_wire_bytes_per_request"] == "1200"
    with ResultWriter(tmp_path) as recovered:
        assert recovered.rows == {tuple(["1", "4", "off", "1", "0", "0"]): rows[0]}


def test_duplicate_plan_window_in_new_attempt_is_rejected(tmp_path):
    initialized(tmp_path)
    first = receipt(tmp_path)
    second = receipt(tmp_path, attempt="b" * 32)
    with ResultWriter(tmp_path) as writer:
        writer.append_window(first)
        with pytest.raises(ValueError, match="duplicate planned measurement"):
            writer.append_window(second)


def test_unknown_derivation_columns_are_rejected_without_commit(tmp_path):
    layout = initialized(tmp_path)
    with ResultWriter(tmp_path) as writer:
        writer.append_window(receipt(tmp_path))
        before = layout.result_csv.read_bytes()
        row = dict(next(iter(writer.rows.values())), invented_metric="12")
        with pytest.raises(ValueError, match="unknown.*field"):
            writer.update_rows([row], reason="invalid producer")
        assert layout.result_csv.read_bytes() == before


@pytest.mark.parametrize("committed", [False, True])
def test_replacement_failure_recovers_whole_version(tmp_path, monkeypatch, committed):
    from acprof.result_store import writer as writer_module
    layout = initialized(tmp_path)
    with ResultWriter(tmp_path) as writer:
        writer.append_window(receipt(tmp_path))
        row = dict(next(iter(writer.rows.values())), latency_s="0.2", packet_total_wire_bytes_per_request="1200")
        original = writer_module.atomic_write_json if committed else writer_module.write_bytes
        def fail(path, payload):
            original(path, payload)
            if path == (layout.result_csv if committed else layout.layer_csv("network")):
                raise OSError("power loss")
        with monkeypatch.context() as guard:
            guard.setattr(writer_module, "atomic_write_json" if committed else "write_bytes", fail)
            with pytest.raises(OSError, match="power loss"):
                writer.update_rows([row], reason="packet derivation")
    if committed:
        assert read_result_layers(tmp_path)[1][0]["latency_s"] == "0.2"
    else:
        with pytest.raises(ValueError, match="requires recovery"):
            read_result_layers(tmp_path)
    with ResultWriter(tmp_path):
        rows = read_result_layers(tmp_path)[1]
        assert rows[0]["latency_s"] == ("0.2" if committed else "nan")
        assert rows[0]["packet_total_wire_bytes_per_request"] == ("1200" if committed else "nan")


def test_frozen_snapshot_is_preserved_after_postprocessing(tmp_path):
    from acprof.result_store.protocol import read_reference
    layout = initialized(tmp_path)
    with ResultWriter(tmp_path) as writer:
        writer.append_window(receipt(tmp_path))
        snapshot = writer.freeze()
        manifest = read_reference(layout, writer.manifest["snapshot"])
        assert snapshot == manifest
        old = {info["path"]: layout.contained(info["path"]).read_bytes() for info in snapshot["layers"].values()}
        row = dict(next(iter(writer.rows.values())), latency_s="0.2")
        writer.update_rows([row], reason="late packet result")
        assert writer.manifest["state"] == "processing"
        assert "snapshot" not in writer.manifest
        writer.freeze()
        assert all(layout.contained(name).read_bytes() == raw for name, raw in old.items())
    assert read_result_layers(tmp_path)[1][0]["latency_s"] == "0.2"


def test_multiple_resource_configs_keep_all_base_orders_and_empty_profilers(tmp_path):
    layout = initialized(tmp_path)
    with ResultWriter(tmp_path) as writer:
        for cpu, index in [(2, 0), (2, 1), (1, 0)]:
            writer.append_window(receipt(tmp_path, index, cpu=cpu))
    for layer in BASE_LAYERS:
        with layout.layer_csv(layer).open(newline="") as stream:
            assert [(row["cpu_cores"], row["repeat_idx"]) for row in csv.DictReader(stream)] == [
                ("2", "0"), ("2", "1"), ("1", "0")]
    for layer in ("torch", "ncu", "nsys", "massif"):
        with layout.layer_csv(layer).open(newline="") as stream:
            assert list(csv.DictReader(stream)) == []


def test_single_writer_lease_rejects_concurrent_publication(tmp_path):
    initialized(tmp_path)
    with ResultWriter(tmp_path) as writer:
        with pytest.raises(BlockingIOError), ResultWriter(tmp_path):
            pytest.fail("second writer acquired the experiment")
        writer.append_window(receipt(tmp_path))
    assert len(read_result_layers(tmp_path)[1]) == 1


def test_file_sync_failure_does_not_advance_visible_commit(tmp_path, monkeypatch):
    import os
    layout = initialized(tmp_path)
    source = receipt(tmp_path)
    original = os.fsync
    def fail(fd):
        if Path(f"/proc/self/fd/{fd}").resolve() == layout.layer_csv("energy"):
            raise OSError("simulated fsync EIO")
        return original(fd)
    with ResultWriter(tmp_path) as writer, monkeypatch.context() as guard:
        guard.setattr(os, "fsync", fail)
        with pytest.raises(OSError, match="fsync EIO"):
            writer.append_window(source)
    assert read_result_layers(tmp_path)[1] == []
    with ResultWriter(tmp_path) as writer:
        writer.append_window(source)
    rows = read_result_layers(tmp_path)[1]
    assert len(rows) == 1
    assert json.loads(rows[0]["workload_contract"]) == {"文字": "第一行\n第二行"}


@pytest.mark.parametrize("after_head", [False, True])
def test_sigkill_recovery_never_duplicates_a_window(tmp_path, after_head):
    import subprocess
    import sys
    initialized(tmp_path)
    source = receipt(tmp_path)
    script = '''
import os, signal, sys
from pathlib import Path
from acprof.result_store import writer as module
from acprof.result_store.writer import ResultWriter
with ResultWriter(sys.argv[1]) as writer:
    if sys.argv[3] == "True":
        original = module.atomic_write_json
        def crash(path, payload):
            original(path, payload)
            if path == writer.layout.result_csv:
                os.kill(os.getpid(), signal.SIGKILL)
        module.atomic_write_json = crash
    else:
        original = writer._append_bytes
        def crash(layer, raw):
            original(layer, raw)
            if layer == "energy":
                os.kill(os.getpid(), signal.SIGKILL)
        writer._append_bytes = crash
    writer.append_window(Path(sys.argv[2]))
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(source), str(after_head)],
                            capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == -9, result.stderr
    assert len(read_result_layers(tmp_path)[1]) == int(after_head)
    with ResultWriter(tmp_path) as writer:
        assert writer.append_window(source) == (not after_head)
        assert not writer.append_window(source)
    assert len(read_result_layers(tmp_path)[1]) == 1


@pytest.mark.parametrize("committed", [False, True])
def test_posthoc_metadata_tracks_committed_patch_after_failure(tmp_path, monkeypatch, committed):
    from types import SimpleNamespace

    from acprof.host.posthoc.incremental import commit_incremental
    from acprof.result_store.protocol import write_bytes
    layout = initialized(tmp_path)
    context = SimpleNamespace(result_dir=tmp_path, static_meta_path=layout.path("static_meta.json"),
                              collection_history_path=layout.path("collection_history.json"))
    write_bytes(context.static_meta_path, b'{"prior":true}\r\n')
    with ResultWriter(tmp_path) as writer:
        writer.append_window(receipt(tmp_path))
        writer.freeze()
    rows = read_result_layers(tmp_path)[1]
    rows[0]["latency_s"] = "0.5"
    original = ResultWriter.update_rows
    def crash(writer, *args, **kwargs):
        if committed:
            original(writer, *args, **kwargs)
        raise OSError("posthoc interrupted")
    with monkeypatch.context() as guard:
        guard.setattr(ResultWriter, "update_rows", crash)
        with pytest.raises(OSError, match="posthoc interrupted"):
            commit_incremental(context, rows=rows, static_meta={"new": True},
                               collection_history={"schema_version": 1})
    if committed:
        assert json.loads(context.static_meta_path.read_bytes()) == {"new": True}
        assert read_result_layers(tmp_path)[1][0]["latency_s"] == "0.5"
        assert context.collection_history_path.exists()
    else:
        assert context.static_meta_path.read_bytes() == b'{"prior":true}\r\n'
        assert read_result_layers(tmp_path)[1][0]["latency_s"] == "nan"
        assert not context.collection_history_path.exists()
