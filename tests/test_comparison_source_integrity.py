"""Independent comparisons reject evidence changes before publishing statistics."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from independent_comparison_fixtures import IndependentComparisonFixture

from acprof.analysis.independent_comparison import compare_experiments
from acprof.artifact_layout import ArtifactLayout
from acprof.artifacts import read_json_object
from acprof.hardware_conditions import conditions_path
from acprof.quality import combine_quality
from acprof.result_store.writer import ResultWriter
from acprof.telemetry import window_evidence
from acprof.window_records import (
    WindowPublicationContext,
    commit_window_record,
    prepare_window_telemetry,
)


@pytest.fixture
def pair(request, tmp_path):
    fixture = IndependentComparisonFixture()
    fixture.build(request, tmp_path)
    left = fixture.replicate("left", 0, [2])
    right = fixture.replicate("right", 0, [3])
    return left, right


def compare(pair):
    left, right = pair
    return compare_experiments([left], [right], metrics=["latency_app_s"], resamples=10)


def incremental_copy(source):
    layout = ArtifactLayout.for_new_run(source.with_name(source.name + "-incremental"))
    layout.initialize()
    for path in source.glob("*.json"):
        if path.name != "result_layers.json":
            target = conditions_path(layout) if path.name == "hardware_conditions.json" else layout.path(path.name)
            shutil.copyfile(path, target)
    manifest = read_json_object(layout.result_csv)
    state_path = layout.path("run_state.json")
    state = read_json_object(state_path)
    state_path.write_text(json.dumps({**state, "run_id": manifest["run_id"]}))
    with (source / "result_all.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    case = layout.case("fixture/model", 1, 4, "off")
    attempt, window = "a" * 32, "case_scale2_r0"
    context = WindowPublicationContext(manifest["run_id"], case.case_id, attempt,
                                       str(case.window_records(attempt)), str(case.telemetry(attempt)))
    row.update(context.identity(window))
    evidence = window_evidence(identity=context.identity(window), context={},
                               collectors={"resource": {"status": "unavailable", "reason": "fixture"}})
    telemetry = prepare_window_telemetry(context, window, evidence)
    commit_window_record(context, window_id=window, row=row,
                         request_record={"record": {"sniff_group_id": window,
                                                    "latency_app_s": [float(row["latency_app_s"])] * 3}},
                         telemetry_reference=telemetry, projections={})
    with ResultWriter(layout.root) as writer:
        writer.append_window(context.record_path(window))
    return layout.root


def legacy_snapshot(source):
    path = source / "result_layers.json"
    manifest = read_json_object(path)
    for record in manifest["layers"].values():
        original = source / record["path"]
        shutil.copyfile(original, source / original.name)
        record["path"] = original.name
    manifest.update(schema_version=1, source_name="result_all.csv",
                    source_sha256=hashlib.sha256((source / "result_all.csv").read_bytes()).hexdigest())
    manifest.pop("generation")
    path.write_text(json.dumps(manifest))
    return source


def change_during_aggregation(monkeypatch, change):
    changed = False

    def aggregate(records):
        nonlocal changed
        if not changed:
            changed = True
            change()
        return combine_quality(records)

    # Quality aggregation begins after the input rows and condition evidence are read.
    monkeypatch.setattr("acprof.analysis.independent_comparison.combine_quality", aggregate)


def test_state_identity_is_captured_before_its_first_read(pair, monkeypatch):
    state_path = pair[0] / "run_state.json"

    def change_after_state_read(path, **kwargs):
        value = read_json_object(path, **kwargs)
        if Path(path) == state_path:
            state_path.write_text(json.dumps({**value, "run_id": "changed-after-read"}))
        return value

    monkeypatch.setattr("acprof.analysis.independent_comparison.read_json_object", change_after_state_read)
    with pytest.raises(ValueError, match="experiment changed.*run_state"):
        compare(pair)


@pytest.mark.parametrize("name", ["model_resolution.json", "runtime_failures.json"])
@pytest.mark.parametrize("operation", ["create", "modify", "delete"])
def test_changes_to_audit_metadata_invalidate_comparison(pair, monkeypatch, name, operation):
    artifact = pair[0] / name
    if operation != "create":
        artifact.write_text("{}")

    def change_metadata():
        if operation == "delete":
            artifact.unlink()
        else:
            artifact.write_text('{"changed": true}')

    change_during_aggregation(monkeypatch, change_metadata)
    with pytest.raises(ValueError, match=f"experiment changed.*{name}"):
        compare(pair)


@pytest.mark.parametrize("change", ["corrupt", "delete"])
@pytest.mark.parametrize("storage", ["v1", "v2", "v3"])
def test_referenced_layer_is_checked_after_statistics(pair, monkeypatch, change, storage):
    if storage == "v1":
        pair = tuple(legacy_snapshot(path) for path in pair)
    elif storage == "v3":
        pair = tuple(incremental_copy(path) for path in pair)
    manifest = read_json_object(pair[0] / "result_layers.json")
    layer = pair[0] / manifest["layers"]["summary"]["path"]
    original = layer.read_bytes()

    def modify_layer():
        if change == "delete":
            layer.unlink()
        else:
            layer.write_bytes(b"!" + original[1:])

    change_during_aggregation(monkeypatch, modify_layer)
    with pytest.raises(ValueError, match="experiment changed.*result_layers"):
        compare(pair)


def test_uncommitted_v3_tail_does_not_change_the_comparison(pair, monkeypatch):
    pair = tuple(incremental_copy(path) for path in pair)
    before = compare(pair)
    assert before["status"] == "compatible"
    assert before["groups"][0]["difference"] == 1
    manifest = read_json_object(pair[0] / "result_layers.json")
    layer = pair[0] / manifest["layers"]["summary"]["path"]

    def append_tail():
        with layer.open("ab") as stream:
            stream.write(b"uncommitted," * 100_000)

    change_during_aggregation(monkeypatch, append_tail)
    assert compare(pair) == before


@pytest.mark.parametrize("version", [1, 2])
def test_imported_csv_dependency_follows_the_snapshot_version(pair, monkeypatch, version):
    if version == 1:
        pair = tuple(legacy_snapshot(path) for path in pair)
    before = compare(pair)
    change_during_aggregation(monkeypatch, lambda: (pair[0] / "result_all.csv").write_text("changed import"))
    if version == 1:
        with pytest.raises(ValueError, match="experiment changed.*result_layers"):
            compare(pair)
    else:
        assert compare(pair) == before


def test_plain_csv_changes_after_statistics_are_rejected(pair, monkeypatch):
    pair = tuple(path / "result_all.csv" for path in pair)
    change_during_aggregation(monkeypatch, lambda: pair[0].write_text("changed CSV"))
    with pytest.raises(ValueError, match="experiment changed.*result_all"):
        compare(pair)


@pytest.mark.parametrize("relative", [False, True])
def test_explicit_csv_supplies_both_statistics_and_fingerprints(pair, relative, monkeypatch):
    selected = tuple(path / "result_all.csv" for path in pair)
    with selected[0].open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, row = reader.fieldnames, next(reader)
    assert fields is not None
    with selected[0].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerow({**row, "latency_app_s": "9"})
    if relative:
        root = pair[0].parent
        monkeypatch.chdir(root)
        report = compare(tuple(path.relative_to(root) for path in selected))
    else:
        report = compare(selected)
    assert report["groups"][0]["left"]["mean"] == 9
    assert report["groups"][0]["difference"] == -6
    assert report["source_sha256"][str(selected[0])] == hashlib.sha256(selected[0].read_bytes()).hexdigest()
    assert str(pair[0] / "result_layers.json") not in report["source_sha256"]


@pytest.mark.parametrize("name", ["model_resolution.json", "runtime_failures.json"])
def test_audit_metadata_fingerprints_record_presence_and_absence(pair, name):
    path = pair[0] / name
    assert compare(pair)["source_sha256"][str(path)] is None
    payload = b"{}\n"
    path.write_bytes(payload)
    assert compare(pair)["source_sha256"][str(path)] == hashlib.sha256(payload).hexdigest()


def test_cli_does_not_publish_statistics_when_a_layer_changes(pair, monkeypatch, tmp_path, capsys):
    from acprof.cli.compare import main

    manifest = read_json_object(pair[0] / "result_layers.json")
    layer = pair[0] / manifest["layers"]["summary"]["path"]
    output = tmp_path / "comparison.json"
    change_during_aggregation(monkeypatch, layer.unlink)
    with pytest.raises(SystemExit) as error:
        main(["--left", str(pair[0]), "--right", str(pair[1]), "--metric", "latency_app_s",
              "--resamples", "10", "--output", str(output)])
    assert error.value.code == 1
    assert "比较失败" in capsys.readouterr().err
    assert not output.exists()
