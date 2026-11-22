"""Only the primary model's observed identities and resolved commits may be compared."""
import json

import pytest

from acprof.analysis.compatibility import report_results
from acprof.results.layout import ArtifactLayout

MODEL = "fixture/primary"
REVISION = "a" * 40
OTHER_REVISION = "b" * 40


def saved_metadata():
    return {
        "model_name": MODEL, "model_revision": REVISION, "model_source": "modelscope",
        "model_resolution": {
            "model_revision": REVISION,
            "provenance": {
                "model_id": MODEL, "revision": REVISION,
                "sources": {"repository_snapshot": {"source": "modelscope", "model_id": MODEL, "revision": REVISION}},
            },
            "interface_validation": {"status": "ok", "model_id": MODEL, "revision": REVISION},
        },
        "runtime_validation": {"status": "ok", "devices": {}},
        "capability_report": {"full_profile_complete": True},
    }


def write_metadata(source, metadata):
    layout = ArtifactLayout.for_new_run(source)
    layout.initialize()
    layout.path("static_meta.json").write_text(json.dumps(metadata))
    layout.path("quality_checks.json").write_text('{"schema_version":1,"checks":[]}')
    return layout


@pytest.mark.parametrize(("record", "field", "variant"), [
    ("snapshot", "model_id", "fixture/stale"), ("snapshot", "revision", OTHER_REVISION),
    ("interface", "model_id", "fixture/stale"), ("interface", "revision", OTHER_REVISION),
])
@pytest.mark.parametrize("storage", ["embedded", "independent"])
def test_mixed_primary_model_evidence_stays_with_its_actual_file(tmp_path, record, field, variant, storage):
    meta = saved_metadata()
    resolution = meta["model_resolution"]
    target = (resolution["provenance"]["sources"]["repository_snapshot"]
              if record == "snapshot" else resolution["interface_validation"])
    target[field] = variant
    source = tmp_path / "mixed"
    if storage == "independent":
        filename = "model_resolution.json" if record == "snapshot" else "interface_validation.json"
        direct = meta.pop("model_resolution") if record == "snapshot" else resolution.pop("interface_validation")
    else:
        filename = "static_meta.json"
    layout = write_metadata(source, meta)
    if storage == "independent":
        layout.path(filename).write_text(json.dumps(direct))
    healthy = tmp_path / "healthy"
    write_metadata(healthy, saved_metadata())

    row, valid = report_results([source, healthy], tmp_path / "report")["rows"]

    assert row["status"] == "inconclusive"
    assert row["source"] == "unknown"
    assert row["failure"]["reason_code"] == "recorded_evidence_invalid"
    assert row["failure"]["evidence"]["artifact"] == str(layout.path(filename))
    assert field in row["failure"]["detail"]
    assert valid["status"] == "full_success"
    assert valid["source"] == "modelscope"

    from acprof.tui.experiments.catalog import read_experiment_record
    from acprof.tui.models.candidates import candidate_from_record, model_choice
    observed = read_experiment_record(source)
    assert observed.model_source == "unknown"
    assert any(filename in issue for issue in observed.issues)
    assert model_choice(candidate_from_record(observed, {})).actions == {"view"}


@pytest.mark.parametrize(("model", "commit", "expected"), [
    (MODEL, REVISION, "interrupted"), ("fixture/stale", REVISION, "inconclusive"),
    (MODEL, OTHER_REVISION, "inconclusive"),
])
def test_compares_bound_task_not_requested_model_alias_or_ref(tmp_path, model, commit, expected, monkeypatch):
    from acprof.host.detect import TaskInfo
    from acprof.host.images.runtime import ImageInfo
    from acprof.host.input_plan import PlannedInputScales
    from acprof.host.run_state import RunState

    source = tmp_path / "frozen"
    monkeypatch.setattr("acprof.host.run_state.host_identity", lambda project_dir: {})
    state = RunState(source, {"model": "user/requested-alias", "revision": "latest"}, resume=False, project_dir=str(tmp_path))
    try:
        task = TaskInfo(model, "fill-mask", "nlp", "transformers_pipeline", "transformers", commit,
                        "manual", model_source="modelscope")
        state.bind_runtime(task, ImageInfo("sha256:" + "b" * 64), PlannedInputScales([1.0], "fixture"), "", "")
    finally:
        state.close()
    metadata = saved_metadata()
    if expected == "interrupted":
        # The frozen task is the sole source of a resolved commit in this historical directory.
        metadata["model_revision"] = "main"
        resolution = metadata["model_resolution"]
        resolution["model_revision"] = "release-v1"
        resolution["provenance"]["revision"] = "main"
        resolution["provenance"]["sources"]["repository_snapshot"]["revision"] = "main"
        resolution["interface_validation"]["revision"] = "main"
    state.layout.path("static_meta.json").write_text(json.dumps(metadata))

    row, = report_results([source], tmp_path / "report")["rows"]

    assert row["status"] == expected
    if expected == "inconclusive":
        assert row["failure"]["evidence"]["artifact"] == str(state.path)
    else:
        assert row["failure"] is None
        assert row["revision"] == REVISION
        from acprof.tui.experiments.catalog import read_experiment_record
        assert read_experiment_record(source).revision == REVISION


def test_symbolic_refs_dependencies_and_task_overrides_are_not_conflicting_primary_snapshots(tmp_path):
    meta = saved_metadata()
    meta["model_revision"] = "release-v1"
    resolution = meta["model_resolution"]
    resolution["model_revision"] = "main"
    resolution["task"] = "text-generation"
    resolution["dependencies"] = [{"model_id": "fixture/other", "revision": OTHER_REVISION}]
    resolution["provenance"]["sources"]["local_spec"] = {"model_id": "fixture/other", "revision": OTHER_REVISION}
    source = tmp_path / "history"
    write_metadata(source, meta)

    row, = report_results([source], tmp_path / "report")["rows"]

    assert row["status"] == "full_success"
    assert row["source"] == "modelscope"
    assert row["failure"] is None


def test_mixed_model_and_commit_keep_both_diagnoses_under_the_original_failure(tmp_path):
    metadata = saved_metadata()
    snapshot = metadata["model_resolution"]["provenance"]["sources"]["repository_snapshot"]
    snapshot.update(model_id="fixture/other", revision=OTHER_REVISION)
    source = tmp_path / "mixed"
    layout = write_metadata(source, metadata)
    layout.path("runtime_failures.json").write_text(json.dumps({"failures": [
        {"stage": "predict", "reason_code": "request_timeout", "detail": "original failure"},
    ]}))

    row, = report_results([source], tmp_path / "report")["rows"]

    primary, *diagnoses = row["failures"]
    assert primary == row["failure"]
    assert primary["reason_code"] == "request_timeout"
    assert len(diagnoses) == 2
    assert all(failure["evidence"]["artifact"] == str(layout.path("static_meta.json")) for failure in diagnoses)
    assert "model_id" in diagnoses[0]["detail"]
    assert "revision" in diagnoses[1]["detail"]


@pytest.mark.parametrize("embedded", [False, True])
def test_catalog_preserves_model_and_commit_conflicts_and_prevents_reuse(tmp_path, embedded):
    from acprof.tui.experiments.catalog import read_experiment_record
    from acprof.tui.models.candidates import candidate_from_record, model_choice

    metadata = saved_metadata()
    resolution = metadata["model_resolution"]
    resolution["provenance"]["sources"]["repository_snapshot"].update(
        model_id="fixture/different", revision=OTHER_REVISION,
    )
    source = tmp_path / "mixed"
    if not embedded:
        metadata.pop("model_resolution")
    layout = write_metadata(source, metadata)
    if not embedded:
        layout.path("model_resolution.json").write_text(json.dumps(resolution))

    record = read_experiment_record(source)

    assert record.model_source == "unknown"
    assert model_choice(candidate_from_record(record, {})).actions == {"view"}
    artifact_name = "static_meta.json" if embedded else "model_resolution.json"
    assert len(record.issues) == 2
    assert all(artifact_name in issue for issue in record.issues)
    assert "model_id" in record.issues[0]
    assert "revision" in record.issues[1]


@pytest.mark.parametrize(("pin_location", "expected"), [
    ("snapshot", REVISION), ("provenance", REVISION), ("resolution", REVISION),
    ("interface", REVISION), ("none", "main"), ("conflict", "unknown"),
])
def test_report_and_picker_use_the_unique_resolved_primary_commit(tmp_path, pin_location, expected):
    from unittest.mock import Mock

    from acprof.tui.experiments.catalog import read_experiment_record
    from acprof.tui.models.candidates import (
        candidate_from_record,
        coverage_candidates,
        model_choice,
    )
    from acprof.tui.models.controller import ModelController

    metadata = saved_metadata()
    metadata["model_revision"] = "main"
    resolution = metadata["model_resolution"]
    resolution["model_revision"] = REVISION if pin_location == "resolution" else "release-v1"
    provenance = resolution["provenance"]
    provenance["revision"] = REVISION if pin_location in {"provenance", "conflict"} else "main"
    snapshot = provenance["sources"]["repository_snapshot"]
    snapshot["revision"] = REVISION if pin_location == "snapshot" else "main"
    if pin_location == "conflict":
        snapshot["revision"] = OTHER_REVISION
    interface = resolution.pop("interface_validation")
    interface["revision"] = REVISION if pin_location == "interface" else "release-v2"
    resolution["dependencies"] = [{"model_id": "submodel/dependency", "revision": OTHER_REVISION}]
    source = tmp_path / "saved"
    layout = write_metadata(source, metadata)
    layout.path("interface_validation.json").write_text(json.dumps(interface))
    output = tmp_path / "report"
    row, = report_results([source], output)["rows"]
    assert row["revision"] == expected

    local = candidate_from_record(read_experiment_record(source), {})
    archived, = coverage_candidates(output / "coverage.json")
    for candidate in (local, archived):
        assert candidate.revision == expected
        if pin_location == "conflict":
            assert model_choice(candidate).actions == {"view"}
            continue
        app = Mock()
        ModelController(app).model_candidate_selected(("use", candidate), {MODEL})
        app.configuration.select_model_candidate.assert_called_once_with(
            MODEL, "modelscope", REVISION if expected == REVISION else "",
        )


@pytest.mark.parametrize("storage", ["interface", "resolution", "static-nested", "resolution-only"])
def test_preparation_identity_without_frozen_state_is_discovered(tmp_path, storage):
    from acprof.tui.experiments.catalog import scan_experiments
    from acprof.tui.models.candidates import (
        candidate_from_record,
        coverage_candidates,
        model_choice,
    )

    source = tmp_path / "history" / "model"
    source.mkdir(parents=True)
    iface = {"status": "ok", "model_id": MODEL, "revision": REVISION, "runtime_profile": "fixture"}
    if storage == "interface":
        filename, record = "interface_validation.json", iface
    elif storage == "resolution":
        filename, record = "model_resolution.json", {"interface_validation": iface}
    elif storage == "resolution-only":
        filename, record = "model_resolution.json", {"model_id": MODEL, "model_revision": REVISION}
    else:
        filename, record = "static_meta.json", {"model_resolution": {"interface_validation": iface}}
    (source / filename).write_text(json.dumps(record))
    local, = scan_experiments([source.parent]).records
    output = tmp_path / "report"
    row, = report_results([source], output)["rows"]
    archived, = coverage_candidates(output / "coverage.json")
    for candidate in (archived, candidate_from_record(local, {})):
        assert (candidate.model_id, candidate.revision, candidate.model_source) == (MODEL, REVISION, "huggingface")
        assert "use" in model_choice(candidate).actions
    assert row["runtime_status"] == "unverified"


def test_empty_standalone_interface_shadows_a_conflicting_embedded_copy(tmp_path):
    from acprof.tui.experiments.catalog import read_experiment_record

    metadata = saved_metadata()
    metadata["model_resolution"]["interface_validation"].update(model_id="fixture/stale", revision=OTHER_REVISION)
    source = tmp_path / "model"
    layout = write_metadata(source, metadata)
    layout.path("interface_validation.json").write_text("{}")
    local = read_experiment_record(source)
    row, = report_results([source], tmp_path / "report")["rows"]
    assert local.issues == ()
    assert (local.model_source, local.revision) == ("modelscope", REVISION)
    assert row["status"] == "full_success"


@pytest.mark.parametrize("coordinate", ["same", "model", "commit"])
@pytest.mark.parametrize("artifact", ["model_resolution.json", "interface_validation.json"])
def test_duplicate_run_compares_its_independent_primary_identity(tmp_path, coordinate, artifact):
    from acprof.tui.experiments.catalog import scan_experiments
    from acprof.tui.models.candidates import candidate_from_record, model_choice

    for subdir in ("first", "second"):
        source = tmp_path / subdir
        source.mkdir()
        (source / "run_state.json").write_text(json.dumps({"schema_version": 1, "run_id": "shared", "status": "interrupted"}))
        iface = {"status": "ok", "revision": OTHER_REVISION if coordinate == "commit" and subdir == "second" else REVISION,
                 "model_id": "fixture/other" if coordinate == "model" and subdir == "second" else MODEL}
        if artifact == "model_resolution.json":
            iface["model_revision"] = iface.pop("revision")
        (source / artifact).write_text(json.dumps(iface))
        (source / "result_all.csv").write_text("status,value\nok,1\n")
    record, = scan_experiments([tmp_path]).records
    assert len(record.aliases) == 2
    choice = model_choice(candidate_from_record(record, {}))
    if coordinate == "same":
        assert record.issues == ()
        assert record.revision == REVISION
        assert "use" in choice.actions
    else:
        assert "run_id_content_conflict" in record.issues
        assert record.model_source == "unknown"
        assert record.revision == "unknown"
        assert choice.actions == {"view"}
