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
    state.layout.path("static_meta.json").write_text(json.dumps(saved_metadata()))

    row, = report_results([source], tmp_path / "report")["rows"]

    assert row["status"] == expected
    if expected == "inconclusive":
        assert row["failure"]["evidence"]["artifact"] == str(state.path)
    else:
        assert row["failure"] is None
        assert row["revision"] == REVISION


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
