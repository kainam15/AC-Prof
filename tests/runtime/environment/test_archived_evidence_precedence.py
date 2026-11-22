"""Archive fallback is based on file presence and reports the selected physical source."""
import csv
import json

import pytest

from acprof.analysis.compatibility import report_results
from acprof.cli.coverage import main as coverage
from acprof.failures import Failure
from acprof.results.layout import ArtifactLayout


def archive(tmp_path, **files):
    root = tmp_path / "model"
    root.mkdir()
    for name, payload in files.items():
        (root / f"{name}.json").write_text(json.dumps(payload))
    return root


def test_empty_standalone_artifacts_do_not_inherit_a_stale_success_or_failure(tmp_path):
    stale = Failure("predict", "request_timeout", "old snapshot").to_dict()
    metadata = {"model_name": "current/model", "runtime_validation": {"status": "ok"},
                "capability_report": {"full_profile_complete": True},
                "model_resolution": {"failure": stale, "provenance": {"sources": {
                    "repository_snapshot": {"source": "modelscope"}}}},
    }
    root = archive(tmp_path, static_meta=metadata, runtime_validation={}, capability_report={}, model_resolution={})
    row, = report_results([root], tmp_path / "out")["rows"]
    assert row["status"] == row["runtime_status"] == "unverified"
    assert row["source"] == "huggingface"
    assert row["full_profile_complete"] is False
    assert row["failure"] is None


def test_empty_interface_shadows_inner_failure_but_missing_interface_replays_it(tmp_path):
    resolution = {"interface_validation": {"status": "error", "failed_stage": "signature", "error": "old error"}}
    root = archive(tmp_path, static_meta={"model_resolution": resolution}, interface_validation={})
    empty, = report_results([root], tmp_path / "present")["rows"]
    assert empty["status"] == "unverified"
    (root / "interface_validation.json").unlink()
    missing, = report_results([root], tmp_path / "missing")["rows"]
    assert (missing["status"], missing["detail"]) == ("failed", "old error")


@pytest.mark.parametrize(("field", "value"), [
    ("runtime_validation", []),
    ("runtime_validation", {"devices": []}),
    ("runtime_validation", {"devices": {"off": {"failure": 1}}}),
    ("capability_report", "damaged"),
    ("model_resolution", {"failure": "damaged"}),
    ("model_resolution", {"provenance": {"sources": []}}),
    ("model_resolution", {"interface_validation": "damaged"}),
], ids=["probe-type", "devices-type", "device-failure", "capability-type",
        "resolution-failure", "source-map", "interface-type"])
def test_invalid_embedded_evidence_names_its_real_file(tmp_path, field, value):
    root = archive(tmp_path, static_meta={"model_source": "modelscope", field: value})
    row, = report_results([root], tmp_path / "out")["rows"]
    assert row["status"] == "inconclusive"
    assert row["failure"]["reason_code"] == "recorded_evidence_invalid"
    assert row["failure"]["evidence"]["artifact"] == str(root / "static_meta.json")
    if field != "model_resolution":
        assert row["source"] == "modelscope"


def test_invalid_nested_interface_names_the_standalone_resolution(tmp_path):
    root = archive(tmp_path, model_resolution={"interface_validation": ["broken"]})
    row, = report_results([root], tmp_path / "out")["rows"]
    assert row["failure"]["evidence"]["artifact"] == str(root / "model_resolution.json")


def test_existing_nonfile_is_a_local_error_and_never_replays_embedded_capability(tmp_path):
    root = archive(tmp_path, static_meta={"capability_report": {"full_profile_complete": True}})
    (root / "capability_report.json").mkdir()
    healthy = tmp_path / "healthy"
    healthy.mkdir()
    (healthy / "runtime_validation.json").write_text('{"status":"ok"}')
    bad, good = report_results([root, healthy], tmp_path / "out")["rows"]
    assert bad["status"] == "inconclusive"
    assert bad["failure"]["evidence"]["artifact"] == str(root / "capability_report.json")
    assert bad["full_profile_complete"] is False
    assert good["status"] == "ok"


def test_v2_embedded_failure_points_to_the_real_metadata_path(tmp_path):
    layout = ArtifactLayout.for_new_run(tmp_path / "run")
    layout.initialize()
    path = layout.path("static_meta.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model_resolution": {"failure": {"stage": "invalid"}}}))
    row, = report_results([layout.root], tmp_path / "out")["rows"]
    assert row["failure"]["evidence"]["artifact"] == str(path)


@pytest.mark.parametrize("artifact", ["static_meta.json", "quality_checks.json", "runtime_validation.json"])
def test_coverage_report_isolates_invalid_unicode_and_exports_healthy_siblings(tmp_path, artifact):
    bad = archive(tmp_path, static_meta={"model_name": "fixture/damaged"})
    blocker = {"code": "shape_error", "severity": "error", "detail": "Unexpected shape", "evidence": {}}
    if artifact == "static_meta.json":
        payload = {"model_name": "fixture/\ud800"}
    elif artifact == "quality_checks.json":
        damaged = {**blocker, "code": "bad_check", "evidence": {"items": [{"\udc00": 1}]}}
        payload = {"schema_version": 1, "checks": [damaged, blocker]}
    else:
        payload = {"devices": {"\ud800": {"quality_checks": [blocker]}, "off": {"quality_checks": [blocker]}}}
    (bad / artifact).write_text(json.dumps(payload), encoding="utf-8")
    good = tmp_path / "healthy"
    good.mkdir()
    model = "fixture/模型😀"
    (good / "static_meta.json").write_text(json.dumps({"model_name": model}), encoding="utf-8-sig")
    (good / "runtime_validation.json").write_text('{"status":"ok"}', encoding="utf-8")
    out = tmp_path / "report"
    assert coverage(["report", str(bad), str(good), "--output-dir", str(out)]) == 0
    damaged_row, healthy = json.loads((out / "coverage.json").read_text(encoding="utf-8"))["rows"]
    if artifact != "quality_checks.json":
        assert damaged_row["status"] == "inconclusive"
        assert damaged_row["failure"]["reason_code"] == "recorded_evidence_invalid"
        assert damaged_row["failure"]["evidence"]["artifact"] == str(bad / artifact)
    if artifact != "static_meta.json":
        assert damaged_row["quality_status"] == "blocked"
        assert damaged_row["quality_reasons"] == ["quality_evidence_invalid", "shape_error"]
        warning, = (c for c in damaged_row["quality_checks"] if c["code"] == "quality_evidence_invalid")
        assert warning["evidence"]["artifact"] == str(bad / artifact)
    assert (healthy["model_id"], healthy["status"]) == (model, "ok")
    with (out / "models.csv").open(encoding="utf-8", newline="") as stream:
        invalid_csv, valid_csv = csv.DictReader(stream)
    assert invalid_csv["status"] == damaged_row["status"]
    assert valid_csv["model_id"] == model
    assert model in (out / "REPORT.md").read_text(encoding="utf-8")
