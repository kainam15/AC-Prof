"""Recorded collection, probe and capability evidence describe different stages."""
import csv
import json

import pytest

from acprof.analysis.compatibility import report_results, result_status
from acprof.failures import Failure


def recorded_run(source, *, state, runtime="ok", full=True):
    source.mkdir()
    artifacts = {"static_meta.json": {"model_id": source.name},
                 "runtime_validation.json": {"status": runtime, "devices": {}},
                 "capability_report.json": {"full_profile_complete": full},
                 "quality_checks.json": {"schema_version": 1, "checks": []}}
    if state is not None:
        artifacts["run_state.json"] = state
    for name, data in artifacts.items():
        (source / name).write_text(json.dumps(data))
    return source


@pytest.mark.parametrize(("state", "probe", "full", "verdict"), [
    ({"status": "failed"}, "ok", True, "failed"),
    ({"status": "interrupted"}, "ok", True, "interrupted"),
    ({"status": "preparing"}, "ok", False, "preparing"),
    ({"status": "running"}, "ok", True, "running"),
    ({"status": "complete", "outcome": "partial"}, "ok", True, "partial"),
    ({"status": "complete", "outcome": "ok"}, "ok", True, "full_success"),
    ({"status": "complete"}, "ok", False, "ok"),
    (None, "ok", True, "full_success"),
    (None, "error", True, "error"),
    (None, "resource_limited", True, "resource_limited"),
])
def test_file_exports_agree_without_losing_probe_and_capability_results(tmp_path, state, probe, full, verdict):
    source = recorded_run(tmp_path / "model", state=state, runtime=probe, full=full)
    output = tmp_path / "report"
    report_results([source], output)
    row, = json.loads((output / "coverage.json").read_text())["rows"]
    with (output / "models.csv").open(newline="") as stream:
        exported, = csv.DictReader(stream)
    assert row["status"] == result_status(row) == exported["status"] == verdict
    assert row["runtime_status"] == exported["runtime_status"] == probe
    assert row["full_profile_complete"] is full
    assert row.get("run_status") == (state or {}).get("status")
    assert exported["run_status"] == (state or {}).get("status", "")
    assert exported["run_outcome"] == (state or {}).get("outcome", "")
    assert row.get("run_outcome") == (state or {}).get("outcome")
    assert f"| {verdict} |" in (output / "REPORT.md").read_text()


def test_typed_failure_wins_without_hiding_the_recorded_run_state(tmp_path):
    source = recorded_run(tmp_path / "model", state={"status": "failed"})
    failure = Failure("predict", "request_timeout", "request exceeded its budget").to_dict()
    (source / "runtime_failures.json").write_text(json.dumps({"failures": [failure]}))
    row, = report_results([source], tmp_path / "report")["rows"]
    assert row["status"] == "inconclusive"
    assert row["failure"] == failure
    assert row["run_status"] == "failed"
    assert row["runtime_status"] == "ok"
    assert row["full_profile_complete"]


@pytest.mark.parametrize("state", [{}, {"status": ["failed"]}, {"status": "unexpected"},
                                   {"status": "complete", "outcome": 2}])
def test_invalid_state_remains_local_to_its_source(tmp_path, state):
    damaged = recorded_run(tmp_path / "damaged", state=state)
    healthy = recorded_run(tmp_path / "healthy", state={"status": "complete"})
    bad, good = report_results([damaged, healthy], tmp_path / "report")["rows"]
    assert bad["status"] == "inconclusive"
    assert bad["failure"]["reason_code"] == "recorded_evidence_invalid"
    assert bad["failure"]["evidence"]["artifact_name"] == "run_state.json"
    assert good["status"] == "full_success"


def test_latest_attempt_diagnostics_ignore_a_completed_preparation(tmp_path):
    source = recorded_run(tmp_path / "model", state={"status": "failed", "attempts": [
        {"preparation_evidence": "previous.json"}, {"preparation_evidence": "current.json"}]}, runtime="not_run")
    (source / "previous.json").write_text('{"status":"failed","reason_code":"previous_failure"}')
    (source / "current.json").write_text('{"status":"complete","reason_code":"preparation_complete","last_stage":"ready"}')
    row, = report_results([source], tmp_path / "report")["rows"]
    assert row["status"] == "failed"
    assert not row.get("reason_code")
    assert not row.get("stage")


@pytest.mark.parametrize(("kind", "artifact"), [
    ("runtime_validation", "runtime_validation.json"), ("interface_validation", "interface_validation.json"),
    ("embedded_runtime", "static_meta.json"), ("embedded_interface", "static_meta.json"),
    ("preparation", "prep.json"),
])
def test_invalid_stage_status_cannot_confirm_a_successful_run(tmp_path, kind, artifact):
    source = recorded_run(tmp_path / "model", state={"status": "complete"})
    malformed = {"status": ["error"]}
    if kind == "embedded_runtime":
        (source / "runtime_validation.json").unlink()
        record = {"runtime_validation": malformed}
    elif kind == "embedded_interface":
        record = {"model_resolution": {"interface_validation": malformed}}
    else:
        record = malformed
    if kind == "preparation":
        (source / "run_state.json").write_text(json.dumps(
            {"status": "complete", "attempts": [{"preparation_evidence": artifact}]}))
    (source / artifact).write_text(json.dumps(record))
    row, = report_results([source], tmp_path / "report")["rows"]
    assert row["status"] == "inconclusive"
    assert row["failure"]["evidence"]["artifact"] == str(source / artifact)
    assert row["full_profile_complete"]


def test_failed_interface_supplies_missing_preparation_detail(tmp_path):
    state = {"status": "failed", "attempts": [{"preparation_evidence": "preparation.json"}]}
    source = recorded_run(tmp_path / "model", state=state, runtime="not_run")
    (source / "preparation.json").write_text('{"status":"failed","last_stage":"environment"}')
    (source / "interface_validation.json").write_text('{"status":"error","error":"probe exited early"}')
    row, = report_results([source], tmp_path / "report")["rows"]
    assert (row["stage"], row["reason_code"], row["detail"]) == (
        "environment", "interface_validation_failed", "probe exited early")
