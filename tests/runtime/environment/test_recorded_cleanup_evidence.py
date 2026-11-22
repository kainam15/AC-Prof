"""Compatibility reports keep validation cleanup evidence alongside the primary outcome."""
import csv
import json

import pytest

from acprof.analysis.compatibility import report_results
from acprof.failures import Failure


def record(source, **artifacts):
    source.mkdir()
    for name, value in artifacts.items():
        (source / f"{name}.json").write_text(json.dumps(value))
    return source


@pytest.mark.parametrize("embedded", [False, True], ids=["standalone", "embedded"])
def test_process_cleanup_evidence_preserves_typed_timeout_and_its_own_cause(tmp_path, embedded):
    failure = Failure("predict", "request_timeout", "prediction timed out", device="off").to_dict()
    validation = {"status": "error", "cleanup_status": "incomplete", "devices": {"off": {
        "status": "inconclusive", "failure": failure,
        "cleanup_error": {"status": "incomplete", "detail": "supervisor still present",
                          "run_error": "request expired"}}}}
    artifacts = {"static_meta": {"runtime_validation": validation}} if embedded else {"runtime_validation": validation}
    source = record(tmp_path / "model", **artifacts)
    row, = report_results([source], tmp_path / "report")["rows"]
    assert row["status"] == "inconclusive"
    assert row["failure"] == failure
    assert row["runtime_status"] == "error"
    assert row["cleanup_status"] == "incomplete"
    assert row["cleanup_errors"] == [validation["devices"]["off"]["cleanup_error"]]


def test_docker_cleanup_only_uses_operation_evidence_and_redacts_exported_text(tmp_path):
    cleanup = {"status": "incomplete", "final_state": "unknown", "container_id": "a" * 64,
               "operations": [{"operation": "rm", "returncode": 1, "stderr": "remove denied password=FAKE_CLEANUP_MARKER"},
                              {"operation": "inspect", "returncode": 0, "stderr": "successful command"}],
               "run_error": {"exception_type": "FixtureError", "detail": "original inference cause"}}
    source = record(tmp_path / "model", runtime_validation={"status": "error", "cleanup_status": "incomplete",
                                                            "devices": {}, "cleanup_error": cleanup})
    output = tmp_path / "report"
    row, = report_results([source], output)["rows"]
    assert (row["status"], row["stage"], row["reason_code"]) == ("error", "cleanup", "cleanup_incomplete")
    assert "remove denied" in row["detail"]
    assert "rm" in row["detail"]
    assert "successful command" not in row["detail"]
    assert "original inference cause" not in row["detail"]
    assert row["failure"] is None
    assert row["cleanup_errors"][0]["run_error"] == cleanup["run_error"]
    with (output / "models.csv").open(newline="") as stream:
        exported, = csv.DictReader(stream)
    assert json.loads(exported["cleanup_errors"]) == row["cleanup_errors"]
    for path in output.iterdir():
        assert "FAKE_CLEANUP_MARKER" not in path.read_text()
    assert "FAKE_CLEANUP_MARKER" in (source / "runtime_validation.json").read_text()


def test_docker_interface_cleanup_without_wrapper_status_keeps_existing_diagnosis(tmp_path):
    cleanup = {"status": "incomplete", "final_state": "present", "operations": []}
    source = record(tmp_path / "model", interface_validation={
        "status": "error", "failed_stage": "interface", "error": "interface check stopped", "cleanup_error": cleanup})
    row, = report_results([source], tmp_path / "out")["rows"]
    assert row["reason_code"] == "interface_validation_failed"
    assert row["cleanup_status"] == "incomplete"
    assert row["cleanup_errors"] == [cleanup]


def test_cleanup_debt_without_error_text_never_becomes_verified_success(tmp_path):
    source = record(tmp_path / "model", runtime_validation={"status": "ok", "cleanup_status": "incomplete"},
                    capability_report={"full_profile_complete": True})
    row, = report_results([source], tmp_path / "out")["rows"]
    assert row["status"] == "error"
    assert row["runtime_status"] == "ok"
    assert row["full_profile_complete"]
    assert row["reason_code"] == "cleanup_incomplete"
    assert row["cleanup_errors"] == []


def test_invalid_embedded_cleanup_is_attributed_locally_without_hiding_healthy_rows(tmp_path):
    invalid = record(tmp_path / "bad", static_meta={"runtime_validation": {
        "status": "error", "cleanup_status": "incomplete", "devices": {"off": {"cleanup_error": "invalid"}}}})
    valid = record(tmp_path / "good", runtime_validation={"status": "ok"})
    bad, good = report_results([invalid, valid], tmp_path / "report")["rows"]
    assert bad["failure"]["reason_code"] == "recorded_evidence_invalid"
    assert bad["failure"]["evidence"]["artifact_name"] == "static_meta.json"
    assert "off" in bad["failure"]["detail"]
    assert good["status"] == "ok"
    assert "cleanup_status" not in good


def test_process_interface_redacts_the_same_cleanup_exception_in_both_places(tmp_path):
    message = "supervisor cleanup failed: token=FAKE_SHARED_MARKER"
    source = record(tmp_path / "process", interface_validation={
        "status": "error", "error": message, "failed_stage": "interface", "cleanup_status": "incomplete",
        "cleanup_error": {"status": "incomplete", "detail": message}})
    output = tmp_path / "out"
    row, = report_results([source], output)["rows"]
    assert row["reason_code"] == "interface_validation_failed"
    assert row["detail"] == row["cleanup_errors"][0]["detail"] == "supervisor cleanup failed: token=[REDACTED]"
    assert all("FAKE_SHARED_MARKER" not in path.read_text() for path in output.iterdir())
