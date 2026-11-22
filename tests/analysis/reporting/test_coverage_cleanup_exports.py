import csv
import json

import pytest

from acprof.analysis.compatibility import write_compatibility_report
from acprof.tui.results.reports import read_report


@pytest.mark.parametrize("recorded", [False, True], ids=["sample", "recorded-results"])
def test_cleanup_exports_preserve_primary_diagnostics_and_separate_evidence(tmp_path, recorded):
    process = {"schema_version": 1, "status": "incomplete", "detail": "worker still running",
               "run_error": "earlier process exception"}
    docker = {"schema_version": 1, "status": "incomplete", "container_id": "c" * 64, "final_state": "present",
              "docker_state": {"Running": True},
              "operations": [{"operation": "rm", "returncode": 1, "stderr": "container busy"},
                             {"operation": "inspect", "returncode": 0, "stderr": ""}],
              "run_error": {"exception_type": "RuntimeError", "detail": "earlier Docker exception"}}
    rows = [
        {"model_id": "fixture/process", "failure": {"reason_code": "request_timeout", "stage": "predict",
         "detail": "request timed out"}, "cleanup_status": "incomplete", "cleanup_errors": [process]},
        {"model_id": "fixture/docker", "runtime_status": "failed", "stage": "image",
         "reason_code": "preparation_failed", "detail": "image unavailable",
         "cleanup_status": "incomplete", "cleanup_errors": [
             {**docker, "docker_state": {"Running": True, "Health": {"Log": ["UNPUBLISHED_FIXTURE"]}}}]},
        {"model_id": "fixture/unknown-cause", "runtime_status": "error", "cleanup_status": "incomplete"},
        {"model_id": "fixture/healthy", "runtime_status": "ok"},
    ]
    write_compatibility_report(tmp_path, rows, recorded_results=recorded)
    assert rows[1]["cleanup_errors"][0]["docker_state"]["Health"]["Log"] == ["UNPUBLISHED_FIXTURE"]
    with (tmp_path / "models.csv").open(newline="") as stream:
        process_csv, docker_csv, unknown_csv, healthy_csv = csv.DictReader(stream)
    assert (process_csv["status"], process_csv["stage"], process_csv["reason_code"], process_csv["detail"]) == (
        "inconclusive", "predict", "request_timeout", "request timed out")
    assert (docker_csv["status"], docker_csv["stage"], docker_csv["reason_code"], docker_csv["detail"]) == (
        "failed", "image", "preparation_failed", "image unavailable")
    for row, errors in ((process_csv, [process]), (docker_csv, [docker]), (unknown_csv, [])):
        assert row["cleanup_status"] == "incomplete"
        assert json.loads(row["cleanup_errors"]) == errors
    assert healthy_csv["cleanup_status"] == ""
    assert healthy_csv["cleanup_errors"] == ""

    markdown = (tmp_path / "REPORT.md").read_text()
    process_md, docker_md, unknown_md, healthy_md = [line for line in markdown.splitlines() if "| fixture/" in line]
    assert "request_timeout | request timed out |" in process_md
    assert "worker still running" in process_md
    assert "preparation_failed | image unavailable |" in docker_md
    assert "present" in docker_md
    assert "rm" in docker_md
    assert "incomplete" in unknown_md
    assert "incomplete" not in healthy_md
    assert "earlier process exception" not in markdown
    assert "earlier Docker exception" not in markdown
    assert "UNPUBLISHED_FIXTURE" not in markdown

    scope = "recorded_results; no_reexecution" if recorded else "selected_sample_only; no_formal_measurement"
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps({"schema_version": 1 if recorded else 2, "scope": scope, "rows": rows}))
    process_view, docker_view, *_ = read_report(path).rows
    assert process_view.cells[2] == "request_timeout"
    assert "worker still running" in process_view.detail
    assert docker_view.cells[2] == "preparation_failed"
    assert "container busy" in docker_view.detail


@pytest.mark.parametrize("recorded", [False, True], ids=["sample", "recorded-results"])
def test_coverage_without_cleanup_keeps_original_export_layout(tmp_path, recorded):
    write_compatibility_report(tmp_path, [{"model_id": "fixture/legacy", "runtime_status": "ok"}],
                               recorded_results=recorded)
    header = (tmp_path / "models.csv").read_text().splitlines()[0]
    assert header.endswith("source,run_status,run_outcome,runtime_status" if recorded else "source")
    assert "cleanup" not in header
    markdown = (tmp_path / "REPORT.md").read_text()
    assert "| Source | Model | Revision | Status | reason_code | Detail |\n" in markdown
    assert markdown.endswith("| huggingface | fixture/legacy |  | ok |  |  |\n")
