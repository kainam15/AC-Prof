"""Collector evidence remains local to its metrics when auditing and comparing rows."""
import csv
import json

import pytest

from acprof.analysis.comparison import audit_and_compare_results
from acprof.config import CSV_FIELDS
from acprof.results.profile import collection_field_outcomes, metric_observation
from tests.support.comparison_fixtures import ComparisonFixture


@pytest.fixture
def measured_window(tmp_path):
    fixture = ComparisonFixture()
    fixture.build(tmp_path)
    path = fixture.left / "result_all.csv"
    with path.open(newline="") as stream:
        row = next(csv.DictReader(stream))

    def record(state, prefix="", *, group="client_http", **values):
        evidence = {"schema_version": 1, "metrics": {}, "collectors": {
            group: state, "protocol": {"status": "available"}}}
        measured = {**row, **values, "resource_metric_quality": prefix + json.dumps(evidence)}
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, CSV_FIELDS)
            writer.writeheader()
            writer.writerow(measured)
        fixture.refresh(fixture.left)
        report = audit_and_compare_results(fixture.left, fixture.right)
        return measured, report

    return record


@pytest.mark.parametrize("prefix", ["", " \t\n"])
def test_collector_partial_quality_survives_json_whitespace(measured_window, prefix):
    row, audit = measured_window({"status": "partial", "reason": "incomplete_requests"}, prefix)
    assert metric_observation(row, "latency_app_s")["quality"] == "partial"
    assert audit["partial_metrics"]["latency_app_s"] == {"incomplete_requests": 1}
    comparison = audit["comparison"]
    assert comparison["status"] == "compatible"
    assert comparison["metric_comparability"]["latency_app_s"] == {
        "status": "not comparable", "reason": "incomplete_metric_quality"}


@pytest.mark.parametrize("state", [None, [], "bad"])
def test_invalid_collector_state_affects_only_its_metrics(measured_window, state):
    row, audit = measured_window(state)
    observation = metric_observation(row, "latency_app_s")
    assert observation["quality"] == "error"
    assert "client_http" in observation["reason"]
    assert metric_observation(row, "cpu_cores")["quality"] == "complete"
    assert audit["missing_metrics"]["latency_app_p95_s"] == {"error": 1}
    comparison = audit["comparison"]
    assert comparison["status"] == "compatible"
    assert comparison["metric_comparability"]["latency_app_s"] == {
        "status": "not comparable", "reason": "incomplete_metric_quality"}
    assert comparison["metric_comparability"]["cpu_cores"] == {"status": "comparable", "reason": ""}


@pytest.mark.parametrize("sampling", [None, [], [["sample_hz", 10]]])
def test_energy_metrics_require_an_object_for_recorded_sampling(measured_window, sampling):
    row, audit = measured_window({"status": "available", "sampling": sampling}, group="cpu_energy",
                                 vcpu_energy_eff_j="1", cpu_energy_total_j="2")
    metrics = audit["comparison"]["metric_comparability"]
    for name in ("vcpu_energy_eff_j", "cpu_energy_total_j"):
        observation = metric_observation(row, name)
        assert observation["quality"] == "error"
        assert "sampling" in observation["reason"]
        assert metrics[name] == {"status": "not comparable", "reason": "incomplete_metric_quality"}
    assert metrics["latency_app_s"]["status"] == "comparable"


@pytest.mark.parametrize(("field", "evidence", "value"), [
    ("reader_errors", None, "1"), ("reader_errors", {"latency_app_s": None}, "1"),
    ("reader_errors", {"latency_app_s": {"status": "error"}}, "1"),
    ("reader_errors", {"latency_app_s": {"detail": "failed"}}, "nan"),
    ("preflight_missing", [], "nan"), ("preflight_missing", {"latency_app_s": {}}, "nan"),
    ("preflight_missing", {"latency_app_s": {"detail": "denied", "status": []}}, "nan"),
])
def test_malformed_consumed_diagnostics_grade_the_metric_as_error(measured_window, field, evidence, value):
    row, audit = measured_window({"status": "available", field: evidence}, latency_app_s=value)
    observation = metric_observation(row, "latency_app_s")
    assert observation["quality"] == "error"
    assert field in observation["reason"]
    if value == "nan":
        assert audit["missing_metrics"]["latency_app_s"] == {"error": 1}
    else:
        assert audit["comparison"]["metric_comparability"]["latency_app_s"] == {
            "status": "not comparable", "reason": "incomplete_metric_quality"}
    assert metric_observation(row, "cpu_cores")["quality"] == "complete"


def test_sibling_diagnostics_and_unconsumed_collector_fields_do_not_weaken_other_metrics(measured_window):
    row, audit = measured_window({"status": "available", "sampling": None,
                                 "reader_errors": {"latency_app_p95_s": None}, "preflight_missing": []})
    assert metric_observation(row, "latency_app_s")["quality"] == "complete"
    assert audit["comparison"]["metric_comparability"]["latency_app_s"]["status"] == "comparable"
    assert audit["missing_metrics"]["latency_app_p95_s"] == {"error": 1}


def test_reader_diagnostic_needs_status_only_when_the_value_is_missing(measured_window):
    row, audit = measured_window({"status": "available", "preflight_missing": None,
                                 "reader_errors": {"latency_app_s": {"detail": "reader gap"},
                                                   "latency_app_p95_s": {"status": "permission_denied", "detail": "denied"}}})
    assert metric_observation(row, "latency_app_s")["quality"] == "partial"
    assert audit["partial_metrics"]["latency_app_s"] == {"reader gap": 1}
    assert audit["missing_metrics"]["latency_app_p95_s"] == {"permission_denied": 1}


def test_inapplicable_collector_requires_a_text_reason(measured_window):
    row, _ = measured_window({"status": "not_applicable", "reason": {"unexpected": "shape"}})
    field = collection_field_outcomes([row])["latency_app_s"]
    assert field["windows"] == {"error": 1}
    assert field["reasons"] == {"metric reason must be text": 1}
    assert metric_observation(row, "cpu_cores")["quality"] == "complete"
