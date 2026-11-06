"""Independent comparison classifies windows once and parses each selected cell once."""
import csv
from collections import Counter
from unittest.mock import Mock

import pytest

from acprof.analysis import independent_comparison as comparison
from tests.support.independent_comparison_fixtures import IndependentComparisonFixture

METRICS = (
    "latency_app_s", "latency_app_p50_s", "latency_app_p90_s", "latency_app_p95_s",
    "latency_app_max_s", "latency_app_std_s", "cpu_energy_total_j", "throughput_samples_per_s",
)


@pytest.fixture(name="runs")
def comparison_runs(request, tmp_path):
    fixture = IndependentComparisonFixture()
    fixture.build(request, tmp_path)
    return fixture


@pytest.mark.parametrize("metric_count", [1, 4, 8])
def test_additional_metrics_do_not_reclassify_windows_or_reparse_cells(runs, monkeypatch, metric_count):
    left = runs.replicate("left", 0, [2.5] * 12)
    right = runs.replicate("right", 0, [3.5] * 12)
    classify = Mock(wraps=comparison.is_measured_success)
    status = Mock(wraps=comparison.result_row_status)
    parse = comparison.number
    parsed = Counter()

    def count_parse(value):
        parsed[value] += 1
        return parse(value)

    monkeypatch.setattr(comparison, "is_measured_success", classify)
    monkeypatch.setattr(comparison, "result_row_status", status)
    monkeypatch.setattr(comparison, "number", count_parse)
    report = comparison.compare_experiments([left], [right], metrics=list(METRICS[:metric_count]), resamples=20)

    groups = {group["metric"]: group for group in report["groups"]}
    assert len(groups) == metric_count
    latency = groups["latency_app_s"]
    assert (latency["left"]["mean"], latency["right"]["mean"], latency["difference"]) == (2.5, 3.5, 1)
    assert latency["left"]["runs"][0]["valid_windows"] == 12
    assert latency["left"]["failed_windows"] == 0
    for metric in METRICS[1:metric_count]:
        assert groups[metric]["left"]["mean"] is None
        assert groups[metric]["left"]["missing_windows"] == 12
    assert parsed["2.5"] == parsed["3.5"] == 12
    assert parsed["nan"] == 24 * (metric_count - 1)
    assert classify.call_count <= 24
    assert status.call_count <= 24


def _mixed_run(runs, side, complete):
    directory = runs.replicate(side, 0, [2, 999, 4, 999, 999, 999, 999, 999, 999])
    runs.fixture.change_json(directory, "run_state.json", lambda state: state.update(
        status="complete" if complete else "failed",
        options={**state["options"], "repeat": 8, "warmup": 1},
    ))
    path = directory / "result_all.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    assert fields is not None
    changes = [
        {"status": " OK ", "cpu_energy_total_j": "10"},
        {"latency_app_s": "nan", "cpu_energy_total_j": "20"},
        {"cpu_energy_total_j": "inf"},
        {"status": "warn", "error": "fixture warning", "cpu_energy_total_j": "999"},
        {"status": "error", "error": "fixture failure", "cpu_energy_total_j": "999"},
        {"result_origin": "inferred_not_measured", "cpu_energy_total_j": "999"},
        {"warmup": "1", "repeat_idx": "0", "cpu_energy_total_j": "999"},
        {"latency_app_s": "invalid", "cpu_energy_total_j": "nan"},
        {"latency_app_s": "-inf", "cpu_energy_total_j": "nan"},
    ]
    for row, change in zip(rows, changes, strict=True):
        row.update(change)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    runs.fixture.refresh(directory)
    return directory


@pytest.mark.parametrize("complete", [True, False])
def test_multi_metric_summaries_keep_failure_missing_inferred_and_completion_evidence(runs, complete):
    left = _mixed_run(runs, "left", complete)
    right = _mixed_run(runs, "right", True)
    before = {path: path.read_bytes() for root in (left, right) for path in root.rglob("*") if path.is_file()}
    report = comparison.compare_experiments(
        [left], [right], metrics=["latency_app_s", "cpu_energy_total_j"], resamples=20,
    )

    groups = {group["metric"]: group for group in report["groups"]}
    for metric, expected_mean in (("latency_app_s", 3), ("cpu_energy_total_j", 15)):
        side = groups[metric]["left"]
        assert side["mean"] == (expected_mean if complete else None)
        assert (side["n_runs"], side["missing_runs"]) == (int(complete), int(not complete))
        assert (side["failed_windows"], side["missing_windows"], side["inferred_windows"]) == (2, 3, 1)
        record = side["runs"][0]
        assert (record["valid_windows"], record["eligible"]) == (2, complete)
        assert groups[metric]["right"]["mean"] == expected_mean
    assert {path: path.read_bytes() for path in before} == before
