"""CSV status spelling must not change comparison evidence or failure counts."""
import csv
import json
import re
from pathlib import Path

import pytest

from acprof.analysis.comparison import load_comparison_snapshot
from acprof.analysis.independent_comparison import compare_experiments
from acprof.cli.compare import main
from tests.support.independent_comparison_fixtures import IndependentComparisonFixture


@pytest.fixture(name="runs")
def comparison_runs(request, tmp_path) -> IndependentComparisonFixture:
    fixture = IndependentComparisonFixture()
    fixture.build(request, tmp_path)
    return fixture


def _change_rows(runs: IndependentComparisonFixture, directory: Path, changes: list[dict[str, str]]) -> None:
    path = directory / "result_all.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    assert fields is not None
    for row, change in zip(rows, changes, strict=True):
        row.update(change)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    runs.fixture.refresh(directory)


def _source_bytes(directories):
    return {path: path.read_bytes() for directory in directories for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("status", ["ok", "OK", " ok ", "\tOk\n"])
@pytest.mark.parametrize("layout", ["flat", "layered"])
def test_success_status_spelling_keeps_workload_hardware_and_metric_evidence(runs, status, layout):
    left = runs.replicate("left", 0, [2, 4])
    right = runs.replicate("right", 0, [3, 3])
    _change_rows(runs, left, [{"status": status}] * 2)
    before = _source_bytes((left, right))
    source = left / "result_all.csv" if layout == "flat" else left
    other = right / "result_all.csv" if layout == "flat" else right

    snapshot = load_comparison_snapshot(source)
    assert snapshot["conditions"]["actual_workload"] is not None
    assert snapshot["hardware"]["cpu_model"] == {"1c_4g_off": ["fixture CPU"]}
    assert snapshot["metric_availability"]["latency_app_s"] is True
    report = compare_experiments([source], [other], metrics=["latency_app_s"], resamples=30)
    group = report["groups"][0]
    side = group["left"]
    assert report["status"] == "compatible"
    assert (side["mean"], side["failed_windows"], side["missing_windows"]) == (3, 0, 0)
    assert side["runs"][0]["valid_windows"] == 2
    assert (group["difference"], group["ratio"]) == (0, 1)
    assert group["reason"] == "insufficient_independent_runs"
    assert _source_bytes((left, right)) == before


@pytest.mark.parametrize("status", ["warn", "WARN", " warn "])
def test_warning_spelling_retains_workload_evidence_without_successful_metrics(runs, status):
    left = runs.replicate("left", 0, [2, 4])
    _change_rows(runs, left, [{"status": status, "error": "fixture warning"}] * 2)
    snapshot = load_comparison_snapshot(left)
    workload = snapshot["conditions"]["actual_workload"]
    assert workload is not None
    assert next(iter(workload.values()))["request_count"] == 6
    assert snapshot["hardware"]["cpu_model"] == {"1c_4g_off": ["fixture CPU"]}
    assert snapshot["metric_availability"]["latency_app_s"] is False


def test_failure_missing_inferred_and_warmup_counts_stay_distinct(runs):
    left = runs.replicate("left", 0, [2, 999, 999, 999, 999, 999])
    right = runs.replicate("right", 0, [4] * 6)
    for directory in (left, right):
        runs.fixture.change_json(directory, "run_state.json",
                                 lambda state: state["options"].update(repeat=5, warmup=1))
    _change_rows(runs, left, [
        {"status": " OK "},
        {"status": " OK ", "latency_app_s": "nan"},
        {"status": " WARN ", "error": "fixture warning"},
        {"status": " ERROR ", "error": "fixture failure"},
        {"status": " OK ", "result_origin": "inferred_not_measured"},
        {"status": " OK ", "warmup": "1", "repeat_idx": "0"},
    ])
    _change_rows(runs, right, [{}, {}, {}, {}, {}, {"warmup": "1", "repeat_idx": "0"}])
    report = compare_experiments([left], [right], metrics=["latency_app_s"], resamples=30)
    side = report["groups"][0]["left"]
    assert (side["mean"], side["failed_windows"], side["missing_windows"], side["inferred_windows"]) == (2, 2, 1, 1)
    run = side["runs"][0]
    assert (run["valid_windows"], run["failed_windows"], run["missing_windows"], run["inferred_windows"]) == (1, 2, 1, 1)


@pytest.mark.parametrize("status", ["success", "okay", ""])
def test_other_status_text_does_not_become_success(runs, status):
    left = runs.replicate("left", 0, [999])
    right = runs.replicate("right", 0, [4])
    _change_rows(runs, left, [{"status": status}])
    report = compare_experiments([left], [right], metrics=["latency_app_s"], resamples=30)
    side = report["groups"][0]["left"]
    assert side["mean"] is None
    assert side["failed_windows"] == 1
    assert side["runs"][0]["valid_windows"] == 0


def test_cli_html_json_and_csv_agree_on_normalized_success_counts(runs, tmp_path, capsys):
    left = runs.replicate("left", 0, [2, 2, 2])
    right = runs.replicate("right", 0, [4, 4, 4])
    _change_rows(runs, left, [{"status": "OK"}, {"status": " ok "}, {"status": "Ok"}])
    before = _source_bytes((left, right))
    html, table = tmp_path / "comparison.html", tmp_path / "comparison.csv"

    assert main(["--left", str(left), "--right", str(right), "--metric", "latency_app_s",
                 "--resamples", "30", "--html-output", str(html), "--csv-output", str(table)]) == 0
    report = json.loads(capsys.readouterr().out)
    match = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html.read_text(), re.S)
    assert match is not None
    payload = json.loads(match.group(1))
    assert payload["experiment_comparisons"][0]["report"]["groups"] == report["groups"]
    group = report["groups"][0]
    assert (group["left"]["failed_windows"], group["difference"], group["ratio"]) == (0, 2, 2)
    with table.open(newline="") as stream:
        exported = next(csv.DictReader(stream))
    assert (exported["left.failed_windows"], exported["difference"], exported["ratio"]) == ("0", "2.0", "2.0")
    assert _source_bytes((left, right)) == before
