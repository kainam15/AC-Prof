"""Ordered comparisons reuse run summaries without retaining raw CSV snapshots."""
import csv
import gc
import weakref
from importlib import import_module
from unittest.mock import patch

import pytest

from acprof.analysis.model import load_analysis
from acprof.analysis.report_comparison import build_experiment_comparisons
from tests.analysis.comparison.test_comparison_source_integrity import incremental_copy
from tests.support.independent_comparison_fixtures import IndependentComparisonFixture

comparison = import_module("acprof.analysis.independent_comparison")


@pytest.fixture
def runs(request, tmp_path):
    fixture = IndependentComparisonFixture()
    fixture.build(request, tmp_path)
    return fixture


def sources(runs, storage="csv"):
    windows = 1 if storage == "v3" else 3
    roots = [runs.replicate("left", index, [index + 1] * windows) for index in range(4)]
    return ([incremental_copy(root) / "result_layers.json" for root in roots]
            if storage == "v3" else [root / "result_all.csv" for root in roots])


@pytest.mark.parametrize("storage", ["csv", "v3"])
def test_ordered_pairs_classify_each_source_window_once(runs, storage):
    paths = sources(runs, storage)
    model = load_analysis(paths)
    with patch("acprof.analysis.independent_comparison.measurement_key",
               wraps=comparison.measurement_key) as classify:
        entries = build_experiment_comparisons(
            model, statistics_options={"metrics": ["latency_app_s"], "resamples": 19},
        )

    reports = [entry["report"] for entry in entries]
    assert len(reports) == 12
    assert all(report["status"] == "compatible" for report in reports)
    assert [report["groups"][0]["difference"] for report in reports] == [1, 2, 3, -1, 1, 2, -2, -1, 1, -3, -2, -1]
    windows = 1 if storage == "v3" else 3
    assert all(report["groups"][0]["left"]["runs"][0]["valid_windows"] == windows for report in reports)
    assert classify.call_count == 4 * windows


@pytest.mark.parametrize("storage", ["csv", "v3"])
def test_live_comparison_session_releases_parsed_csv_snapshots(runs, storage):
    paths = sources(runs, storage)
    snapshots = []
    original = comparison.read_result_csv_snapshot

    def observe(*args, **kwargs):
        snapshot = original(*args, **kwargs)
        snapshots.append(weakref.ref(snapshot))
        return snapshot

    session = comparison.ExperimentComparisonSession()
    with patch("acprof.analysis.independent_comparison.read_result_csv_snapshot", observe):
        report = session.compare([paths[0]], [paths[1]], metrics=["latency_app_s"], resamples=19)
    assert report["groups"][0]["difference"] == 1
    assert len(snapshots) == 2
    gc.collect()
    assert all(snapshot() is None for snapshot in snapshots)


def test_summary_reuse_respects_new_metric_selection(runs):
    paths = sources(runs)
    session = comparison.ExperimentComparisonSession()
    first = session.compare([paths[0]], [paths[1]], metrics=["latency_app_s"], resamples=19)
    assert first["groups"][0]["difference"] == 1

    second = session.compare([paths[0]], [paths[1]], metrics=["throughput_samples_per_s"], resamples=19)
    assert [group["metric"] for group in second["groups"]] == ["throughput_samples_per_s"]
    assert second["groups"][0]["left"]["mean"] is None
    assert second["groups"][0]["left"]["missing_windows"] == 3

    third = session.compare([paths[0]], [paths[1]],
                            metrics=["throughput_samples_per_s", "latency_app_s"], resamples=19)
    assert {group["metric"] for group in third["groups"]} == {"latency_app_s", "throughput_samples_per_s"}
    assert next(group for group in third["groups"] if group["metric"] == "latency_app_s")["difference"] == 1


def test_returned_run_records_cannot_mutate_later_summary_results(runs):
    paths = sources(runs)
    session = comparison.ExperimentComparisonSession()
    options = {"metrics": ["latency_app_s"], "resamples": 19}
    first = session.compare([paths[0]], [paths[1]], **options)
    record = first["groups"][0]["left"]["runs"][0]
    record.update(mean=999, valid_windows=999)
    record["resource_coordinates"]["cpu_cores"] = 999

    second = session.compare([paths[0]], [paths[2]], **options)
    group = second["groups"][0]
    assert group["difference"] == 2
    assert group["left"]["mean"] == 1
    assert group["left"]["runs"][0]["valid_windows"] == 3
    assert group["left"]["runs"][0]["resource_coordinates"] == {"cpu_cores": 1, "mem_cap_gb": 4}


def test_reused_runs_keep_membership_and_weight_in_new_comparison_groups(runs):
    paths = sources(runs)
    session = comparison.ExperimentComparisonSession()
    options = {"metrics": ["latency_app_s"], "resamples": 19}
    session.compare([paths[0]], [paths[1]], **options)

    forward = session.compare(paths[:2], paths[2:], **options)["groups"][0]
    reverse = session.compare(paths[2:], paths[:2], **options)["groups"][0]

    assert (forward["left"]["mean"], forward["right"]["mean"], forward["difference"]) == (1.5, 3.5, 2)
    assert (reverse["left"]["mean"], reverse["right"]["mean"], reverse["difference"]) == (3.5, 1.5, -2)
    assert forward["left"]["n_runs"] == forward["right"]["n_runs"] == 2
    assert [run["run_id"] for run in forward["left"]["runs"]] == ["left-0", "left-1"]
    assert [run["run_id"] for run in reverse["left"]["runs"]] == ["left-2", "left-3"]


def test_resource_scaling_rechecks_case_cardinality_after_other_purpose(runs):
    paths = sources(runs)
    with paths[0].open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    assert fields is not None
    with paths[0].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows([*rows, {**rows[0], "cpu_cores": "2"}])
    runs.fixture.refresh(paths[0].parent)
    session = comparison.ExperimentComparisonSession()
    options = {"metrics": ["latency_app_s"], "resamples": 19}
    first = session.compare([paths[0]], [paths[1]], **options)
    assert len(first["groups"]) == 2

    with pytest.raises(ValueError, match="resource-scaling requires one resource configuration"):
        session.compare([paths[0]], [paths[1]], purpose="resource-scaling", **options)
