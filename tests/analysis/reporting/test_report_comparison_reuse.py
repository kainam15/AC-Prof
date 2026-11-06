"""One report reuses parsed runs while every comparison retains source checks."""
import csv
from importlib import import_module
from itertools import permutations
from unittest.mock import patch

import pytest

from acprof.analysis.model import load_analysis
from acprof.analysis.report_comparison import build_experiment_comparisons
from acprof.plotting.report import write_report
from tests.analysis.comparison.test_comparison_source_integrity import incremental_copy
from tests.support.independent_comparison_fixtures import IndependentComparisonFixture

comparison = import_module("acprof.analysis.independent_comparison")


@pytest.fixture
def runs(tmp_path):
    fixture = IndependentComparisonFixture()
    fixture.build(None, tmp_path)
    return fixture, [fixture.replicate("left", index, [index + 1]) for index in range(4)]


@pytest.mark.parametrize("storage", ["csv", "v3"])
def test_report_loads_each_run_once_for_all_ordered_comparisons(runs, storage):
    _, roots = runs
    paths = [incremental_copy(path) for path in roots] if storage == "v3" else [path / "result_all.csv" for path in roots]
    model = load_analysis(paths)
    options = {"metrics": ["latency_app_s"], "resamples": 19}
    expected = [{**comparison.compare_experiments([left], [right], **options),
                 "engine": "python", "origin": "computed", "rng": "python_random_mt19937"}
                for left, right in permutations([source["path"] for source in model.sources], 2)]
    with patch("acprof.analysis.independent_comparison.read_result_csv_snapshot",
               wraps=comparison.read_result_csv_snapshot) as reads, \
            patch("acprof.analysis.independent_comparison.load_comparison_snapshot",
                  wraps=comparison.load_comparison_snapshot) as audits:
        actual = build_experiment_comparisons(model, statistics_options=options)
    assert [entry["report"] for entry in actual] == expected
    assert [entry["report"]["groups"][0]["difference"] for entry in actual] == [1, 2, 3, -1, 1, 2, -2, -1, 1, -3, -2, -1]
    assert reads.call_count == audits.call_count == 4


@pytest.mark.parametrize("artifact", ["run_state.json", "model_resolution.json", "result_all.csv"])
def test_report_rechecks_cached_sources_before_publication(runs, monkeypatch, tmp_path, artifact):
    _, roots = runs
    model = load_analysis([root / "result_all.csv" for root in roots])
    original = comparison.compare_snapshots
    calls = 0

    def change_in_second_pair(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 5:
            (roots[0] / artifact).write_text('{"changed": true}')
        return original(*args, **kwargs)

    monkeypatch.setattr("acprof.analysis.independent_comparison.compare_snapshots", change_in_second_pair)
    target = tmp_path / "report.html"
    with pytest.raises(ValueError, match="experiment changed during comparison"):
        write_report(model, target, statistics_options={"metrics": ["latency_app_s"], "resamples": 19})
    assert not target.exists()


def test_next_report_reads_new_values_instead_of_reusing_prior_snapshots(runs):
    fixture, roots = runs
    paths = [root / "result_all.csv" for root in roots]
    options = {"metrics": ["latency_app_s"], "resamples": 19}
    first = build_experiment_comparisons(load_analysis(paths), statistics_options=options)
    assert first[0]["report"]["groups"][0]["difference"] == 1
    with paths[0].open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    assert fields is not None
    rows[0]["latency_app_s"] = "9"
    with paths[0].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    fixture.fixture.refresh(roots[0])
    second = build_experiment_comparisons(load_analysis(paths), statistics_options=options)
    assert second[0]["report"]["groups"][0]["difference"] == -7
    assert second[0]["report"]["groups"][0]["left"]["mean"] == 9


def test_returned_comparison_cannot_mutate_evidence_for_later_pairs(runs):
    _, roots = runs
    session = comparison.ExperimentComparisonSession()
    options = {"metrics": ["latency_app_s"], "resamples": 19}
    first = session.compare([roots[0]], [roots[1]], **options)
    first["condition_checks"][0]["conditions"]["measurement_protocol"]["left"]["sample_hz"] = 999
    first["condition_checks"][0]["experiments"]["left"]["issues"].append("mutated by caller")

    second = session.compare([roots[0]], [roots[2]], **options)

    assert second["status"] == "compatible"
    assert second["groups"][0]["difference"] == 2
    assert "mutated by caller" not in second["condition_checks"][0]["experiments"]["left"]["issues"]


def test_cached_run_cannot_count_as_an_independent_replicate(runs):
    _, roots = runs
    session = comparison.ExperimentComparisonSession()
    options = {"metrics": ["latency_app_s"], "resamples": 19}
    session.compare([roots[0]], [roots[1]], **options)

    with pytest.raises(ValueError, match="duplicate independent run: left-0"):
        session.compare([roots[0]], [roots[0]], **options)
