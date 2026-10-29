"""Independent run statistics and condition checks share verified source rows."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest

from acprof.analysis.independent_comparison import compare_experiments
from tests.analysis.comparison.test_comparison_source_integrity import (
    incremental_copy,
    legacy_snapshot,
)
from tests.support.independent_comparison_fixtures import IndependentComparisonFixture


@pytest.fixture
def experiment_fixture(request, tmp_path):
    fixture = IndependentComparisonFixture()
    fixture.build(request, tmp_path)
    return fixture


@pytest.mark.parametrize(("storage", "reads"), [("csv", 2), ("v1", 3), ("v2", 3), ("v3", 2)])
def test_source_reads_only_parse_snapshot_and_verify_final_content(experiment_fixture, monkeypatch, storage, reads):
    left = [experiment_fixture.replicate("left", 0, [2])]
    right = [experiment_fixture.replicate("right", 0, [3])]
    if storage == "v3":
        left, right = ([incremental_copy(path) for path in side] for side in (left, right))
    elif storage == "v1":
        left, right = ([legacy_snapshot(path) for path in side] for side in (left, right))
    if storage == "csv":
        left, right = ([path / "result_all.csv" for path in side] for side in (left, right))
        payloads = {*left, *right}
        sources = payloads
    else:
        sources = {root / "result_layers.json" for root in [*left, *right]}
        payloads = {source.parent / info["path"] for source in sources
                    for info in json.loads(source.read_bytes())["layers"].values()}
    expected = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    opens = defaultdict(list)
    original_open = Path.open

    def observe(candidate, *args, **kwargs):
        if candidate in payloads:
            opens[candidate].append(args[0] if args else kwargs.get("mode", "r"))
        return original_open(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observe)
    report = compare_experiments(left, right, metrics=["latency_app_s"], resamples=10)
    assert report["status"] == "compatible"
    assert report["groups"][0]["difference"] == 1
    assert report["groups"][0]["ratio"] == 1.5
    assert {path: report["source_sha256"][path] for path in expected} == expected
    assert dict(opens) == {path: ["rb"] * reads for path in payloads}


def test_independent_comparison_does_not_calculate_discarded_within_run_intervals(experiment_fixture, monkeypatch):
    left = [experiment_fixture.replicate("left", i, [2, 2, 2]) for i in range(3)]
    right = [experiment_fixture.replicate("right", i, [3, 3, 3]) for i in range(3)]

    def unexpected_bootstrap(*_args, **_kwargs):
        pytest.fail("run comparison must not compute discarded within-run confidence intervals")

    monkeypatch.setattr("acprof.analysis.uncertainty.bootstrap_mean_interval", unexpected_bootstrap)
    report = compare_experiments(left, right, metrics=["latency_app_s"], resamples=20, seed=3)
    group = report["groups"][0]
    assert group["left"]["n_runs"] == group["right"]["n_runs"] == 3
    assert group["difference_ci"] == [1, 1]
    assert group["ratio_ci"] == [1.5, 1.5]


@pytest.mark.parametrize("options", [
    {"metrics": []}, {"metrics": ["latency_app_s", "latency_app_s"]},
    {"metrics": ["unknown"]}, {"metrics": ["cpu_heap_peak_bytes_massif"]},
    {"confidence": 0}, {"confidence": 1}, {"resamples": 0}, {"resamples": 1.5},
])
def test_invalid_statistical_options_are_rejected_before_source_io(monkeypatch, options):
    def unexpected_read(*_args, **_kwargs):
        pytest.fail("invalid comparison options must not read experiment files")

    monkeypatch.setattr(Path, "open", unexpected_read)
    with pytest.raises(ValueError, match="confidence|指标|独立测量窗口"):
        compare_experiments(["left"], ["right"], **{"metrics": ["latency_app_s"], **options})
