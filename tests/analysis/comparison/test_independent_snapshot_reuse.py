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


def test_flat_directories_and_explicit_files_share_report_sources(experiment_fixture):
    from acprof.analysis.model import load_analysis
    from acprof.analysis.report_comparison import build_experiment_comparisons, comparison_sources

    left, right = (experiment_fixture.replicate(side, 0, [value]) for side, value in (("left", 2), ("right", 5)))
    inputs = [left, right]
    for path in inputs:
        (path / "result_layers.json").unlink()
    files = [str(path / "result_all.csv") for path in inputs]
    options = {"metrics": ["latency_app_s"], "resamples": 20}
    report = compare_experiments([left], [right], **options)
    assert report == compare_experiments(files[:1], files[1:], **options)
    assert report["groups"][0]["difference"] == 3
    assert [report["experiments"][side][0]["source"] for side in ("left", "right")] == files
    assert all(report["source_sha256"][path] == hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in files)
    model = load_analysis(inputs)
    assert [source["path"] for source in model.sources] == comparison_sources(inputs) == files
    assert [source["sha256"] for source in model.sources] == [report["source_sha256"][path] for path in files]
    # Old independent reports recorded experiment directories as source names.
    for side, directory in zip(("left", "right"), inputs, strict=True):
        report["experiments"][side][0]["source"] = str(directory)
    compared, = build_experiment_comparisons(model, independent_report=report)
    assert compared["report"]["groups"] == report["groups"]
    with pytest.raises(ValueError, match="duplicate CSV input"):
        load_analysis([left, files[0]])


def test_reused_session_keys_choices_by_artifact(experiment_fixture, monkeypatch):
    from acprof.analysis import independent_comparison

    left, right = (experiment_fixture.replicate(side, 0, [value]) for side, value in (("left", 2), ("right", 5)))
    calls = []
    original = independent_comparison.read_result_csv_snapshot

    def read(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(independent_comparison, "read_result_csv_snapshot", read)
    session = independent_comparison.ExperimentComparisonSession()
    options = {"metrics": ["latency_app_s"], "resamples": 20}
    first = session.compare([left], [right], **options)
    layered = [path / "result_layers.json" for path in (left, right)]
    assert session.compare(layered[:1], layered[1:], **options) == first
    assert calls == layered
    with pytest.raises(ValueError, match="duplicate independent run"):
        session.compare([left], layered[:1], **options)
    assert calls == layered


def test_legacy_report_cannot_rebind_unchanged_flat_digest_to_layered_result(experiment_fixture):
    from acprof.analysis.model import load_analysis
    from acprof.analysis.report_comparison import build_experiment_comparisons

    left, right = (experiment_fixture.replicate(side, 0, [value]) for side, value in (("left", 2), ("right", 5)))
    report = compare_experiments([left / "result_all.csv"], [right / "result_all.csv"],
                                 metrics=["latency_app_s"], resamples=20)
    for side, root in (("left", left), ("right", right)):
        report["experiments"][side][0]["source"] = str(root)
    with pytest.raises(ValueError, match="source.*changed|fingerprint"):
        build_experiment_comparisons(load_analysis([left, right]), independent_report=report)


def test_explicit_csv_alias_reads_metadata_from_original_experiment(experiment_fixture, tmp_path):
    from acprof.analysis.model import load_analysis

    root = experiment_fixture.replicate("left", 0, [2])
    target = root / "result_all.csv"
    alias = tmp_path / "left-export.csv"
    alias.symlink_to(target)
    assert load_analysis([alias]).sources == load_analysis([target]).sources


def test_independent_csv_alias_retains_its_experiment_and_cannot_count_twice(experiment_fixture):
    from acprof.analysis.independent_comparison import ExperimentComparisonSession
    from acprof.analysis.model import load_analysis
    from acprof.analysis.report_comparison import (
        build_experiment_comparisons,
        verify_independent_report,
    )

    first = experiment_fixture.replicate("left", 0, [2])
    second = experiment_fixture.replicate("left", 1, [2])
    right = experiment_fixture.replicate("right", 0, [5])
    target = first / "result_all.csv"
    alias = second / "export.csv"
    alias.symlink_to(target)
    session = ExperimentComparisonSession()
    options = {"metrics": ["latency_app_s"], "resamples": 20}
    direct = session.compare([target], [right], **options)
    assert direct["experiments"]["left"][0]["run_id"] == "left-0"
    linked = session.compare([alias], [right], **options)
    run = linked["experiments"]["left"][0]
    assert (run["run_id"], run["source"]) == ("left-1", str(alias))
    assert linked["source_sha256"][str(alias)] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert str(second / "run_state.json") in linked["source_sha256"]
    assert str(first / "run_state.json") not in linked["source_sha256"]
    verify_independent_report(linked)
    with pytest.raises(ValueError, match="duplicate independent.*source"):
        session.compare([target], [alias], **options)
    with pytest.raises(ValueError, match="source metadata"):
        build_experiment_comparisons(load_analysis([alias, right]), independent_report=linked)
    with pytest.raises(ValueError, match="source metadata"):
        build_experiment_comparisons(load_analysis([alias, right]), groups=([alias], [right]), statistics_options=options)
    # Older reports canonicalized the experiment name while hashing the literal alias.
    run["source"] = str(target)
    with pytest.raises(ValueError, match="source.*changed|fingerprint"):
        verify_independent_report(linked)


def test_distinct_runs_check_the_files_actually_read(experiment_fixture, monkeypatch):
    from acprof.analysis import independent_comparison

    left, right = (experiment_fixture.replicate(side, 0, [2]) for side in ("left", "right"))
    target = left / "result_all.csv"
    replacement = right / "copy.csv"
    replacement.write_bytes(target.read_bytes())
    alias = right / "export.csv"
    alias.symlink_to(target)
    collect = independent_comparison._collect_run_case_groups

    def retarget_after_read(rows, **kwargs):
        if kwargs["run_id"] == "right-0":
            alias.unlink()
            alias.symlink_to(replacement)
        return collect(rows, **kwargs)

    monkeypatch.setattr(independent_comparison, "_collect_run_case_groups", retarget_after_read)
    with pytest.raises(ValueError, match="duplicate independent.*source"):
        compare_experiments([target], [alias], metrics=["latency_app_s"], resamples=20)
