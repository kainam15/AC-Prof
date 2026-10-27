"""Independent comparisons share source audits only within one invocation."""
import copy
import json
from collections import Counter
from pathlib import Path

import pytest
from independent_comparison_fixtures import IndependentComparisonFixture

from acprof.analysis import comparison
from acprof.analysis.independent_comparison import compare_experiments
from acprof.quality import loading_quality


@pytest.fixture(name="runs")
def comparison_runs(request, tmp_path):
    fixture = IndependentComparisonFixture()
    fixture.build(request, tmp_path)
    return ([fixture.replicate("left", i, [2, 2]) for i in range(3)],
            [fixture.replicate("right", i, [3, 3]) for i in range(3)])


@pytest.mark.parametrize("explicit_csv", [False, True])
def test_each_independent_run_is_audited_once(runs, monkeypatch, explicit_csv):
    left, right = runs
    if explicit_csv:
        left, right = ([path / "result_all.csv" for path in side] for side in runs)
    calls = Counter()
    audit = comparison.audit_result

    def record_audit(source):
        calls[Path(source)] += 1
        return audit(source)

    monkeypatch.setattr("acprof.analysis.comparison.audit_result", record_audit)
    report = compare_experiments(left, right, metrics=["latency_app_s"], resamples=10)
    assert report["status"] == "compatible"
    group = report["groups"][0]
    assert group["difference"] == 1
    assert group["ratio"] == 1.5
    assert group["difference_ci"] == [1, 1]
    assert group["left"]["n_runs"] == group["right"]["n_runs"] == 3
    assert len(report["condition_checks"]) == 12
    assert calls == Counter({path: 1 for path in [*left, *right]})


def test_next_comparison_reloads_changed_quality_evidence(runs):
    left, right = runs
    before = compare_experiments(left, right, metrics=["latency_app_s"], resamples=10)
    artifact = right[0] / "quality_checks.json"
    checks = loading_quality({"missing_keys": ["head.weight"]}, source="loader")
    artifact.write_text(json.dumps({"schema_version": 1, "checks": checks}))
    after = compare_experiments(left, right, metrics=["latency_app_s"], resamples=10)
    assert before["groups"][0]["right"]["quality_status"] != "blocked"
    assert after["groups"][0]["right"]["quality_status"] == "blocked"
    assert after["groups"][0]["difference"] == 1
    assert after["source_sha256"][str(artifact)] != before["source_sha256"][str(artifact)]


@pytest.mark.parametrize("purpose", ["same-hardware", "cross-hardware", "resource-scaling"])
def test_snapshot_comparison_does_not_read_or_mutate_inputs(runs, monkeypatch, purpose):
    left = comparison.load_comparison_snapshot(runs[0][0])
    right = comparison.load_comparison_snapshot(runs[1][0])
    right["hardware"]["host_id"]["1c_4g_off"] = "different-host"
    original = copy.deepcopy((left, right))

    def unexpected_read(*_args, **_kwargs):
        pytest.fail("comparing loaded snapshots must not read files")

    monkeypatch.setattr(Path, "open", unexpected_read)
    report = comparison.compare_snapshots(left, right, purpose=purpose)
    assert report["status"] == ("compatible" if purpose == "cross-hardware" else "incompatible")
    assert report["expected_differences"]["runtime_backend"] == {"left": "torch", "right": "onnxruntime"}
    assert report["metric_comparability"]["latency_app_s"]["status"] == (
        "comparable" if purpose == "cross-hardware" else "not comparable"
    )
    assert (left, right) == original


@pytest.mark.parametrize(("compare", "inputs"), [
    (comparison.compare_results, ("missing-left", "missing-right")),
    (comparison.compare_snapshots, ({}, {})),
])
def test_invalid_purpose_is_rejected_before_comparison_io(monkeypatch, compare, inputs):
    def unexpected_read(*_args, **_kwargs):
        pytest.fail("an invalid comparison purpose must not read inputs")

    monkeypatch.setattr(Path, "open", unexpected_read)
    with pytest.raises(ValueError, match="comparison purpose"):
        compare(*inputs, purpose="invalid")
