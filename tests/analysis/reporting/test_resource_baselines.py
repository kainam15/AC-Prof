"""One historical CSV is sufficient for descriptive resource comparisons."""
import csv
import json

import pytest

from acprof.analysis.model import load_analysis
from acprof.plotting.report import report_payload


def pair_for(payload, left, right):
    table = payload["baseline_comparisons"]
    pair = table["pairs"][left["config_id"]][right["config_id"]]
    return {**pair, "resource": table["qualifications"][pair["resource"]]}


def matrix(tmp_path, rows):
    source = tmp_path / "result_all.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    return source


def row(cpu, value, **extra):
    return dict(cpu_cores=cpu, mem_cap_gb=4, gpu_mode="off", input_scale=32,
                repeat_idx=0, warmup=0, status="ok", latency_app_s=value, **extra)


def test_historical_resource_baseline_has_python_deltas_without_fabricating_run_identity(tmp_path):
    source = matrix(tmp_path, [row(2, 4), row(4, 2), row(4, 900, repeat_in_window=1) | {"repeat_idx": 1, "status": "error"}])
    model = load_analysis([source])
    before, after = model.configs
    payload = report_payload(model)
    assert payload["comparison_purpose"] == "within-experiment"
    pair = pair_for(payload, before, after)
    assert pair["resource"]["status"] == "descriptive"
    assert pair["resource"]["changed_resources"] == ["cpu"]
    assert pair["metrics"]["latency_app_s"] == {
        "delta": -2, "percent": -50, "ratio": .5, "performance_ratio": 2, "reason": "",
    }
    assert before["run_id"] == after["run_id"] == "unknown"
    assert before["config_id"].startswith("source-")
    assert payload["window_statistics"][0]["groups"][0]["mean"] == 4


@pytest.mark.parametrize(("changes", "expected"), [
    ({"mem_cap_gb": 8}, ["cpu", "memory"]),
    ({"gpu_mode": "on"}, ["cpu", "gpu"]),
])
def test_multiple_resource_changes_are_explicit(tmp_path, changes, expected):
    source = matrix(tmp_path, [row(2, 4), row(4, 2) | changes])
    payload = report_payload(load_analysis([source]))
    a, b = payload["configs"]
    pair = pair_for(payload, a, b)
    assert pair["resource"]["changed_resources"] == expected
    assert pair["resource"]["status"] == "descriptive"
    assert pair["resource"]["causal_attribution"] is False


def test_different_inputs_are_visible_but_never_receive_a_resource_ratio(tmp_path):
    source = matrix(tmp_path, [row(2, 4), row(4, 2) | {"input_scale": 64}])
    payload = report_payload(load_analysis([source]))
    a, b = payload["configs"]
    pair = pair_for(payload, a, b)
    assert pair["resource"]["status"] == "incompatible"
    assert "input_case" in pair["resource"]["reasons"]
    assert not pair["metrics"]


def test_baseline_retains_reason_when_numeric_metrics_have_different_sampling(tmp_path):
    def measured(cpu, interval):
        evidence = {"latency_app_s": {"quality": "complete", "sampling": {"interval_s": interval}}}
        return row(cpu, cpu, resource_metric_quality=json.dumps(evidence))

    payload = report_payload(load_analysis([matrix(tmp_path, [measured(1, 1), measured(2, .5)])]))
    left, right = payload["configs"]
    for baseline, target in ((left, right), (right, left)):
        pair = pair_for(payload, baseline, target)
        assert pair["resource"]["status"] == "descriptive"
        assert "latency_app_s" not in pair["metrics"]
        assert pair["metric_reasons"] == {"latency_app_s": "different_source_scope_window_or_sampling"}
        assert "metric_reasons" not in pair_for(payload, target, target)


def test_matched_cross_runtime_baseline_requires_complete_formal_windows(tmp_path, monkeypatch):
    from acprof.analysis import resource_comparison

    compare = resource_comparison.compare_profiles

    def matched_profiles(left, right):
        verdict = compare(left, right)
        if left["purpose"] == "same-hardware":
            verdict["matched_cross_runtime"] = True
        return verdict

    monkeypatch.setattr(resource_comparison, "compare_profiles", matched_profiles)
    complete = json.dumps({"latency_app_s": {"quality": "complete"}})
    source = matrix(tmp_path, [row(1, 2, resource_metric_quality=complete), row(2, 4, resource_metric_quality=complete),
                               row(2, "", resource_metric_quality=complete) | {"repeat_idx": 1}, row(3, 6)])
    model = load_analysis([source])
    left, right, legacy = sorted(model.configs, key=lambda c: c["cpu"])
    pairs = resource_comparison.build_baseline_comparisons(model)["pairs"][left["config_id"]]
    assert pairs[left["config_id"]]["metrics"]["latency_app_s"]["delta"] == 0
    pair = pairs[right["config_id"]]
    assert pair["cross_runtime"]
    assert "latency_app_s" not in pair["metrics"]
    assert pair["metric_reasons"]["latency_app_s"] == "incomplete_metric_quality"
    legacy_pair = pairs[legacy["config_id"]]
    assert "latency_app_s" not in legacy_pair["metrics"]
    assert legacy_pair["metric_reasons"]["latency_app_s"] == "incomplete_cross_runtime_metric_evidence"


def test_explicit_statistics_options_invalidate_cached_parameters(tmp_path):
    from acprof.analysis.statistics_report import (
        build_statistics_report,
        save_unique_statistics_report,
    )
    source = matrix(tmp_path, [row(2, value) | {"repeat_idx": i} for i, value in enumerate((1, 2, 3))])
    cached = build_statistics_report(source, metrics=["latency_app_s"], seed=19, resamples=101)
    save_unique_statistics_report(tmp_path / "analysis", cached)
    result = report_payload(load_analysis([source]), statistics_options={
        "metrics": ["latency_app_s"], "seed": 7, "resamples": 99,
    })["window_statistics"][0]
    assert result["origin"] == "computed"
    assert result["parameters"]["seed"] == 7
    assert result["parameters"]["resamples"] == 99
    assert result["cache_diagnostics"][0]["reason"] == "statistics_parameters_changed"


def test_cli_window_statistics_and_html_exclude_inferred_observations_consistently(tmp_path):
    from acprof.analysis.statistics_report import build_statistics_report

    source = matrix(tmp_path, [row(2, 2), row(2, 999, result_origin="inferred_not_measured") | {"repeat_idx": 1}])
    cli = build_statistics_report(source, metrics=["latency_app_s"], resamples=99)
    html = report_payload(load_analysis([source]), statistics_options={"metrics": ["latency_app_s"], "resamples": 99})
    group = html["window_statistics"][0]["groups"][0]
    assert cli["groups"][0]["mean"] == group["mean"] == 2
    assert cli["groups"][0]["n_windows"] == group["n_windows"] == 1
    assert group["excluded_windows"] == {"inferred_not_measured": 1}
    assert html["configs"][0]["metrics"]["latency_app_s"]["value"] == 2


@pytest.mark.parametrize(("value", "packet", "origin", "delta"), [
    ("nan", "unavailable", "measured", 2),
    (101, "partial", "measured", None),
    (101, "partial", "inferred_not_measured", 2),
])
def test_baseline_metric_uses_evidence_from_finite_measured_windows(tmp_path, value, packet, origin, delta):
    quality = json.dumps({"schema_version": 1, "metrics": {}, "collectors": {"packet_latency": {"status": "available"}}})
    def packet_row(cpu, latency, state, kind):
        return row(cpu, 1, latency_s=latency, packet_status=state, result_origin=kind, resource_metric_quality=quality)
    path = matrix(tmp_path, [packet_row(1, 2, "ok", "measured"), packet_row(2, 4, "ok", "measured"),
                             packet_row(2, value, packet, origin) | {"repeat_idx": 1}])
    payload = report_payload(load_analysis([path]))
    left, right = payload["configs"]
    assert right["metrics"]["latency_s"]["value"] == 4
    assert len(right["metric_observations"]["latency_s"]) == 2
    pair = pair_for(payload, left, right)
    if delta is None:
        assert pair["metric_reasons"]["latency_s"] == "incomplete_metric_quality"
    else:
        assert pair["metrics"]["latency_s"]["delta"] == delta
