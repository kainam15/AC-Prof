"""One historical CSV is sufficient for descriptive resource comparisons."""
import csv

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
