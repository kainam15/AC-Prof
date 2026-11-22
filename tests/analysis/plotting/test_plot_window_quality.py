"""PNG window means and intervals consume the same recorded observations as stats."""
import csv
import json

import pandas as pd
import pytest

from acprof.analysis.uncertainty import summarize_windows
from acprof.plotting.data import prepare_df
from acprof.plotting.uncertainty import WindowStatistics, aggregate_plot_metric
from acprof.results.coverage import field_collectors


def window(scale, index, **fields):
    return {"cpu_cores": 2, "mem_cap_gb": 4, "gpu_mode": "on", "input_scale": scale,
            "status": "ok", "warmup": 0, "repeat_idx": index, **fields}


@pytest.mark.parametrize(("metric", "context_field", "first", "second"), [
    ("gpu_energy_total_j", "gpu_energy_source", "nvml_total_energy", "power_integration"),
    ("gpu_energy_total_j", "gpu_energy_source", "", ""),
    ("throughput_samples_per_s", "packet_status", "ok", "missing"),
    ("cpu_instructions_per_request", "cpu_perf_running_pct", 75, ""),
    ("latency_app_s_per_input_megapixel", "input_pixels_per_request", 1000, ""),
])
def test_csv_png_points_match_window_statistics_with_collector_context(
        tmp_path, metric, context_field, first, second):
    group = field_collectors("docker")[metric]
    evidence = json.dumps({"schema_version": 1, "metrics": {}, "collectors": {group: {"status": "available"}}})

    def sample(scale, index, value, context):
        return window(scale, index, **{metric: value, context_field: context, "resource_metric_quality": evidence})

    rows = [sample(1, 0, 2, first), sample(2, 0, 2, first), sample(2, 1, 98, second)]
    if "megapixel" in metric:
        for row in rows:
            row["latency_app_s"] = row[metric] * 0.001
    p = tmp_path / "result_all.csv"
    with p.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with p.open(newline="") as stream:
        expected = summarize_windows(csv.DictReader(stream), [metric])["groups"]
    points = aggregate_plot_metric(prepare_df(str(p)), metric)
    assert points["input_scale"].tolist() == [g["input_scale"] for g in expected]
    for (_, point), group_summary in zip(points.iterrows(), expected, strict=True):
        assert point["n_windows"] == group_summary["n_windows"]
        assert point["missing_windows"] == group_summary["missing_windows"]
        if group_summary["mean"] is None:
            assert pd.isna(point[metric])
        else:
            assert point[metric] == pytest.approx(group_summary["mean"])


def test_plot_cache_respects_a_changed_quality_record_and_keeps_legacy_values():
    metric = "latency_app_s"
    cache = WindowStatistics()
    frame = pd.DataFrame([window(16, idx, latency_app_s=value) for idx, value in enumerate((1, 3, 99))])
    all_values = aggregate_plot_metric(frame, metric, statistics=cache).iloc[0]
    assert all_values[metric] == pytest.approx(103 / 3)
    frame.loc[2, "resource_metric_quality"] = json.dumps({metric: {"quality": "partial", "reason": "incomplete_read"}})
    observed = aggregate_plot_metric(frame, metric, statistics=cache).iloc[0]
    assert (observed[metric], observed["n_windows"], observed["missing_windows"]) == (2, 2, 1)
    assert pd.isna(observed["ci_low"])
    assert pd.isna(observed["ci_high"])
