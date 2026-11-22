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


@pytest.mark.parametrize("field", ["resource_metric_quality", "gpu_energy_source"])
def test_csv_quality_text_is_distinct_from_an_empty_cell(tmp_path, field):
    metric = "gpu_energy_total_j"
    evidence = json.dumps({"schema_version": 1, "metrics": {}, "collectors": {"gpu_power": {"status": "available"}}})
    rows = [window(scale, index, gpu_energy_total_j=value, gpu_energy_source="", resource_metric_quality=evidence)
            for scale, index, value in [(1, 0, 2), (2, 0, 2), (2, 1, 98)]]
    rows[-2][field] = ""
    rows[-1][field] = "N/A"
    p = tmp_path / "result_all.csv"
    with p.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with p.open(newline="") as stream:
        expected = summarize_windows(csv.DictReader(stream), [metric])["groups"][1]
    points = aggregate_plot_metric(prepare_df(str(p)), metric)
    point = points[points["input_scale"] == 2].iloc[0]
    assert (point["n_windows"], point["missing_windows"]) == (expected["n_windows"], expected["missing_windows"])
    if field == "resource_metric_quality":
        assert point[metric] == expected["mean"] == 2
    else:
        assert pd.isna(point[metric])
        assert expected["mean"] is None


@pytest.mark.parametrize("quoting", [csv.QUOTE_MINIMAL, csv.QUOTE_ALL], ids=["unquoted", "quoted"])
def test_gpu_method_whitespace_matches_between_stats_and_png(tmp_path, quoting):
    from acprof.cli.stats import main as run_stats

    metric = "gpu_energy_total_j"
    method = "nvml_total_energy"
    evidence = json.dumps({"schema_version": 1, "metrics": {}, "collectors": {"gpu_power": {"status": "available"}}})
    methods = [
        (method, method), (method, method + " "), (method, " " + method),
        (method, "power_integration"), (method, method.upper()), ("001", "1"), ("", "N/A"),
    ]
    rows = []
    for scale, pair in enumerate(methods, 1):
        for index, source in enumerate(pair):
            rows.append(window(scale, index, gpu_energy_total_j=2 if index == 0 else 98,
                               gpu_energy_source=source, resource_metric_quality=evidence))
    path = tmp_path / "result_all.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0], quoting=quoting)
        writer.writeheader()
        writer.writerows(rows)
    original = path.read_bytes()
    output = tmp_path / "stats.json"
    assert run_stats([str(path), "--metric", metric, "--output", str(output)]) == 0
    groups = json.loads(output.read_bytes())["groups"]
    stats = [(g["mean"], g["n_windows"], g["missing_windows"]) for g in groups]
    assert stats == [(50, 2, 0)] * 3 + [(None, 0, 2)] * 4

    points = aggregate_plot_metric(prepare_df(str(path)), metric)
    png = [(None if pd.isna(mean) else mean, valid, missing)
           for mean, valid, missing in points[[metric, "n_windows", "missing_windows"]].itertuples(index=False, name=None)]
    assert png == stats
    assert path.read_bytes() == original
