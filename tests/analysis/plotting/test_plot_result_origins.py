"""Performance figures use measured rows while the feasibility map keeps planned cells."""
import csv
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from acprof.plotting import data, diagnostics, pipeline, process, tail_latency

INFERRED = "inferred_not_measured"
PRUNED = "not_measured_after_startup_oom_pruning: confirmed startup OOM"


def write_windows(directory: Path, *, include_origin=True, include_status=True):
    coordinates = dict(cpu_cores=2, mem_cap_gb=4, gpu_mode="off", input_scale=1, warmup=0)
    rows = [
        {**coordinates, "status": status, "result_origin": origin, "repeat_idx": repeat,
         "error": error, "latency_app_p50_s": p50, "latency_app_p95_s": 2 * p50}
        for repeat, (status, origin, p50, error) in enumerate([
            ("ok", "measured", 1, ""), ("ok", f" {INFERRED} ", 101, PRUNED),
            ("warn", "", 3, ""), ("error", INFERRED, 401, PRUNED),
        ])
    ]
    fields = list(rows[0])
    if not include_origin:
        fields.remove("result_origin")
    if not include_status:
        fields.remove("status")
    path = directory / "result_all.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("include_status", [False, True])
def test_prepared_metric_windows_exclude_known_placeholders(tmp_path, include_status):
    path = write_windows(tmp_path, include_status=include_status)
    original = path.read_bytes()
    expected = [1] if include_status else [1, 3]

    metrics = data.prepare_df(str(path))
    assert metrics["latency_app_p50_s"].tolist() == expected
    assert data.prepare_df(str(path), only_ok=False)["repeat_idx"].tolist() == [0, 1, 2, 3]
    assert path.read_bytes() == original


def test_historical_windows_without_an_origin_column_still_plot(tmp_path):
    path = write_windows(tmp_path, include_origin=False)
    assert data.prepare_df(str(path))["latency_app_p50_s"].tolist() == [1, 101]


@pytest.mark.parametrize("only_ok", [False, True])
def test_plot_pipeline_uses_one_performance_population_for_both_runtimes(tmp_path, monkeypatch, only_ok):
    path = write_windows(tmp_path)
    captured = {}

    def render_tail(frame, **_options):
        captured["tail"] = tail_latency._aggregate_tail_latency(frame)["latency_app_p50_s"].tolist()

    def record(label):
        def store(frame, *_args, **_kwargs):
            captured[label] = frame["repeat_idx"].tolist()
        return store

    monkeypatch.setattr(pipeline.plotting_tail_latency, "plot_tail_latency_overview", render_tail)
    monkeypatch.setattr(diagnostics, "plot_resource_feasibility_heatmap", record("feasibility"))
    monkeypatch.setattr(diagnostics, "plot_latency_energy_pareto", record("pareto"))
    monkeypatch.setattr(diagnostics, "plot_cold_start_breakdown", record("cold"))
    monkeypatch.setattr(pipeline.plotting_metrics, "plot_cold_start_bar", record("startup"))
    for registry in ("METRIC_OVERVIEW_PLOTS", "PLOT_METRICS", "ENERGY_POWER_OVERVIEW_PLOTS"):
        monkeypatch.setattr(pipeline.plotting_config, registry, ())
    def publish_model(frame, *_):
        captured["model"] = frame["repeat_idx"].tolist()
        return SimpleNamespace(report_path=tmp_path / "model.json", residuals_path=tmp_path / "missing")

    monkeypatch.setattr(pipeline.analysis_latency_report, "write_latency_model_report", publish_model)
    pipeline.plot_results(str(path), only_ok=only_ok, save_png=False)
    monkeypatch.setattr(data, "read_static_meta", lambda _: {"runtime_kind": "process"})
    monkeypatch.setattr(process, "plot_process_results", record("process"))
    pipeline.plot_results(str(path), only_ok=only_ok, save_png=False)

    expected = [0] if only_ok else [0, 2]
    assert captured["feasibility"] == [0, 1, 2, 3]
    assert captured["tail"] == ([1.] if only_ok else [2.])
    assert all(captured[chart] == expected for chart in ("pareto", "cold", "startup", "model", "process"))


@pytest.mark.parametrize(("inferred_error", "observed", "expected"), [
    ("", [], "skipped"), (PRUNED, [], "pruned_startup_oom"), ("prior runtime_oom", [], "skipped"),
    (PRUNED, [("ok", "")], "ok"), (PRUNED, [("warn", "monitor")], "warn"),
    (PRUNED, [("error", "request_timeout")], "timeout"), (PRUNED, [("error", "")], "error"),
    (PRUNED, [("ok", ""), ("error", "runtime_oom")], "mixed"),
    (PRUNED, [("error", "client_request_timeout: planned_request_attempted=unknown")], "unknown"),
])
def test_inferred_feasibility_never_claims_a_measured_outcome(inferred_error, observed, expected):
    coordinates = dict(cpu_cores=2, mem_cap_gb=4, gpu_mode="off", input_scale=1)
    frame = pd.DataFrame([
        {**coordinates, "status": "ok", "error": inferred_error, "result_origin": INFERRED},
        *({**coordinates, "status": status, "error": error} for status, error in observed),
    ])
    assert diagnostics.summarize_resource_feasibility(frame)["state"].tolist() == [expected]
