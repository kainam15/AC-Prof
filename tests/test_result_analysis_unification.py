"""Results share statistics without mandatory analysis artifacts or measurement changes."""
from __future__ import annotations

import asyncio
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from textual.widgets import Button, DataTable, Input, Static, TabbedContent
from tui_fixtures import AcprofTui

from acprof.analysis.export import export_analysis, statistics_rows
from acprof.analysis.uncertainty import summarize_windows
from acprof.experiment import RunConfig
from acprof.plotting.metrics import plot_metric
from acprof.plotting.uncertainty import (
    WindowStatistics,
    aggregate_plot_metric,
    aggregate_plot_series,
)
from acprof.tui.analysis_output import AnalysisOutput
from acprof.tui.commands import PendingLaunch
from acprof.tui.reports import report_view


def windows(**changes):
    return [dict(cpu_cores=2, mem_cap_gb=8, gpu_mode="off", input_scale=64,
                 warmup=0, repeat_idx=index, status="ok", error="", repeat_in_window=100 if index == 0 else 1,
                 latency_app_s=value, latency_s=value, container_attributed_energy_eff_j=.1, **changes)
            for index, value in enumerate((.01, .02, .03))]


def statistics_payload():
    return {**summarize_windows(windows(), ["latency_app_s"]),
            "result_csv": "/example/summary.csv", "result_sha256": "a" * 64}


@pytest.fixture
def result_csv(tmp_path):
    path = tmp_path / "experiment;one.csv"
    records = windows()
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return path


def test_plot_and_export_use_the_same_mean_deviation_and_interval():
    rows = windows()
    rows += [{**rows[0], "warmup": 1, "repeat_idx": 4, "latency_app_s": 99},
             {**rows[0], "status": "error", "repeat_idx": 5, "latency_app_s": 999}]
    payload = summarize_windows(rows, ["latency_app_s"])
    result = aggregate_plot_metric(pd.DataFrame(rows), "latency_app_s").iloc[0]
    expected = payload["groups"][0]
    assert result["latency_app_s"] == pytest.approx(.02)
    assert result["std"] == pytest.approx(.01)
    assert result["n_windows"] == 3
    for field in ("std", "ci_low", "ci_high", "n_windows"):
        assert result[field] == expected[field] == statistics_rows(payload)[0][field]


def test_plot_cache_keys_include_changed_values_and_reuses_identical_groups():
    rows = windows()
    cache = WindowStatistics()
    with patch("acprof.plotting.uncertainty.summarize_windows", wraps=summarize_windows) as calculate:
        first = aggregate_plot_metric(pd.DataFrame(rows), "latency_app_s", statistics=cache)
        repeated = aggregate_plot_metric(pd.DataFrame(rows[::-1]), "latency_app_s", statistics=cache)
        assert calculate.call_count == 1
        pd.testing.assert_frame_equal(first, repeated)
        rows[0]["latency_app_s"] = .04
        changed = aggregate_plot_metric(pd.DataFrame(rows), "latency_app_s", statistics=cache)
        assert calculate.call_count == 2
        assert changed.iloc[0]["latency_app_s"] == pytest.approx(.03)


def test_process_inherited_resources_and_gaps_keep_real_identity():
    rows = windows(runtime_kind="process", runtime_id="env", resource_scope="observed_process_tree", config="inherited")
    for row in rows:
        row.update(cpu_cores=np.nan, mem_cap_gb=np.nan)
    rows.extend({**row, "input_scale": 128, "latency_app_s": np.nan} for row in rows[:3])
    result = aggregate_plot_series(pd.DataFrame(rows), "latency_app_s", config_columns=("config",))
    assert result["input_scale"].tolist() == [64, 128]
    assert result.iloc[0]["latency_app_s"] == pytest.approx(.02)
    assert result.iloc[1]["n_windows"] == 0
    assert np.isnan(result.iloc[1]["ci_low"])
    assert np.isnan(result.iloc[1]["latency_app_s"])


def test_plot_refuses_to_pool_distinct_runtime_identities():
    rows = windows(runtime_kind="docker", runtime_id="a")
    rows[1]["runtime_id"] = "b"
    with pytest.raises(ValueError, match="mixes runtime"):
        aggregate_plot_metric(pd.DataFrame(rows), "latency_app_s")


def test_small_sample_zero_and_byte_conversion_do_not_invent_intervals():
    rows = windows()[:1]
    rows[0].update(latency_app_s=0, gpu_mem_used_peak_bytes=1024 ** 3, gpu_mem_used_peak_gib=1.)
    result = aggregate_plot_metric(pd.DataFrame(rows), "latency_app_s").iloc[0]
    assert result["latency_app_s"] == 0
    assert np.isnan(result["ci_low"])
    assert result["reason"] == "insufficient_windows"
    memory = aggregate_plot_metric(pd.DataFrame(rows), "gpu_mem_used_peak_gib").iloc[0]
    assert memory["gpu_mem_used_peak_gib"] == 1
    assert np.isnan(memory["ci_high"])


def test_median_and_non_window_metrics_remain_unmodified():
    frame = pd.DataFrame(windows())
    median = aggregate_plot_metric(frame, "latency_app_s", agg_func="median")
    assert "ci_low" not in median
    assert median.iloc[0]["latency_app_s"] == .02
    frame["cold_start_s"] = [1, 2, 3]
    assert "ci_low" not in aggregate_plot_metric(frame, "cold_start_s")


def test_png_draws_interval_and_retains_sample_mean(monkeypatch):
    figures = []
    original = plt.close
    def close(figure=None):
        if plt.get_fignums():
            figures.append(plt.gcf() if figure is None else figure)
        original(figure)
    monkeypatch.setattr(plt, "close", close)
    plot_metric(pd.DataFrame(windows()), "latency_app_s", "Latency", "s", "tokens", None)
    axis = figures[-1].axes[0]
    assert axis.lines[0].get_ydata()[0] == pytest.approx(.02)
    assert len(axis.collections) == 2
    assert "95% within-run" in axis.get_title()


def test_export_csv_and_json_are_explicit_and_preserve_parameters(tmp_path):
    payload = statistics_payload()
    csv_path, json_path = tmp_path / "stats.csv", tmp_path / "stats.json"
    export_analysis(payload, csv_path, format="csv")
    export_analysis(payload, json_path, format="json")
    with csv_path.open() as stream:
        row, = list(csv.DictReader(stream))
    assert float(row["mean"]) == .02
    assert row["analysis.seed"] == "0"
    assert row["analysis.resamples"] == "5000"
    assert row["analysis.result_sha256"] == "a" * 64
    assert row["unit"] == "s"
    assert json.loads(json_path.read_text()) == payload
    saved = csv_path.read_bytes()
    with pytest.raises(FileExistsError):
        export_analysis(payload, csv_path, format="csv")
    assert csv_path.read_bytes() == saved


def test_export_rejects_wrong_suffix_symlink_and_empty_csv(tmp_path):
    payload = statistics_payload()
    target = tmp_path / "stats.csv"
    target.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        export_analysis(payload, target, format="csv")
    with pytest.raises(ValueError, match="new .json"):
        export_analysis(payload, tmp_path / "wrong.csv", format="json")
    with pytest.raises(ValueError, match="no statistics rows"):
        export_analysis({**payload, "groups": []}, tmp_path / "empty.csv", format="csv")
    assert not (tmp_path / "empty.csv").exists()


def test_analysis_transport_validates_payload_without_files(tmp_path):
    output = AnalysisOutput()
    content = json.dumps(statistics_payload())
    for offset in range(0, len(content), 17):
        output.append(content[offset:offset + 17])
    view = output.finish("stats", 0, tmp_path)
    assert view.payload["groups"][0]["mean"] == .02
    assert list(tmp_path.iterdir()) == []
    malformed = AnalysisOutput()
    malformed.append("traceback: failed")
    with pytest.raises(ValueError, match="no valid JSON"):
        malformed.finish("stats", 0, tmp_path)


def test_analysis_transport_is_bounded(monkeypatch):
    monkeypatch.setattr("acprof.tui.analysis_output.MAX_ANALYSIS_BYTES", 3)
    output = AnalysisOutput()
    with pytest.raises(ValueError, match="exceeds"):
        output.append("中文")


@pytest.mark.parametrize("size", [(80, 24), (120, 30), (160, 45)])
async def test_unified_results_statistics_and_explicit_exports(size, result_csv, tmp_path):
    original = result_csv.read_bytes()
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=size) as pilot:
        app.results.select_result(str(result_csv), summarize=False)
        await pilot.pause()
        app.results.show_result_section("statistics-view")
        await pilot.pause()
        assert await pilot.click("#report-calculate")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.query_one("#main-tabs", TabbedContent).active == "results-tab"
        assert not list(app.query("#reports-tab, #plot-tab, #report-source"))
        assert app.results.report_view is not None
        assert app.query_one("#report-table", DataTable).row_count == 3
        assert not list(tmp_path.glob("**/window-statistics-*.json"))
        assert app.query_one("#report-json-path", Input).value == ""
        assert result_csv.read_bytes() == original
        app.results.show_result_section("exports-view")
        target = tmp_path / "statistics.csv"
        app.query_one("#analysis-export-path", Input).value = str(target)
        await pilot.pause()
        button = app.query_one("#analysis-export-csv", Button)
        button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(button)
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert target.is_file()
        assert not (tmp_path / "statistics.json").exists()
        assert result_csv.read_bytes() == original


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_cancel_analysis_reaps_child_and_leaves_no_report(result_csv, tmp_path, language):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        command = (sys.executable, "-u", "-c", "import time; time.sleep(30)")
        app.launch(PendingLaunch(command, "stats", result_csv=str(result_csv)))
        try:
            async with asyncio.timeout(5):
                while app.query_one("#results-cancel", Button).disabled:
                    await pilot.pause()
            process = app.lifecycle.process
            assert process is not None
            for name in ("results-cancel", "report-calculate", "plot-results"):
                button = app.query_one("#" + name, Button)
                assert button.region.right <= app.size.width
            assert await pilot.click("#results-cancel")
            await pilot.pause()
            assert await pilot.click("#confirm-yes")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert process.poll() is not None
            assert not app.is_busy()
            assert app.results.report_view is None
            cancelled = "分析已取消" if language == "zh" else "Analysis cancelled"
            assert cancelled in str(app.query_one("#report-error", Static).content)
            assert not list(tmp_path.glob("**/window-statistics-*.json"))
        finally:
            app.lifecycle.stop()


async def test_stale_analysis_cannot_replace_changed_selection(tmp_path):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test() as pilot:
        app.results.analysis_selection = app.results.selection_identity()
        app.query_one("#result-csv", Input).value = "changed.csv"
        await pilot.pause()
        app.results.accept_analysis(report_view(Path("old.csv"), statistics_payload()))
        assert app.results.analysis_view is None
