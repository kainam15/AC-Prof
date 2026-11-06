"""HTML is the primary analysis destination; failed comparisons stay at their inputs."""
from unittest.mock import patch

import pytest
from textual.widgets import Button, Input, Static, TabbedContent

from acprof.experiments.configuration import RunConfig
from acprof.tui.commands import PendingLaunch
from acprof.tui.results.analysis_output import AnalysisOutput
from tests.support.tui_fixtures import AcprofTui


def test_failed_comparison_plain_error_never_enters_json_decoder(tmp_path):
    output = AnalysisOutput()
    output.append("比较失败：metadata artifact missing: run_state.json\n")
    with patch("acprof.tui.results.analysis_output.json.loads", side_effect=AssertionError("not JSON")):
        with pytest.raises(ValueError, match="比较失败：metadata artifact missing"):
            output.finish("compare", 1, tmp_path)


async def test_failed_comparison_stays_on_comparison_form(tmp_path):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.results.show_result_section("comparison-view")
        app.results.begin_analysis(PendingLaunch(("acprof", "compare", "--left", str(tmp_path)), "compare"))
        app.results.finish_analysis(error="missing run_state.json", cancelled=False)
        await pilot.pause()
        assert app.query_one("#result-views", TabbedContent).active == "comparison-view"
        assert "missing run_state.json" in str(app.query_one("#comparison-status", Static).content)
        assert app.results.report_view is None


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_primary_analysis_action_generates_html_and_keeps_parameters(tmp_path, language):
    from dataclasses import replace

    source = tmp_path / "result_all.csv"
    source.write_text("cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_s\n"
                      "2,4,off,32,0,0,ok,1\n")
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.results.select_result(str(source), summarize=False)
        app.results.show_result_section("statistics-view")
        app.query_one("#analysis-resamples", Input).value = "101"
        app.query_one("#analysis-seed", Input).value = "7"
        await pilot.pause()
        with patch.object(app, "launch") as launch:
            assert await pilot.click("#report-calculate")
            await pilot.pause()
            pending = launch.call_args.args[0]
            assert pending.kind == "html-report"
            assert pending.command[pending.command.index("--resamples") + 1] == "101"
            assert pending.command[pending.command.index("--seed") + 1] == "7"
        button = app.query_one("#report-calculate", Button)
        assert button.region.right <= 80
        assert app.query_one("#html-report-panel").region.area > 0


async def test_successful_comparison_offers_existing_html_and_preserves_json_export(tmp_path):
    from acprof.tui.results.reports import ReportView

    report = tmp_path / "comparison.html"
    report.write_text("<!doctype html><title>fixture</title>")
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        app.results.begin_analysis(PendingLaunch(("acprof", "compare", "--left", str(tmp_path)), "compare", report_path=str(report)))
        view = ReportView(tmp_path, "comparison", (), (), "fixture", payload={"html_report_path": str(report), "groups": []})
        app.results.accept_analysis(view)
        app.results.finish_analysis(error="", cancelled=False)
        await pilot.pause()
        assert app.query_one("#result-views", TabbedContent).active == "statistics-view"
        assert app.query_one("#html-report-path", Input).value == str(report)
        assert app.results.report_view.payload == view.payload
        assert not app.query_one("#analysis-export-json", Button).disabled
