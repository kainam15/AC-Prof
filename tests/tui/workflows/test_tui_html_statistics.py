"""Statistics parameters belong to the offline HTML, with a reachable TUI launch."""
import asyncio
from dataclasses import replace
from unittest.mock import Mock

import pytest
from textual.widgets import Button, Input

from acprof.experiments.configuration import RunConfig
from tests.analysis.reporting.test_report_statistics import row, write_rows
from tests.support.tui_fixtures import AcprofTui


@pytest.mark.parametrize(("language", "size"), [("zh", (80, 24)), ("en", (120, 30))])
async def test_html_generation_needs_only_an_experiment_selection(tmp_path, monkeypatch, language, size):
    source = tmp_path / "result_all.csv"
    write_rows(source, [row(index, index + 1) for index in range(3)])
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    launch = Mock()
    monkeypatch.setattr(app, "launch", launch)
    monkeypatch.setattr("acprof.platform.can_open_local_browser", lambda: True)
    open_file = Mock()
    monkeypatch.setattr(app.results.html, "run_file_action", open_file)
    async with app.run_test(size=size) as pilot:
        await app.workers.wait_for_complete()
        app.activate_tab("results-tab")
        app.results.show_result_section("statistics-view")
        app.query_one("#experiment-source", Input).value = str(source)
        await pilot.pause()
        for name in ("metrics", "confidence", "resamples", "seed", "block-size"):
            assert not app.query("#analysis-" + name)
        button = app.query_one("#report-calculate", Button)
        async with asyncio.timeout(5):
            while button.disabled:
                await pilot.pause()
        assert not button.disabled
        assert button.region.width > 0
        assert button.region.bottom <= size[1]
        assert app.get_widget_at(button.region.x + 1, button.region.y + 1)[0] is button
        await pilot.click("#report-calculate")
        await pilot.pause()
        launch.assert_called_once()
        command = launch.call_args.args[0].command
        assert "report" in command
        assert str(source) in command
        for flag, value in (("--confidence", "0.95"), ("--resamples", "5000"), ("--seed", "0"), ("--block-size", "1")):
            assert command[command.index(flag) + 1] == value
        output = tmp_path / "analysis.html"
        output.write_text("<!doctype html><html><body>offline report</body></html>")
        app.query_one("#html-report-path", Input).value = str(output)
        app.results.update_report_actions()
        await pilot.pause()
        open_button = app.query_one("#html-report-open", Button)
        open_button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert open_button.region.width > 0
        assert not open_button.disabled
        await pilot.click("#html-report-open")
        open_file.assert_called_once_with("open", output)
        app.logs.flush()
