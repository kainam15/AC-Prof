from unittest.mock import patch

import pytest
from textual.widgets import Button, Collapsible, Input, Static, TabbedContent
from tests.support.tui_fixtures import AcprofTui

from acprof.experiments.configuration import RunConfig


class TestTuiResultTools:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.directory = tmp_path
        self.csv_path = self.directory / "绘图结果.csv"
        self.csv_path.write_text("status,warmup,latency_app_s\nok,0,0.02\n", encoding="utf-8")
        self.settings_path = self.directory / "tui.json"

    def make_app(self):
        return AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)

    async def test_results_page_keeps_plotting_and_optional_profiling_separate(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            tabs = app.query_one("#main-tabs", TabbedContent)
            assert [tabs.get_tab(pane).label.plain for pane in app.query("#main-tabs > ContentSwitcher > TabPane")] == [
                "实验配置", "运行监控", "实验结果", "镜像管理", "应用设置"]
            assert not app.query("#profile-tab, #result-dir")
            app.activate_tab("results-tab")
            app.query_one("#result-csv", Input).value = str(self.csv_path)
            await pilot.pause()
            assert await pilot.click("#summarize-results", offset=(2, 1))
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "成功 1" in app.query_one("#result-summary", Static).content
            assert app.query_one("#profile-section", Collapsible).collapsed
            app.query_one("#profile-section", Collapsible).collapsed = False
            await pilot.pause()
            # A loose CSV is fine for plotting, but cannot carry immutable replay identity.
            assert "独立 CSV" in str(app.query_one("#profile-status", Static).content)
            assert app.query_one("#open-result-profiling", Button).disabled

            with patch.object(app.process, "execute_command") as execute:
                assert await pilot.click("#plot-results", offset=(2, 1))
                await pilot.pause()
                execute.assert_called_once()
                command, kind = execute.call_args.args
                assert kind == "plot"
                assert command[1:5] == ["-u", "-m", "acprof", "plot"]
                assert command[5] == str(self.csv_path)
                assert tabs.active == "monitor-tab"
                assert all(widget.disabled for widget in app.query(
                    "#summarize-results, #plot-results, #open-result-profiling"))
                app.process_finished("plot", 0, None, "")
                await pilot.pause()
                assert not app.query_one("#plot-results", Button).disabled
                assert app.query_one("#open-result-profiling", Button).disabled

    @pytest.mark.parametrize("command", ["results", "summary"])
    async def test_summary_commands_show_the_results_page(self, command):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            field = app.query_one("#slash-command", Input)
            field.value = f'/{command} "{self.csv_path}"'
            field.focus()
            await pilot.pause()
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.query_one("#main-tabs", TabbedContent).active == "results-tab"
            summary = app.query_one("#result-summary", Static)
            assert "成功 1" in summary.content
            assert summary.region.height > 0
            assert not app.query_one("#plot-results", Button).disabled
