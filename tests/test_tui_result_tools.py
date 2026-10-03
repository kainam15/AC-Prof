from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, Checkbox, Input, Static, TabbedContent, TabPane
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig


class TestTuiResultTools:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        temporary = tmp_path
        self.directory = Path(str(temporary))
        self.csv_path = self.directory / "绘图结果.csv"
        self.csv_path.write_text("status,warmup,latency_app_s\nok,0,0.02\n", encoding="utf-8")
        self.settings_path = self.directory / "tui.json"

    def make_app(self):
        return AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)

    async def test_separate_tabs_keep_drafts_and_plot_uses_its_csv(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            tabs = app.query_one("#main-tabs", TabbedContent)
            assert ([tabs.get_tab(pane).label.plain for pane in tabs.query(TabPane)]) == (["实验配置", "运行监控", "绘图工具", "统计报告", "补采工具", "镜像管理", "应用设置"])

            assert (await pilot.click(tabs.get_tab("profile-tab")))
            await pilot.pause()
            assert (tabs.active) == ("profile-tab")
            assert not (app.query("#profile-tab #result-csv, #profile-tab #plot-results"))
            directory_draft = str(self.directory / "未提交的补采目录")
            app.query_one("#result-dir", Input).value = directory_draft
            assert (await pilot.click("#profile-tool-nsys", offset=(2, 1)))
            await pilot.pause()

            assert (await pilot.click(tabs.get_tab("plot-tab")))
            await pilot.pause()
            assert (tabs.active) == ("plot-tab")
            assert not (app.query("#plot-tab #result-dir, #plot-tab .profile-tool, #plot-tab #profile-run"))
            app.query_one("#result-csv", Input).value = str(self.csv_path)
            assert (await pilot.click("#summarize-results", offset=(2, 1)))
            await pilot.pause()
            assert ("成功 1") in (app.query_one("#result-summary", Static).content)
            assert (app.query_one("#result-dir", Input).value) == (directory_draft)

            assert (await pilot.click(tabs.get_tab("profile-tab")))
            await pilot.pause()
            assert (app.query_one("#profile-tool-nsys", Checkbox).value)
            assert (app.query_one("#result-dir", Input).value) == (directory_draft)
            assert (await pilot.click(tabs.get_tab("plot-tab")))
            await pilot.pause()
            assert (app.query_one("#result-csv", Input).value) == (str(self.csv_path))

            with patch.object(app, "_execute_command") as execute:
                assert (await pilot.click("#plot-results", offset=(2, 1)))
                await pilot.pause()
                execute.assert_called_once()
                command, kind = execute.call_args.args
                assert (kind) == ("plot")
                assert (command[1:5]) == (["-u", "-m", "acprof", "plot"])
                assert (command[5]) == (str(self.csv_path))
                assert (tabs.active) == ("monitor-tab")
                controls = list(app.query(
                    "#summarize-results, #plot-results, #profile-dry-run, #profile-run, .profile-tool"
                ))
                assert (all(widget.disabled for widget in controls))
                app._process_finished("plot", 0, None, "")
                await pilot.pause()
                assert (all(not widget.disabled for widget in controls))

    @pytest.mark.parametrize('command', ('results', 'summary'))
    async def test_summary_commands_show_the_plotting_page(self, command):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            field = app.query_one("#slash-command", Input)
            field.value = f'/{command} "{self.csv_path}"'
            field.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert (app.query_one("#main-tabs", TabbedContent).active) == ("plot-tab")
            summary = app.query_one("#result-summary", Static)
            assert ("成功 1") in (summary.content)
            assert (summary.region.height) > (0)
            assert not (app.query_one("#plot-results", Button).disabled)
