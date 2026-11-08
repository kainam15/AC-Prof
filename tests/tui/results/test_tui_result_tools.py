from dataclasses import replace
from unittest.mock import patch

import pytest
from rich.cells import cell_len
from textual.widgets import Button, Input, Static, TabbedContent

from acprof.experiments.configuration import RunConfig
from acprof.tui.i18n import message_id
from acprof.tui.ui.structured import Callout
from tests.support.tui_fixtures import AcprofTui


class TestTuiResultTools:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.directory = tmp_path
        self.csv_path = self.directory / "绘图结果.csv"
        self.csv_path.write_text("status,warmup,latency_app_s\nok,0,0.02\n", encoding="utf-8")
        self.settings_path = self.directory / "tui.json"

    def make_app(self):
        app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)
        app.configuration.preferences = replace(app.configuration.preferences, show_command_bar=True)
        return app

    @pytest.mark.parametrize("plot_entry", ["button", "command"])
    async def test_results_page_keeps_plotting_and_optional_profiling_separate(self, plot_entry):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            tabs = app.query_one("#main-tabs", TabbedContent)
            assert [tabs.get_tab(pane).label.plain for pane in app.query("#main-tabs > ContentSwitcher > TabPane")] == [
                "实验配置", "运行监控", "实验结果", "补采实验", "镜像管理", "设置"]
            assert not app.query("#profile-tab, #result-dir")
            app.activate_tab("results-tab")
            app.query_one("#experiment-source", Input).value = str(self.csv_path)
            await pilot.pause()
            assert await pilot.click("#refresh-experiment-state", offset=(2, 1))
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "有效正式窗口：1" in app.query_one("#result-data-status", Static).content
            assert not app.query("#results-tab #profile-status, #results-tab #open-result-profiling")
            assert await pilot.click("#nav-backfill-tab")
            await pilot.pause()
            assert tabs.active == "backfill-tab"
            assert app.query_one("#backfill-source", Input).value == str(self.csv_path)
            # A loose CSV is fine for plotting, but cannot carry immutable replay identity.
            callout = app.query_one("#backfill-callout", Callout)
            assert callout.content.problem == message_id("profiling.view.csv_problem")
            assert str(self.csv_path) in callout.content.detail
            assert app.query_one("#open-result-profiling", Button).disabled

            assert await pilot.click("#nav-results-tab")
            await pilot.pause()
            if plot_entry == "button":
                for language, label in (("zh", "生成 PNG 图表"), ("en", "Generate PNG")):
                    app.configuration.preferences = replace(app.configuration.preferences, language=language)
                    app.configuration.apply_ui_preferences()
                    for size in ((80, 24), (120, 30), (150, 45), (80, 24)):
                        await pilot.resize_terminal(*size)
                        await pilot.pause()
                        buttons = [app.query_one(selector, Button) for selector in (
                            "#refresh-experiment-state", "#report-calculate", "#plot-results")]
                        for button in buttons:
                            assert button.region.area > 0
                            assert button.content_region.width >= cell_len(str(button.label))
                            assert app.query_one("#result-actions").content_region.contains_region(button.region)
                            assert app.get_widget_at(*button.region.center)[0] is button
                        assert all(left.region.right <= right.region.x for left, right in zip(buttons, buttons[1:], strict=False))
                        assert str(buttons[-1].label) == label
                        buttons[-2].focus()
                        await pilot.press("tab")
                        assert app.focused is buttons[-1]
                        await pilot.press("shift+tab")
                        assert app.focused is buttons[-2]
            with patch.object(app.process, "execute_command") as execute:
                # Both visible and command entries reuse the selected experiment.
                if plot_entry == "button":
                    assert await pilot.click("#plot-results")
                else:
                    field = app.query_one("#slash-command", Input)
                    field.value = "/plot"
                    field.focus()
                    await pilot.press("enter")
                await pilot.pause()
                execute.assert_called_once()
                command, kind = execute.call_args.args
                assert kind == "plot"
                assert command[1:5] == ["-u", "-m", "acprof", "plot"]
                assert command[5] == str(self.csv_path)
                assert tabs.active == "monitor-tab"
                assert all(widget.disabled for widget in app.query(
                    "#refresh-experiment-state, #plot-results, #open-result-profiling"))
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
            assert "有效正式窗口：1" in app.query_one("#result-data-status", Static).content
            assert summary.region.height > 0
            assert not app.query_one("#plot-results", Button).disabled
