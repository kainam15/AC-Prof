"""页面滚动、缩放和交互状态不能改变操作栏位置或颜色含义。"""
import colorsys
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from rich.cells import cell_len
from textual.containers import VerticalScroll
from textual.widgets import Button, Static
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.stages import Stage
from acprof.tui.themes import UI_THEMES


class TestTuiPageChrome:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        temporary = tmp_path
        self.app = AcprofTui(
            RunConfig.smoke("demo/model"),
            settings_path=Path(str(temporary)) / "tui.json",
        )
        refresh = patch.object(self.app.images, 'refresh_images')
        refresh.start()
        self._request.addfinalizer(partial(refresh.stop))

    def assert_reachable(self, button):
        region = button.region
        assert (region.width) > (0), button.id
        assert (region.height) == (3), button.id
        assert (cell_len(button.label.plain)) <= (button.content_region.width), button.id
        center_x, center_y = region.center
        for x in (region.x + 1, center_x, region.right - 2):
            assert (self.app.get_widget_at(x, center_y)[0]) is (button), button.id

    def assert_actions_aligned(self, pane, secondary_ids, primary):
        secondary = [pane.query_one("#" + widget_id) for widget_id in secondary_ids]
        assert (secondary[0].region.x) == (pane.region.x + 2)
        for left, right in zip(secondary, secondary[1:], strict=False):
            assert (right.region.x) == (left.region.right + 1), right.id
        for action in secondary:
            assert (action.region.y) == (primary.region.y), action.id
        assert (primary.region.x) >= (secondary[-1].region.right + 1)

    @pytest.mark.parametrize(("page", "primary", "secondary"), [('run-tab', 'start-run', ('open-run-settings', 'probe-largest')), ('monitor-tab', 'stop-run', ('copy-log-all', 'follow-log', 'expand-log', 'clear-log')), ('results-tab', 'plot-results', ('summarize-results',)), ('images-tab', 'image-delete', ('image-toggle', 'image-model', 'image-clear')), ('settings-tab', 'save-ui-settings', ('restore-ui-defaults',))])
    @pytest.mark.parametrize('language', ['zh', 'en'])
    @pytest.mark.parametrize('size', [(80, 24), (120, 30), (150, 45)])
    async def test_all_pages_keep_top_content_and_rightmost_actions_fixed_when_scrolling_and_resizing(self, page, primary, secondary, language, size):
        app = self.app
        async with app.run_test(size=(150, 45)) as pilot:
            await pilot.resize_terminal(*size)
            app.configuration.preferences = replace(app.configuration.preferences, language=language)
            app.configuration.apply_ui_preferences()
            app.activate_tab(page)
            await pilot.pause()
            pane = app.query_one("#" + page)
            button = app.query_one("#" + primary, Button)
            self.assert_reachable(button)
            assert (button.region.bottom) == (pane.region.bottom)
            assert (button.region.right) == (pane.region.right - 2)
            self.assert_actions_aligned(pane, secondary, button)
            top = pane.query_one(".page-header")
            top_region, button_region = top.region, button.region
            assert (top_region.y) == (pane.region.y)
            for scroll in pane.query(VerticalScroll):
                if scroll.display and scroll.region.height:
                    scroll.scroll_end(animate=False, immediate=True)
            await pilot.pause()
            assert (top.region) == (top_region)
            assert (button.region) == (button_region)
            self.assert_actions_aligned(pane, secondary, button)
            for action in pane.query(".action-bar Button"):
                if action.display:
                    self.assert_reachable(action)
            # Footer remains anchored when the global command box is hidden.
            app.configuration.preferences = replace(app.configuration.preferences, show_command_bar=False)
            app.configuration.apply_ui_preferences()
            await pilot.pause()
            assert (button.region.bottom) == (pane.region.bottom)
            self.assert_reachable(button)
            self.assert_actions_aligned(pane, secondary, button)
            app.configuration.preferences = replace(app.configuration.preferences, show_command_bar=True)
            app.configuration.apply_ui_preferences()

    def assert_color_family(self, color, family):
        hue, saturation, _ = colorsys.rgb_to_hsv(*(channel / 255 for channel in color.rgb))
        if family == "gray":
            assert (saturation) < (0.15)
        else:
            assert (saturation) > (0.2)
            low, high = {"cyan": (0.46, 0.56), "yellow": (0.10, 0.18),
                         "green": (0.35, 0.46), "red": (0.95, 1.0)}[family]
            assert (hue) >= (low)
            assert (hue) <= (high)

    @pytest.mark.parametrize(("page", "selector", "family"), [('run-tab', '#start-run', 'cyan'), ('images-tab', '#image-delete', 'red')])
    @pytest.mark.parametrize("theme", UI_THEMES)
    async def test_action_semantics_survive_theme_hover_focus_and_disabled_states(self, page, selector, family, theme):
        app = self.app
        async with app.run_test(size=(120, 30)) as pilot:
            app.theme = theme
            await pilot.pause()
            app.activate_tab(page)
            button = app.query_one(selector, Button)
            button.disabled = False
            await pilot.pause()
            original = button.styles.color
            self.assert_color_family(original, family)
            assert (await pilot.hover(button))
            await pilot.pause()
            assert (button.styles.color) == (original)
            button.focus()
            await pilot.pause()
            assert (button.styles.color) == (original)
            assert (button.styles.text_style.underline)
            assert not (button.styles.text_style.reverse)
            assert (button.styles.background.a) == (0)
            button.disabled = True
            await pilot.pause()
            self.assert_color_family(button.styles.color, "gray")
            assert not (button.styles.text_style.underline)
            button.disabled = False
            await pilot.pause()
            assert (button.styles.color) == (original)
            app.activate_tab("results-tab")
            await pilot.pause()
            normal = app.query_one("#summarize-results", Button)
            normal.disabled = False
            await pilot.pause()
            foreground = app.current_theme.foreground
            assert (normal.styles.color.hex) == (foreground.upper())
            await pilot.hover(normal)
            normal.focus()
            await pilot.pause()
            assert (normal.styles.color.hex) == (foreground.upper())

    @pytest.mark.parametrize(("stage", "family"), [(Stage.SERVER_READY, 'green'), (Stage.COMPLETED, 'green'), (Stage.MEASURING, 'cyan'), (Stage.FAILED, 'red')])
    async def test_ready_success_and_failure_feedback_have_semantic_colors(self, stage, family):
        app = self.app
        async with app.run_test(size=(80, 24)) as pilot:
            app.activate_tab("monitor-tab")
            app.render_snapshot(ProgressSnapshot(stage=stage))
            await pilot.pause()
            self.assert_color_family(app.query_one("#status-stage").styles.color, family)
            app.activate_tab("settings-tab")
            await pilot.pause()
            assert (await pilot.click("#save-ui-settings"))
            await pilot.pause()
            status = app.query_one("#settings-status", Static)
            assert ("已保存") in (str(status.content))
            self.assert_color_family(status.styles.color, "green")
            with patch("acprof.tui.configuration_controller.save_settings", side_effect=OSError("read only")):
                assert (await pilot.click("#save-ui-settings"))
                await pilot.pause()
            assert ("保存失败") in (str(status.content))
            self.assert_color_family(status.styles.color, "red")

    async def test_confirmation_keeps_the_action_color_and_cancel_does_not_launch(self):
        app = self.app
        with patch.object(app, 'launch') as launch:
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.press("f5")
                await pilot.pause()
                launch.assert_called_once()
                launch.reset_mock()
                app.profiles.request_profile_run(
                    result_dir=str(app.configuration.settings_path.parent), tools="torch")
                await pilot.pause()
                self.assert_color_family(app.screen.query_one("#confirm-yes").styles.color, "yellow")
                await pilot.press("escape")
                app.process_kind = "run"
                process = Mock()
                process.poll.return_value = None
                app.lifecycle.process = process
                try:
                    app.set_busy(True)
                    app.action_request_stop()
                    await pilot.pause()
                    self.assert_color_family(app.screen.query_one("#confirm-yes").styles.color, "red")
                    await pilot.press("escape")
                finally:
                    app.lifecycle.process = None
                launch.assert_not_called()
                app.process_kind = ""
                app.set_busy(False)
