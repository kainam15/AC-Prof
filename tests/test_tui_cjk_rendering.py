"""中文浮层在最终屏幕与局部终端输出中保持完整。"""
import tempfile
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.segment import Segment
from rich.text import Text
from textual.geometry import Region, Size
from textual.strip import Strip
from textual.widgets._toast import Toast
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.tui.views import ConfirmActionScreen


class TestCjkComposition:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        temporary = tmp_path
        self.app = AcprofTui(settings_path=Path(str(temporary)) / "tui.json")
        self.compositor = self.app.get_default_screen()._compositor
        self.compositor.size = Size(12, 1)

    def configure_layers(self, layers):
        viewport = self.compositor.size.region
        self.compositor._visible_widgets = {
            object(): (region, viewport) for region, _ in layers
        }
        self.compositor._cuts = None
        renders = [(region, viewport, [Strip([Segment(text)])]) for region, text in layers]
        mocked = patch.object(self.compositor, "_get_renders", return_value=renders)
        mocked.start()
        self._request.addfinalizer(partial(mocked.stop))

    def test_hidden_widget_edges_do_not_erase_double_width_characters(self):
        self.configure_layers([
            (Region(1, 0, 10, 1), "甲乙丙丁戊"),
            (Region(4, 0, 3, 1), "TOP"),
            (Region(0, 0, 12, 1), "." * 12),
        ])
        assert (self.compositor.render_strips()[0].text) == (".甲乙丙丁戊.")

    def test_real_occlusion_still_clips_the_underlying_wide_character(self):
        self.configure_layers([
            (Region(4, 0, 3, 1), "TOP"),
            (Region(1, 0, 10, 1), "甲乙丙丁戊"),
            (Region(0, 0, 12, 1), "." * 12),
        ])
        assert (self.compositor.render_strips()[0].text) == (".甲 TOP丁戊.")

    @pytest.mark.parametrize('column', (3, 4))
    def test_partial_update_emits_the_whole_character_at_either_half(self, column):
        self.configure_layers([
            (Region(1, 0, 10, 1), "甲乙丙丁戊"),
            (Region(4, 0, 3, 1), "TOP"),
            (Region(0, 0, 12, 1), "." * 12),
        ])
        self.compositor._dirty_regions = {Region(column, 0, 1, 1)}
        update = self.compositor.render_partial_update()
        assert (update) is not None
        assert ("乙") in (Text.from_ansi(update.render_segments(self.app.console)).plain)
        segments = self.app.console.render(update)
        assert ("乙") in ("".join(segment.text for segment in segments if not segment.control))
        assert (self.compositor.render_partial_update()) is None

    def test_clipped_and_empty_foreground_rows_do_not_crash_or_shift_text(self):
        self.configure_layers([
            (Region(-2, 0, 10, 1), "甲乙丙丁"),
            (Region(3, 0, 3, 1), "TOP"),
            (Region(0, 0, 12, 1), "." * 12),
        ])
        assert (self.compositor.render_strips()[0].text) == ("甲乙丙丁....")
        self.compositor._get_renders.return_value = [(Region(0, 0, 0, 1), Region(0, 0, 12, 1), [Strip([])])]
        assert (self.compositor.render_strips()[0].text) == ("")


@pytest.mark.parametrize('size', ((80, 24), (81, 24), (120, 30), (121, 30), (150, 45), (151, 45)))
@pytest.mark.parametrize('language,expected', (('zh', '已恢复完整默认配置'), ('en', 'Full defaults restored')))
async def test_toast_survives_button_borders_language_changes_and_resize(size, language, expected):
    with tempfile.TemporaryDirectory() as directory:
        app = AcprofTui(
            RunConfig.smoke("demo/model"), settings_path=Path(directory) / "tui.json",
        )
        with patch.object(app, "refresh_images"):
            async with app.run_test(size=(80, 24), notifications=True) as pilot:
                await pilot.pause()
                app.clear_notifications()
                app.ui_preferences = replace(
                    app.ui_preferences, language=language, show_command_bar=False,
                )
                app._apply_ui_preferences()
                await pilot.pause()
                app.notify("已恢复完整默认配置", timeout=60)
                await pilot.pause()
                await pilot.resize_terminal(*size)
                await pilot.pause()
                toast = app.query_one(Toast)
                lines = app.screen._compositor.render_strips()
                rendered = "\n".join(lines[y].text for y in toast.region.line_range)
                assert (expected) in (rendered)
                assert (await pilot.click(app.query_one(Toast)))
                await pilot.pause()
                assert not (app.query(Toast))

@pytest.mark.parametrize('size', ((81, 24), (120, 30), (150, 45)))
async def test_titled_notifications_on_confirmation_screen_keep_text_and_cancel(size):
    with tempfile.TemporaryDirectory() as directory:
        app = AcprofTui(settings_path=Path(directory) / "tui.json")
        with patch.object(app, "_launch") as launch:
            async with app.run_test(size=(81, 24), notifications=True) as pilot:
                await pilot.pause()
                app.clear_notifications()
                await app.push_screen(ConfirmActionScreen("开始采集", "检查配置后再开始采集"))
                app.notify("请等待当前任务完成", title="设置未保存", severity="error", timeout=60)
                await pilot.pause()
                await pilot.resize_terminal(*size)
                await pilot.pause()
                text = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
                assert ("设置未保存") in (text)
                assert ("请等待当前任务完成") in (text)
                await pilot.press("escape")
                await pilot.pause()
                assert not isinstance(app.screen, ConfirmActionScreen)
                launch.assert_not_called()
