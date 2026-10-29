"""Failure details behave like logs while recovery actions remain reachable."""
from unittest.mock import patch

import pytest
from rich.cells import cell_len
from textual.widgets import Button, Collapsible, Select, Tooltip

from acprof.experiment import RunConfig
from acprof.tui.log import SelectableLog
from acprof.tui.preparation import PreparationScreen
from tests.support.tui_fixtures import AcprofTui


@pytest.mark.parametrize(("size", "language"), [
    ((80, 24), "zh"), ((80, 24), "en"), ((120, 30), "en"), ((150, 45), "zh"),
])
async def test_failure_details_scroll_copy_resize_and_keep_actions_visible(tmp_path, size, language):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    detail = "Traceback (most recent call last):\n" + "\n".join(
        f'  File "/models/中文模型/runtime.py", line {line}, in prepare [literal]'
        for line in range(80)
    ) + "\nRuntimeError: 构建期间代码或依赖配置发生变化"
    replies = []
    async with app.run_test(size=size) as pilot:
        app.query_one("#ui-language", Select).value = language
        await pilot.pause()
        screen = PreparationScreen({"stage": "image", "request": {
            "id": 1, "kind": "error", "detail": detail,
        }}, respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        assert not screen.query("#preparation-show-diagnostics")
        details = screen.query_one("#preparation-diagnostics", Collapsible)
        assert not details.collapsed
        log = screen.query_one("#preparation-traceback", SelectableLog)
        assert log.text == detail
        assert log.read_only
        actions = list(screen.query("#preparation-actions Button"))
        assert [button.id for button in actions] == [
            "preparation-cancel", "preparation-continue", "preparation-rebuild",
        ]
        for resized in (size, (80, 24), (150, 45)):
            await pilot.resize_terminal(*resized)
            await pilot.pause()
            assert log.content_region.height >= 3
            assert actions[-1].region.right == screen.query_one("#preparation-actions").content_region.right
            for button in actions:
                assert button.region.height == 3
                assert cell_len(str(button.label)) <= button.content_region.width
                assert screen.region.contains_region(button.region)
                assert log.region.bottom <= button.region.y
                assert app.get_widget_at(*button.region.center)[0] is button
            log.focus()
            await pilot.press("home")
            await pilot.pause()
            assert log.scroll_y == 0
            await pilot.press("end")
            await pilot.pause()
            assert log.scroll_y > 0
            assert log.scroll_y == log.max_scroll_y

        with patch.object(app, "copy_to_clipboard") as copy:
            await pilot.press("ctrl+a", "ctrl+c")
            assert log.selected_text == detail
            copy.assert_called_once_with(detail)
        await pilot.press("x")
        assert log.text == detail
        await pilot.press("tab")
        assert app.focused is actions[0]
        await pilot.press("tab", "tab")
        assert app.focused is actions[2]

        assert await pilot.click(details.query_one("CollapsibleTitle"))
        await pilot.pause()
        assert details.collapsed
        assert await pilot.click(actions[2])
        await pilot.pause()
        assert replies == [{"action": "rebuild"}]


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_failure_button_tooltips_explain_actions_above_footer(tmp_path, language):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    app.TOOLTIP_DELAY = 0.01
    replies = []
    async with app.run_test(size=(80, 24), tooltips=True) as pilot:
        app.query_one("#ui-language", Select).value = language
        await pilot.pause()
        screen = PreparationScreen({"stage": "runtime", "request": {
            "id": 1, "kind": "error", "detail": "RuntimeError: invalid output",
        }}, respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        expected = {
            "preparation-cancel": ("确认", "confirmation"),
            "preparation-continue": ("失败阶段", "failed stage"),
            "preparation-rebuild": ("构建缓存", "build cache"),
        }
        tooltip = screen.query_one(Tooltip)
        for identifier, words in expected.items():
            button = screen.query_one("#" + identifier, Button)
            for row in range(button.region.height):
                assert await pilot.hover("#preparation-title")
                await pilot.pause()
                assert await pilot.hover(button, offset=(button.region.width // 2, row))
                await pilot.pause(app.TOOLTIP_DELAY + 0.05)
                assert tooltip.display
                assert words[language == "en"] in str(tooltip.content)
                assert screen.region.contains_region(tooltip.region)
                assert tooltip.region.bottom <= button.region.y
                assert app.get_widget_at(*button.region.center)[0] is button
        assert not replies  # Hovering must never initiate preparation or cancellation.
