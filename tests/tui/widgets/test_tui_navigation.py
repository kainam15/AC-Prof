"""Sidebar navigation keeps mounted page state and header actions reachable."""
from dataclasses import replace

import pytest
from rich.cells import cell_len
from textual.widgets import Button, Input, TabbedContent

from acprof.experiments.configuration import RunConfig
from tests.support.tui_fixtures import AcprofTui


@pytest.mark.parametrize(("language", "size"), [("zh", (80, 24)), ("en", (80, 24)), ("en", (120, 30)), ("zh", (150, 45))])
async def test_sidebar_clicks_keyboard_and_resize_preserve_form(tmp_path, monkeypatch, language, size):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language, show_command_bar=True)
    monkeypatch.setattr("acprof.tui.images.controller.ImageController.refresh_images", lambda self: None)
    async with app.run_test(size=size) as pilot:
        tabs = app.query_one("#main-tabs", TabbedContent)
        model = app.query_one("#model", Input)
        model.value = "demo/edited-model"
        await pilot.pause()
        for page in ("run", "monitor", "results", "backfill", "images", "settings"):
            button = app.query_one(f"#nav-{page}-tab", Button)
            assert button.region.right <= tabs.region.x
            assert button.region.bottom <= app.query_one("#bottom-panel").region.y
            assert cell_len(button.label.plain) <= button.content_region.width
            label_region = button.content_region
            lines = [strip.crop(label_region.x, label_region.right).text for strip in
                     app.screen._compositor.render_strips()[label_region.y:label_region.bottom]]
            assert any(button.label.plain in line for line in lines), (button.label.plain, lines)
            assert app.get_widget_at(*button.region.center)[0] is button
            assert await pilot.click(button)
            await pilot.pause()
            assert tabs.active == f"{page}-tab"
            assert list(app.query("#page-navigation .selected-page")) == [button]

        await pilot.press("up", "enter")
        assert tabs.active == "images-tab"
        await pilot.press("down", "enter")
        assert tabs.active == "settings-tab"
        app.activate_tab("run-tab")
        await pilot.pause()
        assert model.value == "demo/edited-model"
        assert app.focused is app.query_one("#nav-run-tab")
        await pilot.press("f2")
        assert tabs.active == "settings-tab"
        assert app.query_one("#nav-settings-tab").has_class("selected-page")
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert await pilot.click("#nav-run-tab")
        assert model.value == "demo/edited-model"


async def test_environment_is_beside_quit_and_keeps_overlay_below_header(tmp_path):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        environment = app.query_one("#current-environment")
        quit_button = app.query_one("#quit-app")
        assert environment.region.y == quit_button.region.y == 0
        assert environment.region.right == quit_button.region.x
        assert app.get_widget_at(*quit_button.region.center)[0] is quit_button
        assert await pilot.click("#environment-error")
        panel = app.preflight.panel
        assert panel.display
        assert panel.pinned
        assert panel.region.y == 1
        assert panel.region.right <= quit_button.region.x
        await pilot.resize_terminal(120, 30)
        await pilot.pause()
        assert panel.region.right == quit_button.region.x
        await pilot.press("escape")
        assert not panel.display
