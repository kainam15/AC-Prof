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


@pytest.mark.parametrize(("language", "size"), [("zh", (80, 24)), ("en", (80, 24)), ("en", (120, 30)), ("zh", (150, 45))])
async def test_only_selected_activity_toggles_sidebar(tmp_path, language, size):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size) as pilot:
        tabs = app.query_one("#main-tabs", TabbedContent)
        navigation = app.query_one("#page-navigation")
        model = app.query_one("#model", Input)
        model.value = "demo/edited-model"
        await pilot.pause()
        expanded_width = navigation.region.width
        content_width = tabs.region.width
        labels = {button.id: str(button.label) for button in navigation.query(Button)}

        assert await pilot.click("#nav-run-tab")
        await pilot.pause()
        assert navigation.region.width < expanded_width
        collapsed_width = navigation.region.width
        assert tabs.region.width > content_width
        assert str(tabs.active) == "run-tab"
        assert app.query_one("#model") is model
        assert model.value == "demo/edited-model"
        assert app.has_class("narrow") == (tabs.region.width < 110)
        for button in navigation.query(Button):
            assert cell_len(str(button.label)) == 1
            tooltip, pane_id = button.tooltip, button.name
            assert isinstance(tooltip, str)
            assert pane_id is not None
            assert str(tabs.get_tab(pane_id).label) in tooltip
            assert button.region.area > 0
            assert app.get_widget_at(*button.region.center)[0] is button
        assert list(navigation.query(".selected-page")) == [app.query_one("#nav-run-tab")]

        assert await pilot.click("#nav-run-tab")
        await pilot.pause()
        assert navigation.region.width == expanded_width
        assert tabs.region.width == content_width
        assert {button.id: str(button.label) for button in navigation.query(Button)} == labels

        # Enter toggles the focused current activity, just like a mouse click.
        await pilot.press("enter")
        await pilot.pause()
        assert navigation.region.width < expanded_width
        await pilot.press("down", "enter")
        await pilot.pause()
        assert str(tabs.active) == "monitor-tab"
        assert navigation.region.width == collapsed_width
        assert app.focused is app.query_one("#nav-monitor-tab")
        assert list(navigation.query(".selected-page")) == [app.query_one("#nav-monitor-tab")]

        await pilot.press("enter")
        await pilot.pause()
        assert navigation.region.width == expanded_width
        assert await pilot.click("#nav-monitor-tab")
        await pilot.pause()
        assert navigation.region.width == collapsed_width
        assert await pilot.click("#nav-run-tab")
        await pilot.pause()
        assert navigation.region.width == collapsed_width
        assert str(tabs.active) == "run-tab"
        assert model.value == "demo/edited-model"
        assert await pilot.click("#nav-run-tab")
        await pilot.pause()
        assert navigation.region.width == expanded_width
        assert await pilot.click("#nav-monitor-tab")
        await pilot.pause()
        assert str(tabs.active) == "monitor-tab"
        assert navigation.region.width == expanded_width


async def test_collapsed_sidebar_survives_language_resize_and_programmatic_navigation(tmp_path):
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(120, 30)) as pilot:
        tabs = app.query_one("#main-tabs", TabbedContent)
        navigation = app.query_one("#page-navigation")
        assert await pilot.click("#nav-run-tab")
        await pilot.pause()
        collapsed_width = navigation.region.width
        assert cell_len(str(app.query_one("#nav-run-tab", Button).label)) == 1

        app.activate_tab("monitor-tab")
        await pilot.pause()
        await pilot.press("f2")
        await pilot.pause()
        assert tabs.active == "settings-tab"
        assert navigation.region.width == collapsed_width
        assert app.focused is app.query_one("#nav-settings-tab")
        app.configuration.preferences = replace(app.configuration.preferences, language="en")
        app.configuration.apply_ui_preferences()
        await pilot.pause()
        tooltip = app.query_one("#nav-settings-tab").tooltip
        assert isinstance(tooltip, str)
        assert "Settings" in tooltip
        for size in ((80, 24), (150, 45), (120, 30)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            assert navigation.region.width == collapsed_width
            assert app.has_class("narrow") == (tabs.region.width < 110)
            for button in navigation.query(Button):
                assert cell_len(str(button.label)) == 1
                assert button.region.bottom <= app.query_one("#bottom-panel").region.y
                assert app.get_widget_at(*button.region.center)[0] is button

        assert await pilot.click("#nav-settings-tab")
        await pilot.pause()
        assert navigation.region.width > collapsed_width
        assert str(app.query_one("#nav-run-tab", Button).label) == "≡  Experiment"
        await pilot.press("tab", "shift+tab")
        assert app.focused is app.query_one("#nav-settings-tab")


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
