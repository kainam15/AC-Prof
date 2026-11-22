"""VHS fixture scenes must initialize even when a terminal skips focus bubbling."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pytest
from textual.widgets import Button, TabbedContent, TabPane

from tests.visual_png.fixture_app import VisualFixtureApp, isolate_fixture

pytestmark = pytest.mark.visual_png


async def test_docker_scene_ready_without_descendant_focus(tmp_path) -> None:
    class NoFocusFixture(VisualFixtureApp):
        def on_descendant_focus(self, event) -> None:
            pass

    with isolate_fixture() as guard:
        app = NoFocusFixture("docker-frontend-small", settings_path=tmp_path / "settings.json")
        activate = app.activate_tab
        activations = []

        def activate_with_initial_focus(page):
            activate(page)
            if not activations:
                # A focus notification from the initial form may still be queued.
                initial = app.query_one("#run-tab", TabPane)
                initial.post_message(TabPane.Focused(initial))
            activations.append(page)

        app.activate_tab = activate_with_initial_focus
        restore = Mock(wraps=app.results.restore_recent_result)
        app.results.restore_recent_result = restore
        mount = app.scenario.mount
        navigation = []

        def record_mount(fixture):
            navigation.append((
                fixture.query_one("#main-tabs", TabbedContent).active,
                fixture.query_one("#nav-monitor-tab", Button).has_class("selected-page"),
                fixture.query_one("#monitor-tab").region.area > 0,
            ))
            mount(fixture)

        app.scenario = replace(app.scenario, mount=record_mount)
        async with app.run_test(size=(80, 24)) as pilot:
            await app.wait_for_scene(pilot)
            await pilot.pause()
            restore.assert_called_once_with()
            assert activations.count("monitor-tab") >= 2
            assert navigation == [("monitor-tab", True, True)]
            assert app.query_one("#main-tabs", TabbedContent).active == "monitor-tab"
            assert app.query_one("#download-progress").display
            assert not app.query_one("#stop-run", Button).disabled
    assert not guard.blocked_calls


@pytest.mark.parametrize("scene", ["form", "docker-frontend"])
async def test_scene_refresh_uses_the_terminal_app_driver(tmp_path, scene) -> None:
    with isolate_fixture() as guard:
        app = VisualFixtureApp(scene, settings_path=tmp_path / "settings.json")

        async def check_scene(pilot):
            try:
                await app.wait_for_scene(pilot)
                assert app.query_one("#main-tabs", TabbedContent).active == app.scenario.page
            finally:
                app.exit()

        await app.run_async(headless=True, auto_pilot=check_scene,
                            size=(app.scenario.columns, app.scenario.rows))
    assert guard.closed
