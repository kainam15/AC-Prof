"""VHS fixture scenes must initialize even when a terminal skips focus bubbling."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pytest
from textual.widgets import Button, TabbedContent

from tests.visual_png.fixture_app import VisualFixtureApp, isolate_fixture

pytestmark = pytest.mark.visual_png


async def test_docker_scene_ready_without_descendant_focus(tmp_path) -> None:
    class NoFocusFixture(VisualFixtureApp):
        def on_descendant_focus(self, event) -> None:
            pass

    with isolate_fixture() as guard:
        app = NoFocusFixture("docker-frontend-small", settings_path=tmp_path / "settings.json")
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
            assert navigation == [("monitor-tab", True, True)]
            assert app.query_one("#main-tabs", TabbedContent).active == "monitor-tab"
            assert app.query_one("#download-progress").display
            assert not app.query_one("#stop-run", Button).disabled
    assert not guard.blocked_calls
