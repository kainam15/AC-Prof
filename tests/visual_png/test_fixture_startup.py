"""VHS fixture scenes must initialize even when a terminal skips focus bubbling."""

from __future__ import annotations

import asyncio

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
        async with app.run_test(size=(80, 24)) as pilot:
            await asyncio.wait_for(app.scene_ready.wait(), timeout=8)
            await pilot.pause()
            assert app.scene_started
            assert app.query_one("#main-tabs", TabbedContent).active == "monitor-tab"
            assert app.query_one("#download-progress").display
            assert not app.query_one("#stop-run", Button).disabled
    assert not guard.blocked_calls
