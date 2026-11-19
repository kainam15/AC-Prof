"""Check the visible text VHS waits for and the geometry of selected controls."""

from __future__ import annotations

import asyncio
import re
from itertools import combinations
from pathlib import Path

import pytest
from textual.widgets import Button, Static

from acprof.tui.ui.structured import StatusBanner
from tests.visual_png.fixture_app import VisualFixtureApp, isolate_fixture
from tests.visual_png.scenarios import SCENARIOS

pytestmark = pytest.mark.visual_png


@pytest.mark.parametrize("scene", tuple(SCENARIOS))
async def test_scene_shows_vhs_marker_and_fits_controls(tmp_path: Path, scene: str) -> None:
    spec = SCENARIOS[scene]
    with isolate_fixture() as isolation:
        app = VisualFixtureApp(scene, settings_path=tmp_path / "settings.json")
        async with app.run_test(size=(spec.columns, spec.rows)) as pilot:
            await asyncio.wait_for(app.scene_ready.wait(), timeout=5)
            await app.workers.wait_for_complete()
            await pilot.pause()
            visible_text = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
            assert re.search(spec.ready_pattern, visible_text), (
                f"{scene}: missing VHS marker {spec.ready_pattern!r}"
            )
            controls = [app.screen.query_one(selector) for selector in spec.visible_controls]
            if scene.startswith("docker-frontend"):
                stage = app.execution.session.snapshot.stage
                banner = app.query_one("#monitor-summary", StatusBanner)
                assert banner.summary.headline == stage.label
                assert banner.summary.level == "info"
                assert str(banner.query_one(".status-headline", Static).content) == f"… {app.tr(stage.label)}"
                assert not app.query_one("#stop-run", Button).disabled
                assert not app.execution.session.snapshot.measurement_active
            state = {"main": app.query_one("#main-tabs").active,
                     "results": app.query_one("#result-views").active,
                     "focused": str(app.focused), "pane_display": app.query_one("#results-tab").display}
            for control in controls:
                region = control.region
                assert region.width > 0, f"{scene}: invisible {control.id}; {state}"
                assert region.height > 0, f"{scene}: invisible {control.id}"
                assert region.x >= 0, f"{scene}: negative {control.id}"
                assert region.y >= 0, f"{scene}: negative {control.id}"
                assert region.right <= spec.columns, f"{scene}: {control.id} extends right: {region}"
                assert region.bottom <= spec.rows, f"{scene}: {control.id} extends down: {region}"
            # Buttons must not occupy the same clickable cells.
            buttons = [control for control in controls if isinstance(control, Button)]
            for left, right in combinations(buttons, 2):
                a, b = left.region, right.region
                assert a.right <= b.x or b.right <= a.x or a.bottom <= b.y or b.bottom <= a.y, (
                    f"{scene}: buttons overlap: {left.id}, {right.id}"
                )
    assert isolation.blocked_calls == []
