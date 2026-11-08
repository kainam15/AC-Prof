"""Geometry checks complement PNG diffs: a reviewed baseline can also be wrong."""

from __future__ import annotations

import asyncio
from itertools import combinations

import pytest
from textual.widgets import Button, Static

from tests.visual_png.fixture_app import VisualFixtureApp
from tests.visual_png.scenarios import SCENARIOS

pytestmark = pytest.mark.visual_png


@pytest.mark.parametrize(
    ("scene", "selectors"),
    [
        ("form-small", ("#model", "#start-run")),
        ("error-modal", ("#preflight-inspection", "#preflight-retry")),
        ("statistics", ("#experiment-source", "#refresh-experiment-state", "#report-calculate")),
        ("statistics-en", ("#experiment-source", "#refresh-experiment-state", "#report-calculate")),
        ("docker-frontend", ("#download-progress", "#stop-run")),
        ("docker-frontend-small", ("#download-progress", "#stop-run")),
        ("docker-frontend-en", ("#download-progress", "#stop-run")),
        ("docker-frontend-en-small", ("#download-progress", "#stop-run")),
        (
            "task-selection",
            ("#preparation-cancel", "#preparation-select-all", "#preparation-apply"),
        ),
    ],
)
async def test_visible_controls_fit_without_overlap(
    tmp_path, scene: str, selectors: tuple[str, ...]
) -> None:
    spec = SCENARIOS[scene]
    app = VisualFixtureApp(scene, settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(spec.columns, spec.rows)) as pilot:
        await asyncio.wait_for(app.scene_ready.wait(), timeout=5)
        await app.workers.wait_for_complete()
        await pilot.pause()
        controls = [app.screen.query_one(selector) for selector in selectors]
        if scene.startswith("docker-frontend"):
            stage = app.execution.session.snapshot.stage
            assert str(app.query_one("#status-stage", Static).content) == app.localization.tr(stage.label)
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
