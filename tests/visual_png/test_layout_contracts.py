"""Geometry checks complement PNG diffs: a reviewed baseline can also be wrong."""

from __future__ import annotations

from itertools import combinations

import pytest
from textual.widgets import Button

from tests.visual_png.fixture_app import VisualFixtureApp
from tests.visual_png.scenarios import SCENARIOS

pytestmark = pytest.mark.visual_png


@pytest.mark.parametrize(
    ("scene", "selectors"),
    [
        ("form-small", ("#model", "#start-run")),
        ("error-modal", ("#preflight-close", "#preflight-retry")),
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
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        controls = [app.screen.query_one(selector) for selector in selectors]
        for control in controls:
            region = control.region
            assert region.width > 0, f"{scene}: invisible {control.id}"
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
