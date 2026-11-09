"""Unrelated UI tests use a ready host without running startup hardware probes.

The startup suite imports the production app and exercises its real workers.
This fixture preserves the startup state machine and configuration invalidation.
"""

from contextlib import asynccontextmanager
from unittest.mock import patch

from textual.widgets import Checkbox

from acprof.tui.app import AcprofTui as ProductionTui
from acprof.tui.ui.structured import FactGrid, FactValue

# Keep the reviewed Golden header stable across releases, only in visual fixtures.
SNAPSHOT_TITLE = "AC-Prof 0.3.0"


def checkbox_text(checkbox: Checkbox) -> str:
    """Read visible terminal cells, including the mark, without relying on color."""
    region = checkbox.content_region
    assert region.width > 0
    assert region.height > 0
    assert 0 <= region.x < region.right <= checkbox.screen.size.width
    assert 0 <= region.y < region.bottom <= checkbox.screen.size.height
    lines = checkbox.screen._compositor.render_strips()
    return "\n".join(line.crop(region.x, region.right).text for line in lines[region.y:region.bottom])


def fact_rows(grid: FactGrid) -> tuple[tuple[str, str], ...]:
    """Read the localized labels and values rendered by a structured fact grid."""
    return tuple(
        (str(label.content), value.render().plain)
        for label, value in zip(grid.query(".fact-label"), grid.query(FactValue), strict=True)
    )


class AcprofTui(ProductionTui):
    CSS_PATH = ProductionTui.CSS_PATH

    @asynccontextmanager
    async def run_test(self, *args, **kwargs):
        with patch.object(
            self.preflight, "execute_quick_check",
            side_effect=lambda config, token: self.preflight.show_quick_check([], "", token),
        ):
            async with super().run_test(*args, **kwargs) as pilot:
                await pilot.pause()
                yield pilot
