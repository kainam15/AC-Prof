"""Unrelated UI tests use a ready host without running startup hardware probes.

The startup suite imports the production app and exercises its real workers.
This fixture preserves the startup state machine and configuration invalidation.
"""

from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from textual.widgets import Checkbox

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui as ProductionTui
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.ui.structured import FactGrid, FactValue
from tests.support.tui_waits import wait_for_condition

# Keep the reviewed Golden header stable across releases, only in visual fixtures.
SNAPSHOT_TITLE = "AC-Prof 0.3.0"


def write_interrupted_preparation(config: RunConfig) -> Path:
    """Create a failed preparation through the run-state writer in an isolated fixture."""
    from acprof.experiments.arguments import build_parser
    from acprof.experiments.configuration import build_run_command
    from acprof.host import run_state

    project = Path.cwd()
    command = build_run_command(config, project_dir=project)
    args = build_parser().parse_args(command[command.index("--model"):])
    for name, value in asdict(RunConfig.from_namespace(args).validate(project_dir=project)).items():
        setattr(args, name, value)
    directory = config.result_dir(project)
    state = run_state.RunState(directory, run_state.run_options(args), resume=False, project_dir=project)
    state.close("failed")
    return directory


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
            side_effect=lambda config, token: self.preflight.show_quick_check(
                [PreflightCheck('Fixture environment', 'ok', 'Fixture requirements verified', check_id='environment')], "", token),
        ):
            async with super().run_test(*args, **kwargs) as pilot:
                await wait_for_condition(
                    pilot,
                    lambda: self.preflight.check_completed and self.updates.started and not self.updates.checking,
                    description="fixture startup preflight and version check", timeout=10,
                )
                await pilot.pause()
                yield pilot


async def expand_check_source(app, pilot, identifier: str) -> None:
    """Open a source through its real title before interacting with its checks."""
    from textual.widgets import Collapsible

    from acprof.tui.run.source_groups import source_group

    source = app.query_one('#preflight-section-' + source_group(identifier), Collapsible)
    if source.collapsed:
        title = source.query_one('CollapsibleTitle')
        title.scroll_visible(immediate=True, force=True)
        title.focus()
        await pilot.press('enter')
        await pilot.pause()
    assert not source.collapsed
