import tempfile
from pathlib import Path

import pytest
from textual.widgets import Checkbox, ContentSwitcher
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig, build_run_command


@pytest.mark.parametrize('size', ((80, 24), (120, 30), (150, 45)))
async def test_resume_checkbox_reaches_run_command_at_supported_sizes(size):
    with tempfile.TemporaryDirectory() as directory:
        app = AcprofTui(RunConfig.smoke("org/model"), settings_path=Path(directory) / "tui.json")
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            app.query_one("#experiment-pages", ContentSwitcher).current = "advanced-form"
            await pilot.pause()
            checkbox = app.query_one("#resume-run", Checkbox)
            checkbox.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert not (checkbox.value)
            assert (await pilot.click(checkbox))
            await pilot.pause()
            config = app._collect_config()
            command = build_run_command(config, project_dir=Path.cwd())
            assert ("--resume") in (command)
