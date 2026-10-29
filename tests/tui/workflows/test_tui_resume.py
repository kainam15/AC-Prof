from dataclasses import replace
from pathlib import Path

import pytest
from textual.widgets import Checkbox, ContentSwitcher, Select

from acprof.experiment import RunConfig, build_run_command
from acprof.tui.input import BarCursorInput as Input
from tests.support.tui_fixtures import AcprofTui


@pytest.mark.parametrize(("language", "size"), [("zh", (80, 24)), ("en", (120, 30)), ("zh", (150, 45))])
async def test_restored_run_can_configure_a_new_experiment_without_a_resume_toggle(tmp_path, language, size):
    restored = replace(RunConfig.smoke("org/model"), resume=True, output_dir=str(tmp_path / "original"))
    app = AcprofTui(restored, settings_path=tmp_path / "tui.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size) as pilot:
        assert app.configuration.collect_config() == replace(restored, resume=False)
        app.configuration.apply_config(restored)
        app.query_one("#run-preset", Select).value = "main"
        await pilot.pause()
        assert app.configuration.collect_config().profiling_mode == "full"
        app.query_one("#experiment-pages", ContentSwitcher).current = "advanced-form"
        await pilot.pause()
        assert not app.query("#resume-run")
        app.query_one("#output-dir", Input).value = str(tmp_path / "new")
        app.query_one("#rebuild-image", Checkbox).value = True
        await pilot.pause()
        config = app.configuration.collect_config()
        command = build_run_command(config, project_dir=Path.cwd())
        assert not config.resume
        assert config.output_dir == str(tmp_path / "new")
        assert "--resume" not in command
        assert "--rebuild-image" in command
