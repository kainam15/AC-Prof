"""The model interface declaration must survive form input, commands and settings."""
import tempfile
from pathlib import Path

import pytest
from textual.widgets import ContentSwitcher, Input, Select
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig, RunConfigError, build_run_command
from acprof.tui.commands import build_probe_command
from acprof.tui.settings import TuiSettings, load_settings, save_settings


@pytest.mark.parametrize('size', ((80, 24), (120, 30), (150, 45)))
async def test_declaration_input_reaches_run_probe_and_saved_defaults(size):
    root = Path(__file__).resolve().parents[1]
    path = "examples/multimodal/ultravox.model.json"
    with tempfile.TemporaryDirectory() as directory:
        settings_path = Path(directory) / "tui.json"
        app = AcprofTui(RunConfig.smoke("example/custom"), settings_path=settings_path)
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            app.query_one("#experiment-pages", ContentSwitcher).current = "advanced-form"
            field = app.query_one("#model-spec", Input)
            field.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert (await pilot.click(field))
            await pilot.press(*path)
            assert (field.value) == (path)
            app.query_one("#ui-language", Select).value = "en"
            await pilot.pause()
            assert (app.query_one("#model-spec", Input).value) == (path)
            config = app._collect_config()
            for builder in (build_run_command, build_probe_command):
                command = builder(config, project_dir=root)
                assert (command[command.index("--model-spec") + 1]) == (path)
            save_settings(settings_path, TuiSettings(run_defaults=config), root)
            restored, warning = load_settings(settings_path, root)
            assert (warning) == ("")
            assert (restored.run_defaults.model_spec) == (path)
            app._apply_config(restored.run_defaults)
            assert (field.value) == (path)

def test_missing_declaration_is_reported_before_command_execution():
    with tempfile.TemporaryDirectory() as directory:
        config = RunConfig(model="example/custom", model_spec="missing.json")
        with pytest.raises(RunConfigError, match="missing.json"):
            build_run_command(config, project_dir=Path(directory))
