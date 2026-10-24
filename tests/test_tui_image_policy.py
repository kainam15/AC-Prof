"""Image rebuilding is an explicit launch action, never a saved default."""
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from textual.widgets import Checkbox, Collapsible, Select
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig, build_run_command
from acprof.tui.commands import PendingLaunch, build_probe_command
from acprof.tui.preparation import PreparationScreen
from acprof.tui.recovery_actions import RecoveryActions
from acprof.tui.settings import load_settings


def test_failed_run_recovery_does_not_repeat_a_consumed_rebuild():
    controller = Mock()
    controller._active_run_config = RunConfig(model="demo/model", rebuild_image=True)
    RecoveryActions.review_failed_run(controller)
    pending = controller._review_run_destination.call_args.args[0]
    assert not pending.config.rebuild_image
    assert "--rebuild-image" not in pending.command


@pytest.mark.parametrize("kind", ("run", "probe"))
@pytest.mark.parametrize("size,language", (((80, 24), "zh"), ((120, 30), "en"), ((150, 45), "zh")))
async def test_rebuild_is_reachable_once_and_not_saved(tmp_path, kind, size, language):
    path = tmp_path / "settings.json"
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=path)
    builder = build_run_command if kind == "run" else build_probe_command
    with patch.object(app, "_execute_command") as execute:
        async with app.run_test(size=size) as pilot:
            assert not app.query("#skip-build")
            app.query_one("#ui-language", Select).value = language
            app.open_run_settings()
            await pilot.pause()
            advanced = app.query_one("#image-build-options", Collapsible)
            assert advanced.collapsed
            advanced.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert await pilot.click(advanced.query_one("CollapsibleTitle"))
            await pilot.pause()
            rebuild = app.query_one("#rebuild-image", Checkbox)
            rebuild.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert await pilot.click(rebuild)
            await pilot.pause()
            config = app._collect_config()
            command = builder(config, project_dir=Path.cwd())
            assert "--rebuild-image" in command
            app.save_run_default()
            assert not app._saved_settings.run_defaults.rebuild_image
            assert "rebuild_image" not in json.loads(path.read_text())["run_defaults"]
            assert rebuild.value  # Saving does not consume the next launch action.
            app._launch(PendingLaunch(tuple(command), kind, config))
            execute.assert_called_once_with(command, kind, config=config)
            assert not rebuild.value
            next_command = builder(app._collect_config(), project_dir=Path.cwd())
            assert "--rebuild-image" not in next_command
    restored, warning = load_settings(path, Path.cwd())
    assert not warning
    assert not restored.run_defaults.rebuild_image


@pytest.mark.parametrize("stage", ("image", "runtime"))
async def test_failed_image_preparation_exposes_explicit_rebuild(tmp_path, stage):
    replies = []
    event = {"stage": stage, "request": {"id": 1, "kind": "error", "detail": "invalid image"}}
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        screen = PreparationScreen(event, respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        details = screen.query_one("#preparation-diagnostics", Collapsible)
        assert await pilot.click(details.query_one("CollapsibleTitle"))
        await pilot.pause()
        rebuild = screen.query_one("#preparation-rebuild")
        rebuild.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(rebuild)
        await pilot.pause()
        assert replies == [{"action": "rebuild"}]
