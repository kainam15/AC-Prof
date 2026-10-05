"""Visible platform label and launch policy, without launching collectors."""
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from textual.widgets import Button
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.platform import Environment
from acprof.tui.preparation import PreparationScreen


async def test_partial_launch_keeps_monitor_visible_at_two_sizes():
    wsl = Environment("wsl2")
    for size in ((80, 24), (120, 40)):
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.tui.app.detect_environment", return_value=wsl,
        ), patch("acprof.tui.diagnostics.detect_environment", return_value=wsl):
            config = replace(RunConfig.smoke("example/model"), profiling_mode="basic")
            app = AcprofTui(config, settings_path=Path(directory) / "settings.json")
            async with app.run_test(size=size) as pilot:
                assert ("WSL2 / PARTIAL") in (app.sub_title)
                with patch.object(app, "_execute_command") as launch:
                    await pilot.press("f5")
                    await pilot.pause()
                    launch.assert_called_once()
                    assert app.query_one("#main-tabs").active == "monitor-tab"
                    assert app._preparation_screen is None
                    log = app.query_one("#run-log")
                    stop = app.query_one("#stop-run", Button)
                    assert log.display and log.region.height > 0
                    assert stop.region.width > 0
                    assert stop.region.bottom <= app.screen.region.bottom
                    assert app._pending_launch is None
                    app._process_kind = ""

async def test_unsupported_collector_never_reaches_preparation():
    with tempfile.TemporaryDirectory() as directory, patch(
        "acprof.tui.app.detect_environment", return_value=Environment("wsl2"),
    ):
        config = replace(RunConfig.smoke("example/model"), profiling_mode="full")
        app = AcprofTui(config, settings_path=Path(directory) / "settings.json")
        async with app.run_test(size=(120, 40)) as pilot:
            with patch.object(app, "_launch") as launch:
                await pilot.press("f5")
                await pilot.pause()
                assert not isinstance(app.screen, PreparationScreen)
                assert (app._pending_launch) is None
                launch.assert_not_called()
