"""Visible platform label and launch policy, without launching collectors."""
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from textual.widgets import Button, Static
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.platform import Environment
from acprof.tui.views import ConfirmActionScreen


async def test_partial_confirmation_and_escape_at_two_sizes():
    wsl = Environment("wsl2")
    for size in ((80, 24), (120, 40)):
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.tui.app.detect_environment", return_value=wsl,
        ), patch("acprof.tui.diagnostics.detect_environment", return_value=wsl):
            config = replace(RunConfig.smoke("example/model"), profiling_mode="basic")
            app = AcprofTui(config, settings_path=Path(directory) / "settings.json")
            async with app.run_test(size=size) as pilot:
                assert ("WSL2 / PARTIAL") in (app.sub_title)
                with patch.object(app, "_launch") as launch:
                    await pilot.press("f5")
                    await pilot.pause()
                    assert isinstance(app.screen, ConfirmActionScreen)
                    assert ("WSL2 / PARTIAL") in (str(app.screen.query_one("#confirm-message", Static).render()))
                    for button in app.screen.query(Button):
                        assert (button.region.width) > (0)
                        assert (button.region.bottom) <= (app.screen.region.bottom)
                    if size == (80, 24):
                        assert (await pilot.click("#confirm-no"))
                    else:
                        await pilot.press("escape")
                    await pilot.pause()
                    launch.assert_not_called()
                    assert (app._pending_launch) is None

async def test_unsupported_collector_never_reaches_confirmation():
    with tempfile.TemporaryDirectory() as directory, patch(
        "acprof.tui.app.detect_environment", return_value=Environment("wsl2"),
    ):
        config = replace(RunConfig.smoke("example/model"), profiling_mode="full")
        app = AcprofTui(config, settings_path=Path(directory) / "settings.json")
        async with app.run_test(size=(120, 40)) as pilot:
            with patch.object(app, "_launch") as launch:
                await pilot.press("f5")
                await pilot.pause()
                assert not isinstance(app.screen, ConfirmActionScreen)
                assert (app._pending_launch) is None
                launch.assert_not_called()
