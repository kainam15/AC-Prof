import tempfile
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.tui.views import ConfirmActionScreen


class TestTuiConfirm:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
    def make_app(self):
        temporary = tempfile.TemporaryDirectory()
        self._request.addfinalizer(partial(temporary.cleanup))
        return AcprofTui(
            RunConfig.smoke("demo/model"),
            settings_path=Path(temporary.name) / "tui.json",
        )

    def assert_highlighted(self, button, highlighted):
        assert (bool(button.styles.text_style.underline)) == (highlighted)
        assert not (button.styles.text_style.reverse)
        assert (button.styles.background.a) == (0)

    @pytest.mark.parametrize('theme', ('acprof-dark', 'acprof-light'))
    @pytest.mark.parametrize('size', ((80, 24), (120, 30), (150, 45)))
    async def test_mouse_highlight_only_lasts_while_hovering_confirmation(self, theme, size):
        app = self.make_app()
        app.theme = theme
        with patch.object(app, "_launch") as launch:
            async with app.run_test(size=size) as pilot:
                await pilot.pause()
                assert (await pilot.click("#probe-largest"))
                await pilot.pause()
                assert isinstance(app.screen, ConfirmActionScreen)
                cancel = app.screen.query_one("#confirm-no", Button)
                confirm = app.screen.query_one("#confirm-yes", Button)
                self.assert_highlighted(cancel, False)
                self.assert_highlighted(confirm, False)
                assert (app.screen.focused) is None

                for hovered, other in ((cancel, confirm), (confirm, cancel)):
                    assert (await pilot.hover(hovered))
                    await pilot.pause()
                    self.assert_highlighted(hovered, True)
                    self.assert_highlighted(other, False)
                    assert (app.screen.focused) is None
                    assert (await pilot.hover("#confirm-title"))
                    await pilot.pause()
                    self.assert_highlighted(hovered, False)

                # Resuming or resizing must not restore automatic focus.
                app.app_focus = False
                await pilot.pause()
                app.app_focus = True
                await pilot.resize_terminal(120, 30)
                await pilot.pause()
                assert (app.screen.focused) is None
                self.assert_highlighted(cancel, False)
                self.assert_highlighted(confirm, False)
                assert (await pilot.click(cancel))
                await pilot.pause()
                assert not isinstance(app.screen, ConfirmActionScreen)
                assert (app._pending_launch) is None
                launch.assert_not_called()

    async def test_keyboard_can_select_cancel_or_confirm_after_opening_dialog(self):
        app = self.make_app()
        with patch.object(app, "_launch") as launch:
            async with app.run_test(size=(120, 30)) as pilot:
                app.action_request_probe()
                await pilot.pause()
                dialog = app.screen
                assert isinstance(dialog, ConfirmActionScreen)
                await pilot.press("enter", "space")
                await pilot.pause()
                assert (app.screen) is (dialog)
                launch.assert_not_called()

                cancel = dialog.query_one("#confirm-no", Button)
                confirm = dialog.query_one("#confirm-yes", Button)
                for key, focused, other in (
                    ("tab", cancel, confirm),
                    ("tab", confirm, cancel),
                    ("shift+tab", cancel, confirm),
                ):
                    await pilot.press(key)
                    await pilot.pause()
                    assert (dialog.focused) is (focused)
                    self.assert_highlighted(focused, True)
                    self.assert_highlighted(other, False)
                await pilot.press("enter")
                await pilot.pause()
                assert not isinstance(app.screen, ConfirmActionScreen)
                assert (app._pending_launch) is None
                launch.assert_not_called()

                app.action_request_probe()
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()
                assert not isinstance(app.screen, ConfirmActionScreen)
                assert (app._pending_launch) is None
                launch.assert_not_called()

                app.action_request_probe()
                await pilot.pause()
                await pilot.press("tab", "tab")
                await pilot.pause()
                assert (app.screen.focused.id) == ("confirm-yes")
                await pilot.press("enter")
                await pilot.pause()
                assert not isinstance(app.screen, ConfirmActionScreen)
                assert (app._pending_launch) is None
                launch.assert_called_once()
                assert (launch.call_args.args[0].kind) == ("probe")

    async def test_mouse_can_confirm_without_prior_keyboard_selection(self):
        app = self.make_app()
        with patch.object(app, "_launch") as launch:
            async with app.run_test(size=(120, 30)) as pilot:
                await pilot.pause()
                assert (await pilot.click("#probe-largest"))
                await pilot.pause()
                assert (await pilot.click("#confirm-yes"))
                await pilot.pause()
                assert not isinstance(app.screen, ConfirmActionScreen)
                assert (app._pending_launch) is None
                launch.assert_called_once()
                assert (launch.call_args.args[0].kind) == ("probe")
