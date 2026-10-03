from pathlib import Path
from unittest.mock import patch

import pytest
from rich.cells import cell_len
from textual.widgets import Checkbox, Input, Select
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.tui.views import ConfirmActionScreen


class TestTuiProfileTools:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        temporary = tmp_path
        self.result_dir = Path(str(temporary))
        self.settings_path = self.result_dir / "tui.json"

    def make_app(self):
        return AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)

    async def click_visible(self, app, pilot, selector):
        widget = app.screen.query_one(selector)
        widget.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert (await pilot.click(selector, offset=(2, 1)))
        await pilot.pause()

    async def test_mouse_and_space_selection_drive_plan_and_confirmed_run(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            app._activate_tab("profile-tab")
            app.query_one("#result-dir", Input).value = str(self.result_dir)
            await pilot.pause()
            checkboxes = list(app.query("#profile-tools Checkbox"))
            assert (len(checkboxes)) == (4)
            assert ([box.name for box in checkboxes if box.value]) == (["torch", "ncu"])
            for tool in ("torch", "ncu", "nsys"):
                await self.click_visible(app, pilot, f"#profile-tool-{tool}")
            app.query_one("#profile-tool-massif", Checkbox).focus()
            await pilot.pause()
            await pilot.press("space")
            await pilot.pause()

            with patch.object(app, "_launch") as launch:
                await self.click_visible(app, pilot, "#profile-dry-run")
                launch.assert_called_once()
                planned = launch.call_args.args[0]
                assert (planned.kind) == ("profile-dry-run")
                assert (planned.command[planned.command.index("--tools") + 1]) == ("nsys,massif")
                assert ("--dry-run") in (planned.command)

                launch.reset_mock()
                await self.click_visible(app, pilot, "#profile-run")
                assert isinstance(app.screen, ConfirmActionScreen)
                pending = app._pending_launch
                assert (pending.command[pending.command.index("--tools") + 1]) == ("nsys,massif")
                assert ("--dry-run") not in (pending.command)
                launch.assert_not_called()
                await self.click_visible(app, pilot, "#confirm-yes")
                launch.assert_called_once_with(pending)

            for tool in ("torch", "ncu"):
                await self.click_visible(app, pilot, f"#profile-tool-{tool}")
            prepared = app._profile_command(dry_run=True)
            assert prepared is not None
            command, result_path = prepared
            assert (result_path) == (self.result_dir)
            assert (command[command.index("--tools") + 1]) == ("torch,ncu,nsys,massif")

    async def test_empty_selection_blocks_launch_but_explicit_slash_tools_still_work(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            app._activate_tab("profile-tab")
            app.query_one("#result-dir", Input).value = str(self.result_dir)
            await pilot.pause()
            assert (len(app.query("#profile-tools Checkbox"))) == (4)
            for tool in ("torch", "ncu"):
                await self.click_visible(app, pilot, f"#profile-tool-{tool}")
            with patch.object(app, "_launch") as launch, patch.object(app, "notify") as notify:
                for button in ("profile-dry-run", "profile-run"):
                    await self.click_visible(app, pilot, f"#{button}")
                    launch.assert_not_called()
                    assert (app._pending_launch) is None
                    assert ("请至少勾选一个补采工具") in (str(notify.call_args))
                command_input = app.query_one("#slash-command", Input)
                command_input.value = f"/profile {self.result_dir} nsys,massif"
                command_input.focus()
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                launch.assert_called_once()
                command = launch.call_args.args[0].command
                assert (command[command.index("--tools") + 1]) == ("nsys,massif")

    @pytest.mark.parametrize('language', ('en', 'zh'))
    @pytest.mark.parametrize('size', ((150, 45), (120, 30), (80, 24)))
    async def test_options_fit_after_resize_and_language_change_and_lock_during_run(self, language, size):
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            assert (len(app.query("#profile-tools Checkbox"))) == (4)
            await pilot.resize_terminal(*size)
            app.action_show_settings()
            app.query_one("#ui-language", Select).value = language
            await pilot.pause()
            app._activate_tab("profile-tab")
            await pilot.pause()
            for tool in ("torch", "ncu", "nsys", "massif"):
                box = app.query_one(f"#profile-tool-{tool}", Checkbox)
                box.scroll_visible(animate=False, immediate=True)
                await pilot.pause()
                assert (box.region.x) >= (0)
                assert (box.region.right) <= (size[0])
                assert (box.region.y) >= (0)
                assert (box.region.bottom) <= (app.query_one("#bottom-panel").region.y)
                assert (cell_len(box.label.plain) + 4) <= (box.content_region.width)
                assert (app.get_widget_at(box.region.x + 2, box.region.y + 1)[0]) is (box)
                previous = box.value
                assert (await pilot.click(box, offset=(2, 1)))
                await pilot.pause()
                assert (box.value) == (not previous)
            app._set_busy(True)
            assert (all(box.disabled for box in app.query("#profile-tools Checkbox")))
            app._set_busy(False)
            assert (all(not box.disabled for box in app.query("#profile-tools Checkbox")))
