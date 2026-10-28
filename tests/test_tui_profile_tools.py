from unittest.mock import patch

import pytest
from rich.cells import cell_len
from textual.widgets import Button, Checkbox, Collapsible, Input, Select
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.tui.profile_screen import ProfilerChoiceScreen
from acprof.tui.profile_status import ProfilerOverview, ProfilerStatus
from acprof.tui.views import ConfirmActionScreen


class TestTuiProfileTools:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.result_dir = tmp_path
        self.settings_path = tmp_path / "tui.json"

    def make_app(self):
        return AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)

    def overview(self):
        return ProfilerOverview(self.result_dir, (
            ProfilerStatus("torch", "complete", 8, 8, 0),
            ProfilerStatus("ncu", "missing", 0, 4, 0),
            ProfilerStatus("nsys", "partial", 1, 4, 0),
            ProfilerStatus("massif", "inapplicable", 0, 0, 0),
        ))

    async def open_dialog(self, app, pilot):
        app.activate_tab("plot-tab")
        app.profiles.show_overview(self.overview())
        text = str(app.query_one("#profile-status").content)
        assert "torch" in text
        assert "ncu" in text
        assert "massif" in text
        app.query_one("#profile-section", Collapsible).collapsed = False
        await pilot.pause()
        button = app.query_one("#open-result-profiling", Button)
        button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(button, offset=(2, 1))
        await pilot.pause()
        assert isinstance(app.screen, ProfilerChoiceScreen)

    async def test_results_profile_selection_and_confirmation(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.open_dialog(app, pilot)
            assert [box.name for box in app.screen.query(".profile-tool") if box.value] == ["ncu"]
            assert app.screen.query_one("#profile-tool-torch", Checkbox).disabled
            assert app.screen.query_one("#profile-tool-massif", Checkbox).disabled
            assert not app.screen.query_one("#profile-tool-nsys", Checkbox).disabled
            # A keyboard toggle chooses an additional execution profiler.
            ns = app.screen.query_one("#profile-tool-nsys", Checkbox)
            ns.focus()
            await pilot.press("space")
            await pilot.pause()
            assert ns.value

            with patch.object(app, "launch") as launch:
                assert await pilot.click("#profile-choice-plan")
                await pilot.pause()
                launch.assert_called_once()
                planned = launch.call_args.args[0]
                assert planned.kind == "profile-dry-run"
                assert planned.command[planned.command.index("--tools") + 1] == "ncu,nsys"
                assert "--dry-run" in planned.command
                launch.reset_mock()

                await self.open_dialog(app, pilot)
                assert await pilot.click("#profile-choice-run")
                await pilot.pause()
                assert isinstance(app.screen, ConfirmActionScreen)
                pending = app.experiments.pending_launch
                assert pending.command[pending.command.index("--tools") + 1] == "ncu"
                assert "--dry-run" not in pending.command
                launch.assert_not_called()
                assert await pilot.click("#confirm-yes")
                await pilot.pause()
                launch.assert_called_once_with(pending)

    async def test_no_selection_disables_actions_and_cancel_keeps_data(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.open_dialog(app, pilot)
            assert await pilot.click("#profile-tool-ncu", offset=(2, 1))
            await pilot.pause()
            assert app.screen.query_one("#profile-choice-plan", Button).disabled
            assert app.screen.query_one("#profile-choice-run", Button).disabled
            with patch.object(app, "launch") as launch:
                await pilot.press("escape")
                await pilot.pause()
                launch.assert_not_called()
                assert app.profiles.overview is not None
                # Explicit slash commands remain usable without UI-only checkboxes.
                field = app.query_one("#slash-command", Input)
                field.value = f"/profile {self.result_dir} nsys,massif"
                field.focus()
                await pilot.press("enter")
                await pilot.pause()
                launch.assert_called_once()
                assert launch.call_args.args[0].command[
                    launch.call_args.args[0].command.index("--tools") + 1] == "nsys,massif"

    @pytest.mark.parametrize("language", ["en", "zh"])
    @pytest.mark.parametrize("size", [(150, 45), (120, 30), (80, 24)])
    async def test_dialog_fits_languages_and_sizes_and_busy_state(self, language, size):
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            app.configuration.show_settings()
            app.query_one("#ui-language", Select).value = language
            await pilot.pause()
            await pilot.resize_terminal(*size)
            await self.open_dialog(app, pilot)
            if language == "en":
                assert "Advanced performance analysis" in str(app.screen.query_one("#profiler-choice-title").content)
            for tool in ("torch", "ncu", "nsys", "massif"):
                box = app.screen.query_one(f"#profile-tool-{tool}", Checkbox)
                box.scroll_visible(animate=False, immediate=True)
                await pilot.pause()
                assert box.region.x >= 0
                assert box.region.right <= size[0]
                assert box.region.y >= 0
                assert box.region.bottom <= size[1]
                assert cell_len(box.label.plain) + 4 <= box.content_region.width
            await pilot.press("escape")
            await pilot.pause()
            app.set_busy(True)
            # Actual busy flags, not just set_busy(True), drive launch eligibility.
            app.process_kind = "run"
            app.set_busy(True)
            assert app.query_one("#open-result-profiling", Button).disabled
            app.process_kind = ""
            app.set_busy(False)
            assert not app.query_one("#open-result-profiling", Button).disabled
