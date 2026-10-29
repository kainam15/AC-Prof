import asyncio
from unittest.mock import patch

import pytest
from rich.cells import cell_len
from test_layer_read_cancellation import source_hashes
from test_profile_overview_snapshot import profile_store
from textual.containers import VerticalScroll
from textual.widgets import Button, Checkbox, Collapsible, Input, Select, Static
from tui_fixtures import AcprofTui, checkbox_text

from acprof.experiment import RunConfig
from acprof.tui.profile_screen import ProfilerChoiceScreen
from acprof.tui.profile_status import ProfilerOverview, ProfilerStatus
from acprof.tui.settings import TuiSettings, UiPreferences, save_settings
from acprof.tui.views import ConfirmActionScreen


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_real_summary_missing_profiler_records_show_uncollected_choices(tmp_path, language):
    root = tmp_path / "experiment"
    profile_store(root, 3)
    before = source_hashes(root)
    settings = tmp_path / "settings.json"
    save_settings(settings, TuiSettings(ui=UiPreferences(language=language)), tmp_path)
    app = AcprofTui(settings_path=settings)
    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab("results-tab")
        app.query_one("#result-csv", Input).value = str(root)
        await pilot.pause()
        summarize = app.query_one("#summarize-results", Button)
        summarize.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(summarize)
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=3)
        await pilot.pause()
        assert not app.read_jobs
        assert app.profiles.overview is not None

        title = app.query_one("#profile-section CollapsibleTitle")
        title.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(title)
        await pilot.pause()
        choice = app.query_one("#open-result-profiling", Button)
        choice.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(choice)
        await pilot.pause()
        assert isinstance(app.screen, ProfilerChoiceScreen)
        assert app.screen.query_one("#profile-tool-torch", Checkbox).disabled
        assert app.screen.selected_tools() == ("ncu",)
        for tool in ("ncu", "nsys", "massif"):
            box = app.screen.query_one(f"#profile-tool-{tool}", Checkbox)
            box.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert not box.disabled
            assert box.parent is not None
            status_text = str(box.parent.query_one(".profiler-choice-status", Static).content)
            assert status_text == ("尚未采集" if language == "zh" else "Not collected")
            assert 0 <= box.region.x < box.region.right <= 80
            assert 0 <= box.region.y < box.region.bottom <= 24

        nsys = app.screen.query_one("#profile-tool-nsys", Checkbox)
        nsys.scroll_visible(animate=False, immediate=True)
        nsys.focus()
        await pilot.press("space")
        await pilot.pause()
        assert app.screen.selected_tools() == ("ncu", "nsys")
        app.logs.flush()
        app.save_screenshot(path=str(tmp_path), filename=f"profile-missing-{language}.svg")
        with patch.object(app, "launch") as launch:
            await pilot.press("escape")
            await pilot.pause()
            launch.assert_not_called()
        assert not isinstance(app.screen, ProfilerChoiceScreen)
    assert source_hashes(root) == before


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
        app.activate_tab("results-tab")
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

    @pytest.mark.parametrize("size", [(80, 24), (120, 30)])
    async def test_results_profile_selection_and_confirmation(self, size):
        app = self.make_app()
        async with app.run_test(size=size) as pilot:
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

    @pytest.mark.parametrize("size", [(80, 24), (120, 30)])
    async def test_no_selection_disables_actions_and_cancel_keeps_data(self, size):
        app = self.make_app()
        async with app.run_test(size=size) as pilot:
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
                assert box.content_region.height > 0
                assert cell_len(box.label.plain) + 4 <= box.content_region.width
                assert checkbox_text(box).count("✓") == (1 if tool == "ncu" else 0)
                rendered = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
                assert box.label.plain in rendered
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

    @pytest.mark.parametrize("language", ["zh", "en"])
    async def test_dialog_actions_remain_visible_and_clickable_with_a_long_path(self, language):
        self.result_dir = self.result_dir.joinpath(*["深层实验目录-profiler-results"] * 8)
        self.result_dir.mkdir(parents=True)
        save_settings(self.settings_path, TuiSettings(ui=UiPreferences(language=language)), self.result_dir.parent)
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.open_dialog(app, pilot)
            screen = app.screen
            buttons = tuple(screen.query("#profiler-choice-actions Button"))
            rendered = "\n".join(strip.text for strip in screen._compositor.render_strips())
            for button in buttons:
                assert screen.region.contains_region(button.region)
                assert screen.query_one("#profiler-choice-dialog").content_region.contains_region(button.region)
                assert button.label.plain in rendered
                for x, y in ((button.region.x, button.region.y),
                             (button.region.right - 1, button.region.bottom - 1)):
                    assert app.get_widget_at(x, y)[0] is button
            with patch.object(app, "launch") as launch:
                assert await pilot.click("#profile-choice-plan")
                await pilot.pause()
                launch.assert_called_once()
                assert launch.call_args.args[0].kind == "profile-dry-run"
                assert "--dry-run" in launch.call_args.args[0].command

    @pytest.mark.parametrize("language", ["zh", "en"])
    async def test_dialog_scroll_and_resize_preserve_selection_and_fixed_actions(self, language):
        self.result_dir = self.result_dir.joinpath(*["中文模型-result-group"] * 6, "final-experiment")
        save_settings(self.settings_path, TuiSettings(ui=UiPreferences(language=language)), self.result_dir.parent)
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.open_dialog(app, pilot)
            screen = app.screen
            nsys = screen.query_one("#profile-tool-nsys", Checkbox)
            nsys.scroll_visible(animate=False, immediate=True)
            nsys.focus()
            await pilot.press("space")
            await pilot.pause()
            assert screen.selected_tools() == ("ncu", "nsys")
            body = screen.query_one("#profiler-choice-body", VerticalScroll)
            for size in ((150, 45), (80, 24), (120, 30), (150, 45)):
                await pilot.resize_terminal(*size)
                await pilot.pause()
                assert screen.selected_tools() == ("ncu", "nsys")
                buttons = tuple(screen.query("#profiler-choice-actions Button"))
                positions = [button.region for button in buttons]
                body.focus()
                await pilot.press("home")
                await pilot.pause()
                assert body.scroll_y == 0
                path = screen.query_one("#profiler-choice-experiment", Static)
                assert body.content_region.contains_region(path.content_region)
                lines = screen._compositor.render_strips()
                visible_path = "".join(line.crop(path.content_region.x, path.content_region.right).text.rstrip()
                                       for line in lines[path.content_region.y:path.content_region.bottom])
                assert visible_path == str(self.result_dir)
                await pilot.press("end")
                await pilot.pause()
                assert body.scroll_y == body.max_scroll_y
                if size == (80, 24):
                    assert body.scroll_y > 0
                note = screen.query_one("#profiler-choice-note", Static)
                assert body.content_region.contains_region(note.region)
                assert [button.region for button in buttons] == positions
                rendered = "\n".join(strip.text for strip in screen._compositor.render_strips())
                for button in buttons:
                    assert screen.region.contains_region(button.region)
                    assert button.label.plain in rendered
                    assert app.get_widget_at(*button.region.center)[0] is button
                app.save_screenshot(path=str(self.settings_path.parent),
                                    filename=f"profiler-resize-{language}-{size[0]}x{size[1]}.svg")
            with patch.object(app, "launch") as launch:
                assert await pilot.click("#profile-choice-close")
                await pilot.pause()
                launch.assert_not_called()
