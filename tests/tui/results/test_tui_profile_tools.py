import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from rich.cells import cell_len
from textual import events
from textual.containers import VerticalScroll
from textual.widgets import Button, Checkbox, Input, Select, Static, TabbedContent

from acprof.experiments.configuration import RunConfig
from acprof.tui.results.profile_screen import ProfilerChoiceScreen
from acprof.tui.results.profile_status import ProfilerOverview, ProfilerStatus
from acprof.tui.settings.preferences import TuiSettings, UiPreferences, save_settings
from acprof.tui.ui.dialogs import ConfirmActionScreen
from acprof.tui.ui.structured import Callout, FactGrid
from tests.runtime.containers.test_layer_read_cancellation import source_hashes
from tests.support.tui_fixtures import AcprofTui, checkbox_text
from tests.tui.results.test_profile_overview_snapshot import profile_store


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_experiment_source_editing_is_shared_across_pages(tmp_path, language):
    root = tmp_path / "experiment"
    profile_store(root, 3)
    settings = tmp_path / "settings.json"
    save_settings(settings, TuiSettings(ui=UiPreferences(language=language)), tmp_path)
    app = AcprofTui(settings_path=settings)
    async with app.run_test(size=(120, 30)) as pilot:
        app.activate_tab("backfill-tab")
        await pilot.pause()
        source = app.query_one("#backfill-source")
        assert await pilot.click(source)
        assert app.focused is source
        app.post_message(events.Paste(str(root)))
        await pilot.pause()
        result_source = app.query_one("#experiment-source", Input)
        assert result_source.value == str(root)
        assert await pilot.click("#refresh-backfill-state")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.profiles.overview.result_dir == root
        assert source.value == result_source.value == str(root)

        # A manual edit invalidates both pages, without starting a new reader.
        assert await pilot.click(source)
        await pilot.press("ctrl+a")
        long_path = "/" + "模型 [draft]/" * 18 + "result_all.csv"
        app.post_message(events.Paste(long_path))
        await pilot.pause()
        assert source.value == result_source.value == long_path
        assert app.profiles.overview is None
        assert app.query_one("#open-result-profiling", Button).disabled
        await pilot.pause(0.25)
        await app.workers.wait_for_complete()
        assert not app.read_jobs

        for size in ((120, 30), (80, 24), (150, 45)):
            await pilot.resize_terminal(*size)
            regions = []
            for tab, field, picker_id in (
                ("results-tab", result_source, "choose-experiment-source"),
                ("backfill-tab", source, "choose-backfill-result"),
            ):
                app.activate_tab(tab)
                await pilot.pause()
                assert await pilot.click(field)
                await pilot.press("end")
                assert field.cursor_position == len(long_path)
                assert field.scroll_x > 0
                assert "result_all.csv" in field.render_line(0).text
                await pilot.press("home")
                assert field.cursor_position == 0
                assert field.scroll_x == 0
                assert "模型 [draft]" in field.render_line(0).text
                picker = app.query_one("#" + picker_id, Button)
                assert field.region.height == picker.region.height == 3
                assert field.region.right < picker.region.x
                assert picker.region.right <= size[0]
                regions.append((field.region, picker.region))
                await pilot.press("tab")
                assert app.focused is picker
                with patch.object(app.catalog, "open_experiment_picker") as open_picker:
                    assert await pilot.click(picker)
                    open_picker.assert_called_once_with("experiment-source")
            assert regions[0] == regions[1]

        # Clearing the backfill input also clears the result page selection.
        assert await pilot.click(source)
        await pilot.press("ctrl+a", "backspace")
        assert source.value == result_source.value == ""
        assert source.placeholder == result_source.placeholder
        assert app.query_one("#refresh-backfill-state", Button).disabled
        assert app.query_one("#report-calculate", Button).disabled


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_real_summary_missing_profiler_records_show_uncollected_choices(tmp_path, language):
    root = tmp_path / "experiment"
    profile_store(root, 3)
    before = source_hashes(root)
    settings = tmp_path / "settings.json"
    save_settings(settings, TuiSettings(ui=UiPreferences(language=language)), tmp_path)
    app = AcprofTui(settings_path=settings)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.results.select_result(str(root))
        app.activate_tab("backfill-tab")
        await pilot.pause()
        summarize = app.query_one("#refresh-backfill-state", Button)
        summarize.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(summarize)
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=3)
        await pilot.pause()
        assert not app.read_jobs
        assert app.profiles.overview is not None
        assert app.query_one("#main-tabs", TabbedContent).active == "backfill-tab"
        assert app.query_one("#backfill-source", Input).value == str(root)

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


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_backfill_navigation_shares_selection_and_reports_read_failures(tmp_path, language):
    root = tmp_path / "experiment"
    profile_store(root, 3)
    settings = tmp_path / "settings.json"
    save_settings(settings, TuiSettings(ui=UiPreferences(language=language)), tmp_path)
    app = AcprofTui(settings_path=settings)
    async with app.run_test(size=(80, 24)) as pilot:
        tabs = app.query_one("#main-tabs", TabbedContent)
        app.activate_tab("results-tab")
        await pilot.pause()
        assert not app.query_one("#results-tab").query("#open-result-profiling, #profile-status")
        await pilot.press("down", "enter")
        await pilot.pause()
        assert tabs.active == "backfill-tab"
        assert tabs.get_tab("backfill-tab").label.plain == ("补采实验" if language == "zh" else "Backfill")
        assert app.query_one("#open-result-profiling", Button).disabled
        assert app.query_one("#refresh-backfill-state", Button).disabled
        with patch.object(app.catalog, "open_experiment_picker") as picker:
            assert await pilot.click("#choose-backfill-result")
            picker.assert_called_once_with("experiment-source")
        app.catalog.experiment_selected("experiment-source", ("select", SimpleNamespace(directory=root)))
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=3)
        await pilot.pause()
        assert tabs.active == "backfill-tab"
        assert app.profiles.overview.result_dir == root
        assert not app.query_one("#open-result-profiling", Button).disabled

        app.activate_tab("results-tab")
        await pilot.pause()
        source = app.query_one("#experiment-source", Input)
        assert source.value == str(root)
        source.value = str(root / "missing.csv")
        await pilot.pause(0.25)
        await app.workers.wait_for_complete()
        assert app.profiles.overview is None
        assert app.query_one("#backfill-source", Input).value == str(root / "missing.csv")
        assert await pilot.click("#nav-backfill-tab")
        with patch("acprof.tui.results.controller.inspect_experiment_data", side_effect=ValueError("fixture data check failed")):
            assert await pilot.click("#refresh-backfill-state")
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=3)
            await pilot.pause()
        callout = app.query_one("#backfill-callout", Callout)
        assert "fixture data check failed" in callout.content.detail
        assert "fixture data check failed" not in str(app.query_one("#profile-status", Static).content)
        assert app.query_one("#open-result-profiling", Button).disabled
        assert not app.is_busy()


class TestTuiProfileTools:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.result_dir = tmp_path
        self.settings_path = tmp_path / "tui.json"
        profile_store(tmp_path, 3)

    def make_app(self):
        app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.settings_path)
        app.configuration.preferences = replace(app.configuration.preferences, show_command_bar=True)
        return app

    def overview(self):
        return ProfilerOverview(self.result_dir, (
            ProfilerStatus("torch", "complete", 8, 8, 0),
            ProfilerStatus("ncu", "missing", 0, 4, 0),
            ProfilerStatus("nsys", "partial", 1, 4, 0),
            ProfilerStatus("massif", "inapplicable", 0, 0, 0),
        ))

    async def open_dialog(self, app, pilot):
        # Complete initial TabbedContent activation before changing pages.
        # Otherwise its queued run-tab event may override the test navigation.
        await pilot.pause()
        app.activate_tab("backfill-tab")
        await pilot.pause()
        app.results.select_result(str(self.result_dir))
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.profiles.show_overview(self.overview())
        tools = app.query_one("#backfill-section-0", FactGrid)
        assert tuple(row.label for row in tools.section.rows) == ("torch", "ncu", "nsys", "massif")
        button = app.query_one("#open-result-profiling", Button)
        app.query_one("#backfill-body", VerticalScroll).scroll_end(animate=False)
        await pilot.pause()
        assert app.screen.get_widget_at(button.region.x + 2, button.region.y + 1)[0] is button
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
                await app.workers.wait_for_complete()
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
                await app.workers.wait_for_complete()
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
                await app.workers.wait_for_complete()
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
            app.execution.session.kind = "run"
            app.set_busy(True)
            assert app.query_one("#open-result-profiling", Button).disabled
            app.execution.session.kind = ""
            app.set_busy(False)
            assert not app.query_one("#open-result-profiling", Button).disabled

    @pytest.mark.parametrize("language", ["zh", "en"])
    async def test_dialog_actions_remain_visible_and_clickable_with_a_long_path(self, language):
        self.result_dir = self.result_dir.joinpath(*["深层实验目录-profiler-results"] * 8)
        self.result_dir.mkdir(parents=True)
        profile_store(self.result_dir, 3)
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
                await app.workers.wait_for_complete()
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
