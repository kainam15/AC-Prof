"""Backfill recovery uses shared widgets without changing profiler or measurement gates."""
from dataclasses import replace
from unittest.mock import patch

import pytest
from rich.cells import cell_len
from textual.widgets import Button, Collapsible, Static

from acprof.tui.i18n import message_id
from acprof.tui.results import profile_controller
from acprof.tui.results.backfill_panel import BackfillPanel
from acprof.tui.results.profile_screen import ProfilerChoiceScreen
from acprof.tui.settings.preferences import TuiSettings, UiPreferences, save_settings
from acprof.tui.ui.structured import ActionButtons, ActionRequested, Callout, FactGrid
from tests.runtime.containers.test_layer_read_cancellation import source_hashes
from tests.support.tui_fixtures import AcprofTui
from tests.tui.results.test_profile_overview_snapshot import profile_store


def recovery_button(callout: Callout, action_id: str) -> Button:
    actions = callout.query_one(".callout-actions", ActionButtons)
    index = next(index for index, action in enumerate(actions.actions) if action.id == action_id)
    return list(actions.query(Button))[index]


@pytest.mark.parametrize("language", ["zh", "en"])
@pytest.mark.parametrize("size", [(80, 24), (120, 30)])
async def test_backfill_callout_recovery_is_visible_and_uses_existing_routes(tmp_path, language, size):
    root = tmp_path / "experiment"
    profile_store(root, 3)
    before = source_hashes(root)
    settings = tmp_path / "settings.json"
    save_settings(settings, TuiSettings(ui=UiPreferences(language=language)), tmp_path)
    app = AcprofTui(settings_path=settings)
    async with app.run_test(size=size) as pilot:
        app.activate_tab("backfill-tab")
        app.results.select_result(str(root))
        await app.workers.wait_for_complete()
        await pilot.pause()
        error = "PermissionError: [red]fixture provenance failure[/red]"
        assert app.results.data_status is not None
        app.results.data_status = replace(app.results.data_status, profiler_overview=None, profiler_error=error)
        app.profiles.show_overview(None, error)
        await pilot.pause()
        panel = app.query_one("#backfill-panel", BackfillPanel)
        callout = app.query_one("#backfill-callout", Callout)
        assert panel.view.summary.headline == message_id("profiling.view.error")
        assert panel.view.callout is not None
        assert callout.content == panel.view.callout
        assert str(root) in callout.content.detail
        assert error in callout.content.detail
        assert callout.query_one(".callout-details", Collapsible).collapsed
        headline = app.query_one("#profile-status", Static)
        assert app.tr(panel.view.summary.headline) in str(headline.content)
        assert error not in str(headline.content)
        assert all(not grid.display for grid in panel.query(FactGrid))
        assert app.query_one("#open-result-profiling", Button).disabled

        body = app.query_one("#backfill-body")
        assert body.scroll_y == 0
        assert body.content_region.contains_region(headline.region)
        choose = recovery_button(callout, "backfill-choose")
        refresh = recovery_button(callout, "backfill-refresh")
        assert not choose.disabled
        assert not refresh.disabled
        for button in (choose, refresh):
            assert body.content_region.contains_region(button.region)
            assert app.screen.region.contains_region(button.region)
            center = (button.region.x + button.region.width // 2, button.region.y + button.region.height // 2)
            assert app.get_widget_at(*center)[0] is button
        assert not choose.region.overlaps(refresh.region)
        with patch.object(app.catalog, "open_experiment_picker") as picker:
            assert await pilot.click(choose)
            picker.assert_called_once_with("experiment-source")
        with patch.object(app.results, "refresh_experiment", wraps=app.results.refresh_experiment) as refresh_source:
            assert await pilot.click(refresh)
            await app.workers.wait_for_complete()
            await pilot.pause()
            refresh_source.assert_called_once_with()
        assert app.profiles.overview is not None
        assert app.profiles.overview.result_dir == root
        assert panel.view.callout is None
        assert not app.query_one("#open-result-profiling", Button).disabled
        tools = app.query_one("#backfill-section-0", FactGrid)
        assert tuple(row.label for row in tools.section.rows) == ("torch", "ncu", "nsys", "massif")
    assert source_hashes(root) == before


async def test_backfill_details_preserve_raw_text_copy_focus_and_language_state(tmp_path):
    root = tmp_path / "实验 [source] e\u0301"
    profile_store(root, 3)
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab("backfill-tab")
        app.results.select_result(str(root))
        await app.workers.wait_for_complete()
        await pilot.pause()
        error = "PosthocError: [red]原始异常[/red]\npath=" + str(root / "原文 [1].csv")
        app.profiles.show_overview(None, error)
        await pilot.pause()
        panel = app.query_one("#backfill-panel", BackfillPanel)
        callout = app.query_one("#backfill-callout", Callout)
        details = callout.query_one(".callout-details", Collapsible)
        title = details.query_one("CollapsibleTitle")
        copy = callout.query_one(".callout-copy", Button)
        raw = callout.query_one(".callout-raw", Static)
        retained = callout.content.detail
        assert error in retained
        assert str(root) in retained
        assert str(raw.content) == retained
        assert details.collapsed
        assert copy not in app.screen.focus_chain
        title.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        title.focus()
        await pilot.press("enter")
        assert not details.collapsed
        await pilot.press("tab")
        assert app.focused is copy
        await pilot.press("enter")
        assert app.clipboard == retained

        with patch.object(callout, "update_content", wraps=callout.update_content) as update:
            app.profiles.update_action_availability()
            app.set_busy(False)
            await pilot.pause()
            update.assert_not_called()
        assert app.focused is copy
        assert callout.query_one(".callout-details", Collapsible) is details
        assert not details.collapsed
        app.configuration.preferences = replace(app.configuration.preferences, language="en")
        app.localization.apply_language()
        await pilot.pause()
        assert app.focused is copy
        assert not details.collapsed
        assert details.title == app.tr(message_id("structured.detail.title"))
        for action_id in ("backfill-refresh", "backfill-choose"):
            button = recovery_button(callout, action_id)
            assert button.content_region.width >= cell_len(str(button.label))
        assert str(raw.content) == retained
        assert app.clipboard == retained
        assert panel.query_one(Callout) is callout

        title.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        title.focus()
        await pilot.press("enter")
        assert details.collapsed
        app.configuration.preferences = replace(app.configuration.preferences, language="zh")
        app.localization.apply_language()
        await pilot.pause()
        assert details.collapsed
        assert app.focused is title
        assert copy not in app.screen.focus_chain
        assert str(raw.content) == retained


async def test_backfill_measurement_freezes_projection_and_preserves_operation_gates(tmp_path):
    root = tmp_path / "experiment"
    profile_store(root, 3)
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(120, 30)) as pilot:
        app.activate_tab("backfill-tab")
        app.results.select_result(str(root))
        await app.workers.wait_for_complete()
        await pilot.pause()
        panel = app.query_one("#backfill-panel", BackfillPanel)
        original = app.profiles.overview
        assert original is not None
        assert original.pending
        changed = replace(original, tools=tuple(
            replace(tool, state="complete", completed=tool.total, errors=0) if tool.tool == "ncu" else tool
            for tool in original.tools
        ))
        assert changed != original
        assert changed.pending
        retained = panel.view
        snapshot = app.execution.session.snapshot
        with (
            patch.object(profile_controller, "backfill_view", wraps=profile_controller.backfill_view) as project_view,
            patch("acprof.tui.results.controller.inspect_experiment_data") as read_data,
            patch.object(profile_controller, "inspect_experiment_data") as validate_data,
            patch("acprof.tui.experiments.controller.scan_experiments") as scan,
            patch.object(app, "launch") as launch,
        ):
            try:
                app.execution.session.snapshot = replace(snapshot, measurement_active=True)
                app.set_busy(True)
                app.profiles.show_overview(changed)
                app.set_busy(True)
                # A queued action must still check the live gate after its old button was enabled.
                app.post_message(ActionRequested("backfill-refresh"))
                app.post_message(ActionRequested("backfill-choose"))
                app.profiles.open_selection_dialog()
                await pilot.pause()
                assert app.profiles.overview is changed
                assert panel.view is retained
                assert app.query_one("#open-result-profiling", Button).disabled
                assert not isinstance(app.screen, ProfilerChoiceScreen)
                project_view.assert_not_called()
                read_data.assert_not_called()
                validate_data.assert_not_called()
                scan.assert_not_called()
                launch.assert_not_called()
            finally:
                app.execution.session.snapshot = snapshot
                app.set_busy(False)
            await pilot.pause()
            assert app.profiles.overview is changed
            assert panel.view is not retained
            ncu = next(row for row in panel.view.sections[0].rows if row.label == "ncu")
            assert ncu.status == "complete"
            assert not app.query_one("#open-result-profiling", Button).disabled
            read_data.assert_not_called()
            validate_data.assert_not_called()

        # An ordinary operation also blocks a qualified snapshot until ownership is released.
        app.execution.session.kind = "run"
        try:
            app.set_busy(True)
            app.profiles.open_selection_dialog()
            await pilot.pause()
            assert app.query_one("#open-result-profiling", Button).disabled
            assert not isinstance(app.screen, ProfilerChoiceScreen)
            assert app.profiles.overview is changed
        finally:
            app.execution.session.kind = ""
            app.set_busy(False)
        await pilot.pause()
        assert await pilot.click("#open-result-profiling")
        await pilot.pause()
        assert isinstance(app.screen, ProfilerChoiceScreen)
        await pilot.press("escape")
