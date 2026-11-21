"""The real summary worker releases UI ownership when profiler inspection is cancelled."""
import asyncio
from threading import Event

import pytest
from textual.widgets import Button, Input, Static

from acprof.analysis import audit
from acprof.tui.experiments import catalog
from acprof.tui.i18n import message_id
from acprof.tui.results import profile_status
from acprof.tui.results.backfill_panel import BackfillPanel
from acprof.tui.settings.preferences import TuiSettings, UiPreferences, save_settings
from tests.runtime.containers.test_layer_read_cancellation import source_hashes
from tests.support.tui_actions import fill_form, open_page
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition
from tests.tui.results.test_profile_overview_snapshot import profile_store


async def test_cancel_stops_metadata_overview_and_refresh_reads_the_same_experiment(tmp_path, monkeypatch):
    source = tmp_path / "experiment"
    profile_store(source, 3)
    entered, release = Event(), Event()
    read = catalog.read_artifact
    visited = []

    def pause_at_static_metadata(path, **kwargs):
        visited.append(path.name)
        if path == source / "static_meta.json" and not entered.is_set():
            entered.set()
            assert release.wait(10), "test did not release the metadata read"
        return read(path, **kwargs)

    monkeypatch.setattr(catalog, "read_artifact", pause_at_static_metadata)
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        await open_page(pilot, "results-tab")
        app.results.select_result(str(source))
        request = app.results.overview_request
        assert request is not None
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            cancel = app.query_one("#results-cancel", Button)
            cancel.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert await pilot.click(cancel)
            await wait_for_condition(pilot, request.is_set, description="overview cancellation", timeout=3)
            assert not app.operation_state().allows("run")
        finally:
            release.set()
        await wait_for_condition(pilot, lambda: not app.read_jobs, description="overview release", timeout=3)
        assert visited[-1] == "static_meta.json"
        assert app.results.overview is None
        refresh = app.query_one("#refresh-experiment-state", Button)
        refresh.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(refresh)
        await wait_for_condition(pilot, lambda: app.results.overview is not None and not app.read_jobs,
                                 description="refreshed experiment overview", timeout=3)
        assert app.results.overview.source == source
        assert app.results.overview.metadata_state == "exists"


async def test_cancel_during_audit_releases_collection_after_current_field(tmp_path, monkeypatch):
    source = tmp_path / "experiment"
    profile_store(source, 3)
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    entered, release = Event(), Event()
    fields = []
    observe = audit.metric_observation

    def hold_first_field(row, field):
        fields.append(field)
        if not entered.is_set():
            entered.set()
            assert release.wait(10), "test did not release the audit field"
        return observe(row, field)

    monkeypatch.setattr(audit, "metric_observation", hold_first_field)
    async with app.run_test(size=(80, 24)) as pilot:
        await open_page(pilot, "results-tab")
        app.results.select_result(str(source))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            button = app.query_one("#results-cancel", Button)
            button.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert await pilot.click(button)
            await wait_for_condition(pilot, lambda: app.results.data_cancelled,
                                     description="cancel request to reach the Results worker", timeout=3)
            assert not app.operation_state().allows("run")
        finally:
            release.set()
        await wait_for_condition(pilot, lambda: app.operation_state().allows("run"),
                                 description="cancelled audit to release the collection gate", timeout=3)
        assert len(fields) == 1
        assert app.results.data_status is None
        assert app.profiles.overview is None
        refresh = app.query_one("#refresh-experiment-state", Button)
        await wait_for_condition(pilot, lambda: not refresh.disabled,
                                 description="refresh to become available", timeout=3)
        refresh.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(refresh)
        await wait_for_condition(pilot, lambda: app.results.data_status is not None and not app.read_jobs,
                                 description="the refreshed audit to finish", timeout=3)
        status = app.results.data_status
        assert status is not None
        assert status.audit["counts"]["formal_ok"] == 2
        assert len(fields) > 1


@pytest.mark.parametrize("language", ["zh", "en"])
@pytest.mark.parametrize("action", ["change-source", "change-backfill-source", "cancel-backfill", "cancel-results"])
async def test_selection_change_during_profiler_inspection_cancels_and_allows_retry(tmp_path, monkeypatch, language, action):
    first, second = tmp_path / "first", tmp_path / "second"
    profile_store(first, 3)
    profile_store(second, 3)
    before = source_hashes(tmp_path)
    settings = tmp_path / "settings.json"
    save_settings(settings, TuiSettings(ui=UiPreferences(language=language)), tmp_path)
    app = AcprofTui(settings_path=settings)
    entered, release = Event(), Event()
    original = profile_status.row_tool_complete
    processed = []

    def hold_first_row(row, tool):
        processed.append(id(row))
        if not entered.is_set():
            entered.set()
            assert release.wait(15), "test did not release the current profiler row"
        return original(row, tool)

    monkeypatch.setattr(profile_status, "row_tool_complete", hold_first_row)
    async with app.run_test(size=(80, 24)) as pilot:
        await open_page(pilot, "results-tab")
        selector = app.query_one("#experiment-source", Input)
        app.results.select_result(str(first))
        first_request = app.results.data_request
        assert first_request is not None
        first_cancelled = first_request.is_set
        await pilot.pause()
        button = app.query_one("#refresh-experiment-state", Button)
        button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            assert app.read_jobs
            assert app.profiles.overview is None
            if action == "cancel-results":
                buttons = [app.query_one(identifier, Button) for identifier in (
                    "#refresh-experiment-state", "#results-cancel", "#report-calculate", "#plot-results")]
                for control in buttons:
                    assert control.region.area > 0
                    assert app.query_one("#result-actions").content_region.contains_region(control.region)
                    assert app.get_widget_at(control.region.right - 1, control.region.y + 1)[0] is control
                assert all(not left.region.overlaps(right.region)
                           for index, left in enumerate(buttons) for right in buttons[index + 1:])
                assert await pilot.click("#results-cancel")
                await pilot.pause()
                assert app.tr("读取已取消；等待后台任务释放资源。") in str(app.query_one("#result-data-status", Static).content)
            elif action == "cancel-backfill":
                await open_page(pilot, "backfill-tab")
                button = app.query_one("#refresh-experiment-state", Button)
                assert await pilot.click("#results-cancel")
                await pilot.pause()
                view = app.query_one("#backfill-panel", BackfillPanel).view
                assert view.summary.headline == message_id("profiling.view.cancelled")
                assert view.callout is not None
                assert view.callout.level == "warning"
            else:
                if action == "change-backfill-source":
                    await open_page(pilot, "backfill-tab")
                    selector = app.query_one("#experiment-source", Input)
                    button = app.query_one("#refresh-experiment-state", Button)
                await fill_form(pilot, {f"#{selector.id}": str(second)})
                await wait_for_condition(
                    pilot, lambda: first_cancelled() and app.results.source_path == second,
                    description="the edited source to cancel the first reader", timeout=3,
                    diagnostics=lambda: f"source={app.results.source_path}; cancelled={first_cancelled()}",
                )
        finally:
            release.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=3)
        await pilot.pause()
        assert processed
        if action in {"cancel-backfill", "cancel-results"}:
            assert not app.read_jobs
            assert not app.is_busy()
            assert app.profiles.overview is None
            app.results.select_result(str(second))
        await wait_for_condition(
            pilot,
            lambda: app.profiles.overview is not None and app.profiles.overview.result_dir == second
            and not app.read_jobs,
            description="the second experiment overview after cancellation", timeout=3,
            diagnostics=lambda: f"source={app.results.source_path}; error={app.results.data_error}; readers={len(app.read_jobs)}",
        )
        await pilot.pause()
        assert not app.read_jobs
        assert not app.is_busy()
        assert app.profiles.overview.result_dir == second
        assert selector.value == str(second)
        assert app.query_one("#experiment-source", Input).value == str(second)
        assert not button.disabled
        button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(button)
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=3)
        await pilot.pause()
        assert app.profiles.overview is not None
        assert app.profiles.overview.result_dir == second
        assert not app.is_busy()
    after = source_hashes(tmp_path)
    assert {name: after[name] for name in before} == before
