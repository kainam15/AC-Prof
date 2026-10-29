"""The real summary worker releases UI ownership when profiler inspection is cancelled."""
import asyncio
from threading import Event

import pytest
from tests.runtime.containers.test_layer_read_cancellation import source_hashes
from tests.tui.results.test_profile_overview_snapshot import profile_store
from textual.widgets import Button, Input
from tests.support.tui_fixtures import AcprofTui

from acprof.tui import profile_status
from acprof.tui.settings import TuiSettings, UiPreferences, save_settings


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_selection_change_during_profiler_inspection_cancels_and_allows_retry(tmp_path, monkeypatch, language):
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
        app.activate_tab("results-tab")
        selector = app.query_one("#result-csv", Input)
        selector.value = str(first)
        await pilot.pause()
        button = app.query_one("#summarize-results", Button)
        button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        try:
            assert await pilot.click(button)
            assert await asyncio.to_thread(entered.wait, 2)
            assert app.read_jobs
            assert app.profiles.overview is None
            selector.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert await pilot.click(selector)
            await pilot.press("ctrl+a")
            await pilot.press(*str(second))
        finally:
            release.set()
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=3)
        await pilot.pause()
        assert len(set(processed)) == 1
        assert not app.read_jobs
        assert not app.is_busy()
        assert app.profiles.overview is None
        assert selector.value == str(second)
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
