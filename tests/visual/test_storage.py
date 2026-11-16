"""Storage feedback stays visible while its data scrolls independently."""
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial

import pytest
from textual.widgets import Button, Static

from acprof.experiments.configuration import RunConfig
from acprof.host.images.inventory import DiskSpace, DockerConnection, DockerStorage, StorageUsage
from acprof.host.images.storage_errors import storage_query_error
from acprof.tui.images.storage import StorageSpaceScreen
from acprof.tui.ui.view_models import format_local_timestamp
from tests.support.tui_fixtures import AcprofTui

pytestmark = pytest.mark.visual


@pytest.mark.parametrize("language", ["zh", "en"])
@pytest.mark.parametrize("state", ["loading", "error", "partial", "success"])
def test_storage_feedback(snap_compare, tmp_path, monkeypatch, language, state):
    monkeypatch.setattr("acprof.tui.images.storage.format_local_timestamp", partial(format_local_timestamp, tz=UTC))
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language, theme="acprof-graphite")
    size = (150, 45) if state == "success" else (80, 24)

    async def prepare(pilot):
        await pilot.pause()
        await app.workers.wait_for_complete()
        app.clear_notifications()
        app.set_input_cursor_blink_enabled(False)
        screen = StorageSpaceScreen(lambda: None, selected_count=2)
        await app.push_screen(screen)
        await pilot.pause()
        snapshot = DockerStorage(
            DockerConnection(("--context", "default"), "default"), "fixture-daemon", "/var/lib/docker",
            DiskSpace(512 * 1024**3, 384 * 1024**3, 128 * 1024**3),
            tuple(StorageUsage(kind, used * 1024**3, reclaim * 1024**3) for kind, used, reclaim in (
                ("Images", 46, 18), ("Containers", 3, 1), ("Local Volumes", 12, 2), ("Build Cache", 9, 9),
            )), observed_at=datetime(2026, 11, 16, 4, 5, 6, tzinfo=UTC),
        )
        if state == "error":
            screen.show_storage(None, None, storage_query_error(PermissionError("permission denied")))
        elif state == "partial":
            screen.show_storage(snapshot, 2 * 1024**3)
            screen.show_storage(replace(
                snapshot, disk=None, disk_error=storage_query_error(PermissionError("/var/lib/docker"), filesystem=True),
                observed_at=datetime(2026, 11, 16, 4, 10, 0, tzinfo=UTC),
            ), 2 * 1024**3)
        elif state == "success":
            screen.show_storage(snapshot, 2 * 1024**3)
        await pilot.pause()
        status = screen.query_one("#storage-status", Static)
        assert status.region.y < screen.query_one("#storage-content").region.y
        assert status.region.bottom < size[1] - 4
        for selector in ("#storage-refresh", "#storage-close"):
            button = screen.query_one(selector, Button)
            assert button.region.height == 3
            assert app.get_widget_at(*button.region.center)[0] is button
        assert screen.query_one("#storage-refresh", Button).disabled == (state == "loading")

    assert snap_compare(app, terminal_size=size, run_before=prepare)
