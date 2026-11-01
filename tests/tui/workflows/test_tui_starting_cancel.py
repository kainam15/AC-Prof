"""Starting is cancellable before a child exists, without losing ownership."""
import asyncio
import sys
import threading
from dataclasses import replace
from unittest.mock import patch

from textual.widgets import Button

from acprof.experiments.configuration import RunConfig
from acprof.tui.commands import PendingLaunch
from acprof.tui.run.results import RunArtifacts
from acprof.tui.run.stages import Stage
from tests.support.tui_fixtures import AcprofTui


async def test_cancel_during_history_read_does_not_spawn_and_allows_retry(tmp_path):
    config = replace(RunConfig.smoke("demo/model"), output_dir=str(tmp_path / "results"))
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    entered, release = threading.Event(), threading.Event()
    read_artifacts = RunArtifacts.read
    start_child = app.execution.lifecycle.start

    def delayed_read(directory, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(10), "test did not release the historical artifact reader"
        return read_artifacts(directory, **kwargs)

    def harmless_child(_command, **kwargs):
        return start_child([sys.executable, "-c", "raise SystemExit(1)"], **kwargs)

    with (
        patch.object(RunArtifacts, "read", side_effect=delayed_read),
        patch.object(app.execution.lifecycle, "start", side_effect=harmless_child) as start,
    ):
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            try:
                assert await pilot.click("#start-run")
                assert await asyncio.to_thread(entered.wait, 5)
                assert app.execution.lifecycle.process is None
                assert not app.query_one("#stop-run", Button).disabled
                await pilot.press("ctrl+x")
                await pilot.pause()
                assert await pilot.click("#confirm-yes")
                assert app.execution.session.stop_requested
                assert app.is_busy()
                assert app.query_one("#start-run", Button).disabled
                start.assert_not_called()
            finally:
                release.set()
            await asyncio.wait_for(asyncio.shield(app.workers.wait_for_complete()), 10)
            await pilot.pause()
            start.assert_not_called()
            assert app.execution.session.snapshot.stage is Stage.STOPPED
            assert app.execution.lifecycle.process is None
            assert not app.is_busy()
            assert not app.query_one("#start-run", Button).disabled

            app.activate_tab("run-tab")
            await pilot.pause()
            assert await pilot.click("#start-run")
            await asyncio.wait_for(asyncio.shield(app.workers.wait_for_complete()), 10)
            await pilot.pause()
            start.assert_called_once()
            assert app.execution.session.snapshot.stage is Stage.FAILED
            assert app.execution.lifecycle.process is None
            assert not app.query_one("#start-run", Button).disabled


def test_queued_old_worker_cannot_launch_or_finish_a_replacement(tmp_path):
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    old = app.execution.begin(PendingLaunch(("old",), "run"))
    app.execution.request_stop(old)
    app.execution.finish("run", 1, None, "")
    current = app.execution.begin(PendingLaunch(("current",), "run"))
    with patch.object(app.execution.lifecycle, "start") as start, patch.object(app, "process_finished") as finished:
        app.process._execute_command_worker(["old"], "run", config=None, token=old)
    start.assert_not_called()
    finished.assert_not_called()
    assert app.execution.session.token is current
    assert app.execution.session.launch_pending
    assert not app.execution.session.stop_requested
