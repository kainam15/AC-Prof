import asyncio
import sys

from acprof.tui.commands import PendingLaunch
from acprof.tui.logging.widget import SelectableLog
from acprof.tui.run.process import ProcessLifecycle
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition


async def test_download_progress_arrives_before_child_finishes_without_log_spam(tmp_path, monkeypatch):
    script = '''
import sys
from dataclasses import replace
from acprof.events.download import DownloadProgress, encode_event
progress = DownloadProgress('example/test', 'huggingface', 'downloading', 1000, 200, 600, 100, 40, 1, 2)
print(encode_event(progress), flush=True)
assert sys.stdin.readline().strip() == 'complete'
print(encode_event(replace(progress, state='complete', completed_bytes=1000, files_completed=2, rate_bytes_s=None)), flush=True)
'''
    original_start = ProcessLifecycle.start

    def start_with_input(owner, command, *, cwd, env):
        # The fixture's generic command needs only a pipe for its completion handshake.
        return original_start(owner, command, cwd=cwd, env={
            **env, "ACPROF_INTERACTIVE_PREPARATION": "1",
        })

    monkeypatch.setattr(ProcessLifecycle, "start", start_with_input)
    app = AcprofTui(settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(120, 30)) as pilot:
        app.launch(PendingLaunch((sys.executable, '-u', '-c', script), 'test'))
        try:
            await wait_for_condition(
                pilot, lambda: app.preflight.latest_download is not None,
                description="streamed download progress before child completion", timeout=5,
            )
            assert app.execution.lifecycle.process is not None
            assert app.execution.lifecycle.process.poll() is None
            assert app.preflight.latest_download.state == 'downloading'
        finally:
            # Release and reap the fixture even if a streaming assertion fails.
            process = app.execution.lifecycle.process
            if process is not None and process.poll() is None:
                app.execution.lifecycle.reply(process, "complete\n")
            await asyncio.wait_for(asyncio.shield(app.workers.wait_for_complete()), timeout=5)
            await pilot.pause()
        assert app.preflight.latest_download.state == 'complete'
        assert 'ACPROF_DOWNLOAD ' not in app.query_one('#run-log', SelectableLog).raw_text


async def test_download_message_in_measurement_window_fails_without_updating_panel(tmp_path):
    script = '''
import signal
import time
from acprof.events.progress import emit_event
from acprof.events.download import DownloadProgress, encode_event
signal.signal(signal.SIGINT, signal.default_int_handler)
emit_event('case_started', 'fixture')
emit_event('measurement_started', 'fixture')
progress = DownloadProgress('example/test', 'huggingface', 'downloading', 100, 0, 10, 10, None, 0, 1)
print(encode_event(progress), flush=True)
time.sleep(10)
'''
    app = AcprofTui(settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(120, 30)) as pilot:
        app.launch(PendingLaunch((sys.executable, '-u', '-c', script), 'run'))
        await asyncio.wait_for(asyncio.shield(app.workers.wait_for_complete()), timeout=45)
        await pilot.pause()
        assert app.execution.lifecycle.process is None
        assert app.preflight.latest_download is None
        assert not app.execution.session.snapshot.measurement_active
        assert not app.query_one('#run-log', SelectableLog).suspended
        assert 'download progress inside measurement window' in app.query_one('#run-log', SelectableLog).raw_text
