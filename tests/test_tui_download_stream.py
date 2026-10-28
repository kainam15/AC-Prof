import asyncio
import sys

from tui_fixtures import AcprofTui

from acprof.tui.commands import PendingLaunch
from acprof.tui.log import SelectableLog


async def test_download_progress_arrives_before_child_finishes_without_log_spam(tmp_path):
    script = '''
import time
from dataclasses import replace
from acprof.download_events import DownloadProgress, encode_event
progress = DownloadProgress('example/test', 'huggingface', 'downloading', 1000, 200, 600, 100, 40, 1, 2)
print(encode_event(progress), flush=True)
time.sleep(1)
print(encode_event(replace(progress, state='complete', completed_bytes=1000, files_completed=2, rate_bytes_s=None)), flush=True)
'''
    app = AcprofTui(settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(120, 30)) as pilot:
        app.launch(PendingLaunch((sys.executable, '-u', '-c', script), 'test'))

        async def wait_for_progress():
            while app.preflight.latest_download is None:
                await pilot.pause()

        await asyncio.wait_for(wait_for_progress(), timeout=5)
        assert app.lifecycle.process is not None
        assert app.lifecycle.process.poll() is None
        assert app.preflight.latest_download.state == 'downloading'
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.preflight.latest_download.state == 'complete'
        assert 'ACPROF_DOWNLOAD ' not in app.query_one('#run-log', SelectableLog).raw_text


async def test_download_message_in_measurement_window_fails_without_updating_panel(tmp_path):
    script = '''
import signal
import time
from acprof.progress_events import emit_event
from acprof.download_events import DownloadProgress, encode_event
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
        assert app.lifecycle.process is None
        assert app.preflight.latest_download is None
        assert not app.latest_snapshot.measurement_active
        assert not app.query_one('#run-log', SelectableLog).suspended
        assert 'download progress inside measurement window' in app.query_one('#run-log', SelectableLog).raw_text
