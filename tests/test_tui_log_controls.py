from dataclasses import replace
from unittest.mock import patch

import pytest
from textual.widgets import Button, Checkbox, Static
from tui_fixtures import AcprofTui

from acprof.download_events import DownloadProgress
from acprof.experiment import RunConfig
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.log import SelectableLog
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.stages import Stage


@pytest.mark.parametrize(('size', 'language'), [((80, 24), 'zh'), ((80, 24), 'en'), ((120, 30), 'en')])
async def test_monitor_search_copy_and_controls_are_reachable(tmp_path, size, language):
    app = AcprofTui(RunConfig.smoke('example/test'), settings_path=tmp_path / 'settings.json')
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size) as pilot:
        app.activate_tab('monitor-tab')
        log = app.query_one('#run-log', SelectableLog)
        raw = '[INFO] start\n[WARN] retry\n[WARN] retry\n#1 0.2 hidden 模型'
        log.write(raw)
        await pilot.pause()
        assert '×2' in log.text
        assert await pilot.click('#log-tools')
        await pilot.pause()
        for selector in ('#log-search', '#copy-log-all', '#export-log', '#log-next', '#log-details'):
            widget = app.query_one(selector)
            assert widget.region.width > 0
            assert widget.region.height > 0
            assert 0 <= widget.region.x < widget.region.right <= size[0]
            assert 0 <= widget.region.y < widget.region.bottom <= size[1]
        await pilot.click('#log-search')
        await pilot.press('h', 'i', 'd', 'd', 'e', 'n', 'enter')
        await pilot.pause()
        assert log.selected_text == 'hidden'
        assert 'hidden 模型' in log.text
        await pilot.click('#copy-log-all')
        await pilot.pause()
        assert app.clipboard == raw
        app.query_one('#log-search', Input).value = ''
        app.query_one('#log-details', Checkbox).value = True
        await pilot.pause()
        assert log.text == raw


async def test_measurement_stops_pending_render_and_disallows_download_updates(tmp_path):
    app = AcprofTui(settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(120, 30)) as pilot:
        log = app.query_one('#run-log', SelectableLog)
        app.logs.append('[INFO] waiting before window')
        app.consume_process_line('', ProgressSnapshot(stage=Stage.MEASURING, measurement_active=True), True)
        before = log.text
        with patch.object(log, '_sync_view', side_effect=AssertionError('rendered during measurement')):
            app.logs.append('[WARN] deferred during window')
            assert app.logs.timer is None
            await pilot.pause(0.35)
            app.logs.view_changed()
            app.logs.find()
            app.logs.copy_all()
            assert log.text == before
        assert app.query_one('#log-search', Input).disabled
        assert app.logs.timer is None
        progress = DownloadProgress('example/test', 'huggingface', 'downloading', 100, 20, 20, None, None, 0, 2)
        with pytest.raises(RuntimeError, match='measurement window'):
            app.preflight.download_progress(progress)
        app.consume_process_line('', ProgressSnapshot(stage=Stage.CLEANING_CASE), True)
        await pilot.pause()
        assert '[INFO] waiting before window' in log.text
        assert '[WARN] deferred during window' in log.text
        assert not app.query_one('#log-search', Input).disabled


async def test_download_panel_is_separate_and_failure_restores_logs(tmp_path):
    app = AcprofTui(settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab('monitor-tab')
        progress = DownloadProgress('example/test', 'huggingface', 'downloading', 1000, 200, 600, 100, 40, 1, 2)
        app.preflight.download_progress(progress)
        widget = app.query_one('#download-progress', Static)
        assert widget.display
        assert '60%' in str(widget.content)
        assert '100 B' in str(widget.content)
        assert 'ACPROF_DOWNLOAD' not in app.query_one('#run-log', SelectableLog).text
        app.preflight.finish_download(cancelled=True)
        assert '取消' in str(widget.content)
        await pilot.pause()


async def test_export_copies_retained_raw_text_and_holds_operation_gate(tmp_path):
    app = AcprofTui(settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(120, 30)) as pilot:
        app.query_one('#run-log', SelectableLog).write('[INFO] original\n[WARN] warning')
        app.query_one('#run-log', SelectableLog).set_view(level='ERROR')
        app.logs.export()
        assert app.read_jobs
        assert app.query_one('#start-run', Button).disabled
        await app.workers.wait_for_complete()
        await pilot.pause()
        exported, = (tmp_path / 'logs').glob('*.log')
        assert exported.read_text() == '[INFO] original\n[WARN] warning'
        assert not app.read_jobs
