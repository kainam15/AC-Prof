from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console
from textual.widgets import Button, Static
from tui_fixtures import AcprofTui

from acprof.download_events import DownloadProgress
from acprof.experiment import RunConfig
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.log import SelectableLog
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.progress_widgets import DownloadProgressPanel
from acprof.tui.stages import Stage


async def test_export_opens_save_dialog_and_cancel_has_no_side_effects(tmp_path):
    app = AcprofTui(settings_path=tmp_path / 'settings.json')
    with patch.object(Path, 'home', return_value=tmp_path / 'home'):
        async with app.run_test(size=(80, 24)) as pilot:
            app.query_one('#run-log', SelectableLog).write('保留日志')
            app.logs.export()
            await pilot.pause()
            assert app.screen.id == 'log-export-screen'
            assert app.screen.query_one('#log-export-directory', Input).value == str(
                tmp_path / 'home' / 'AC-Prof' / 'exports' / 'logs')
            assert not (tmp_path / 'home').exists()
            await pilot.press('escape')
            await pilot.pause()
            assert not app.read_jobs
            assert not (tmp_path / 'settings.json').exists()
            assert not list(tmp_path.rglob('*.log'))


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_measurement_warning_flood_is_bounded_and_reports_omissions(tmp_path, language):
    from acprof.tui.diagnostic_buffer import BoundedLogBuffer

    app = AcprofTui(settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        log = app.query_one("#run-log", SelectableLog)
        app.logs.pending = BoundedLogBuffer(max_entries=4, max_bytes=256, max_entry_bytes=64)
        app.consume_process_line("", ProgressSnapshot(stage=Stage.MEASURING, measurement_active=True), True)
        before = log.text
        with patch.object(log, "_sync_view", side_effect=AssertionError("rendered during measurement")):
            for text in ("[WARN] first", "[WARN] a", "[WARN] b", "[WARN] c", "[WARN] last", "x" * 65):
                app.logs.append(text)
            assert len(app.logs.pending) == 4
            assert app.logs.pending_size <= 256
            assert app.logs.timer is None
            await pilot.pause(0.3)
            assert log.text == before
        app.consume_process_line("", ProgressSnapshot(stage=Stage.CLEANING_CASE), True)
        await pilot.pause()
        assert "[WARN] first" in log.raw_text
        assert "[WARN] last" in log.raw_text
        assert "[WARN] a\n" not in log.raw_text
        assert ("省略 2 条" if language == "zh" else "omitted 2 log records") in log.raw_text
        assert ("其中 1 条" if language == "zh" else "including 1 oversized") in log.raw_text
        app.logs.flush()  # A new stage summary may have queued after the deferred batch.
        assert not app.logs.pending


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
        copy_all = app.query_one('#log-toolbar #copy-log-all', Button)
        assert str(copy_all.label) == ('复制全部' if language == 'zh' else 'Copy all')
        assert await pilot.click(copy_all)
        await pilot.pause()
        assert app.clipboard == '[INFO] start\n[WARN] retry  [×2]'
        await pilot.press('tab')
        assert app.focused.id == 'follow-log'
        await pilot.press('shift+tab')
        assert app.focused is copy_all
        assert await pilot.click('#log-tools')
        await pilot.pause()
        assert not app.query('#log-previous, #log-next, #log-details')
        for selector in ('#log-level', '#log-search', '#copy-log-all', '#export-log'):
            widget = app.query_one(selector)
            assert widget.region.width > 0
            assert widget.region.height > 0
            assert 0 <= widget.region.x < widget.region.right <= size[0]
            assert 0 <= widget.region.y < widget.region.bottom <= size[1]
        await pilot.click('#log-search')
        await pilot.press('tab')
        assert app.focused.id == 'export-log'
        await pilot.press('shift+tab')
        assert app.focused.id == 'log-search'
        await pilot.press('h', 'i', 'd', 'd', 'e', 'n', 'enter')
        await pilot.pause()
        assert log.selected_text == 'hidden'
        assert 'hidden 模型' in log.text
        await pilot.click('#copy-log-all')
        await pilot.pause()
        assert app.clipboard == '#1 0.2 hidden 模型'
        app.query_one('#log-search', Input).value = ''
        await pilot.pause()
        assert log.text == '[INFO] start\n[WARN] retry  [×2]'
        assert await pilot.click('#copy-log-all')
        await pilot.pause()
        assert app.clipboard == '[INFO] start\n[WARN] retry  [×2]'
        assert log.raw_text == raw


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
        assert app.query_one('#copy-log-all', Button).disabled
        assert app.logs.timer is None
        progress = DownloadProgress('example/test', 'huggingface', 'downloading', 100, 20, 20, None, None, 0, 2)
        with pytest.raises(RuntimeError, match='measurement window'):
            app.preflight.download_progress(progress)
        app.consume_process_line('', ProgressSnapshot(stage=Stage.CLEANING_CASE), True)
        await pilot.pause()
        assert '[INFO] waiting before window' in log.text
        assert '[WARN] deferred during window' in log.text
        assert not app.query_one('#log-search', Input).disabled
        assert not app.query_one('#copy-log-all', Button).disabled


async def test_download_panel_is_separate_and_failure_restores_logs(tmp_path):
    app = AcprofTui(settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab('monitor-tab')
        progress = DownloadProgress('example/test', 'huggingface', 'downloading', 1000, 200, 600, 100, 40, 1, 2)
        app.preflight.download_progress(progress)
        widget = app.query_one('#download-progress', DownloadProgressPanel)
        assert widget.display
        rendered = "".join(segment.text for segment in Console(width=80).render(widget.query_one(Static).render()))
        assert '60%' in rendered
        assert '100 B' in rendered
        assert 'ACPROF_DOWNLOAD' not in app.query_one('#run-log', SelectableLog).text
        app.preflight.finish_download(cancelled=True)
        assert '取消' in "".join(segment.text for segment in Console(width=80).render(widget.query_one(Static).render()))
        await pilot.pause()


async def test_export_copies_retained_raw_text_and_holds_operation_gate(tmp_path):
    app = AcprofTui(settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(120, 30)) as pilot:
        app.query_one('#run-log', SelectableLog).write('[INFO] original\n[WARN] warning')
        app.query_one('#run-log', SelectableLog).set_view(level='ERROR')
        app.logs.export()
        await pilot.pause()
        app.screen.query_one('#log-export-directory', Input).value = str(tmp_path / 'exports')
        assert app.read_jobs
        assert app.query_one('#start-run', Button).disabled
        await pilot.click('#log-export-save')
        await app.workers.wait_for_complete()
        await pilot.pause()
        exported, = (tmp_path / 'exports').glob('*.log')
        assert exported.read_text() == '[INFO] original\n[WARN] warning'
        assert not app.read_jobs
