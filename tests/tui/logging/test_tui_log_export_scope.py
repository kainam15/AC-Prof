"""Export original records with the current filters or an explicit retained scope."""
from dataclasses import replace
from unittest.mock import patch

import pytest
from rich.cells import cell_len
from textual.widgets import Button, Select, Static

from acprof.tui.logging.widget import SelectableLog
from acprof.tui.run.progress import ProgressSnapshot
from acprof.tui.run.stages import Stage
from acprof.tui.ui.input import BarCursorInput as Input
from tests.support.tui_fixtures import AcprofTui


def test_filtered_lines_keep_duplicates_json_blank_lines_and_folded_details():
    log = SelectableLog(details=False)
    lines = (
        '[WARN] retry', '[WARN] retry', '',
        '[network-preflight] {"path": "/原始/完整/目录", "ok": true}',
        '[DEBUG] hidden detail', '#1 0.2 build detail',
    )
    log.write('\n'.join(lines))
    assert '×2' in log.text
    assert 'hidden detail' not in log.text
    assert log.filtered_lines == lines
    log.set_view(query='完整')
    assert log.filtered_lines == (lines[3],)
    log.set_view(level='DEBUG', query='hidden')
    assert log.filtered_lines == (lines[4],)


def test_error_export_keeps_matching_build_context_and_traceback_in_order():
    log = SelectableLog()
    log.write('[cmd] build\n#1 0.2 compiler context\n#1 ERROR: compiler failed\n'
              '[ERROR] compiler exception\n  compiler traceback\n[INFO] compiler done')
    log.set_view(level='ERROR', query='compiler')
    assert log.filtered_lines == (
        '#1 0.2 compiler context', '#1 ERROR: compiler failed',
        '[ERROR] compiler exception', '  compiler traceback',
    )


@pytest.mark.parametrize(('language', 'scope', 'wrap'), [('zh', 'filtered', True), ('en', 'all', False)])
async def test_scope_preview_and_export_share_original_snapshot(tmp_path, language, scope, wrap):
    app = AcprofTui(settings_path=tmp_path / 'settings.json')
    app.configuration.preferences = replace(app.configuration.preferences, language=language, log_wrap=wrap)
    lines = (
        '[INFO] retry [a.b] excluded by level',
        '[WARN] Retry [a.b] first',
        '\x1b[33m[WARN] retry [a.b] ' + '模型/long-path/' * 15 + '\x1b[0m',
        '[WARN] retry axb excluded by literal query',
        '[WARN] Retry [a.b] first',
        '[DEBUG] hidden detail',
    )
    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab('monitor-tab')
        app.logs.toggle_tools()
        log = app.query_one('#run-log', SelectableLog)
        log.max_lines = 6
        app.logs.append('[WARN] retry [a.b] evicted\n' + '\n'.join(lines))
        app.query_one('#log-level', Select).value = 'WARN'
        app.query_one('#log-search', Input).value = 'rEtRy [a.b]'
        await pilot.pause()
        assert log.level_filter == 'WARN'
        assert log.search_query == 'rEtRy [a.b]'
        assert log.discarded_lines == 1
        assert await pilot.click('#export-log')
        await pilot.pause()
        scope_select = app.screen.query_one('#log-export-scope', Select)
        assert scope_select.value == 'filtered'
        label = scope_select.query_one('SelectCurrent #label', Static)
        assert str(label.content) == ('导出筛选结果（3 行）' if language == 'zh' else 'Export filtered results (lines: 3)')
        if scope == 'all':
            assert await pilot.click('#log-export-scope SelectCurrent')
            await pilot.press('end', 'enter')
            await pilot.pause()
            assert scope_select.value == 'all'
            assert str(label.content) == 'Export all retained logs (lines: 6)'
        # New output and a later filter change cannot change the previewed export.
        log.write('[WARN] retry [a.b] arrived after preview')
        log.set_view(level='ERROR', query='missing')
        for size in ((120, 30), (80, 24)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            for selector in ('#log-export-scope', '#log-export-save', '#log-export-cancel'):
                widget = app.screen.query_one(selector)
                assert widget.region.width > 0
                assert widget.region.height > 0
                assert app.screen.region.contains_region(widget.region)
            assert cell_len(str(label.content)) <= label.content_region.width
        target = tmp_path / 'exports'
        app.screen.query_one('#log-export-directory', Input).value = str(target)
        filename = app.screen.query_one('#log-export-filename', Input)
        filename.value = 'snapshot.log'
        filename.focus()
        await pilot.press('enter')
        await app.workers.wait_for_complete()
        await pilot.pause()
        expected = (lines[1], lines[2], lines[4]) if scope == 'filtered' else lines
        assert (target / 'snapshot.log').read_bytes() == '\n'.join(expected).encode('utf-8')
        assert not app.read_jobs


@pytest.mark.parametrize('language', ['zh', 'en'])
async def test_empty_filter_blocks_save_but_can_export_all_retained_lines(tmp_path, language):
    app = AcprofTui(settings_path=tmp_path / 'settings.json')
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        log = app.query_one('#run-log', SelectableLog)
        log.write('[INFO] retained')
        log.set_view(level='ERROR', query='missing')
        app.logs.export()
        await pilot.pause()
        target = tmp_path / 'exports'
        app.screen.query_one('#log-export-directory', Input).value = str(target)
        filename = app.screen.query_one('#log-export-filename', Input)
        filename.value = 'all.log'
        assert app.screen.query_one('#log-export-save', Button).disabled
        label = app.screen.query_one('#log-export-scope SelectCurrent #label', Static)
        assert str(label.content) == ('导出筛选结果（0 行）' if language == 'zh' else 'Export filtered results (lines: 0)')
        await pilot.press('enter')
        await pilot.pause()
        assert app.screen.id == 'log-export-screen'
        assert not target.exists()
        assert not (tmp_path / 'settings.json').exists()
        assert await pilot.click('#log-export-scope SelectCurrent')
        await pilot.press('end', 'enter')
        await pilot.pause()
        assert not app.screen.query_one('#log-export-save', Button).disabled
        assert str(label.content) == ('导出全部保留日志（1 行）' if language == 'zh' else 'Export all retained logs (lines: 1)')
        assert await pilot.click('#log-export-save')
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert (target / 'all.log').read_text() == '[INFO] retained'
        # Each new dialog starts with the current filters, regardless of the last scope.
        app.logs.export()
        await pilot.pause()
        assert app.screen.query_one('#log-export-scope', Select).value == 'filtered'
        await pilot.press('escape')
        assert not app.read_jobs


async def test_export_during_measurement_does_not_project_or_start_io(tmp_path):
    app = AcprofTui(settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        log = app.query_one('#run-log', SelectableLog)
        log.write('[INFO] retained')
        app.consume_process_line('', ProgressSnapshot(stage=Stage.MEASURING, measurement_active=True), True)
        with patch.object(log.records, 'project', side_effect=AssertionError('projected during measurement')), \
             patch('acprof.tui.logging.controller.write_log_export', side_effect=AssertionError('wrote during measurement')):
            app.logs.export()
            await pilot.pause()
        assert app.screen.id == '_default'
        assert not app.read_jobs
        assert not list(tmp_path.rglob('*.log'))
