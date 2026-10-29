"""Manual exports never overwrite files, change experiment layout or bypass the task gate."""
import asyncio
import errno
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, OptionList, Static

from acprof.artifact_layout import ArtifactLayout
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.log import SelectableLog
from acprof.tui.log_export import (
    default_log_export_dir,
    default_log_export_filename,
    list_export_directories,
    log_export_path,
    write_log_export,
)
from acprof.tui.settings import TuiSettings, UiPreferences, load_settings, save_settings
from tests.support.tui_fixtures import AcprofTui


def test_default_directory_uses_home_without_creating_it(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    with patch.object(Path, 'home', return_value=tmp_path / 'home'):
        assert default_log_export_dir() == tmp_path / 'home' / 'AC-Prof' / 'exports' / 'logs'
    assert list(tmp_path.iterdir()) == []


def test_filename_uses_local_wall_clock():
    with patch('acprof.tui.log_export.datetime') as clock:
        clock.now.return_value = datetime(2026, 10, 26, 14, 5, 6)
        assert default_log_export_filename() == 'acprof-tui-2026-10-26_14-05-06.log'
        clock.now.assert_called_once_with()


@pytest.mark.parametrize('filename', ['', '.', '..', '../other.log', '/other.log', 'dir/other.log', 'a\\b', 'bad\0.log'])
def test_filename_cannot_escape_the_selected_directory(tmp_path, filename):
    with pytest.raises(ValueError, match='文件名'):
        log_export_path(str(tmp_path), filename)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('directory', ['', 'relative/path', 'bad\0'])
def test_directory_requires_an_explicit_host_path(directory):
    with pytest.raises(ValueError, match='绝对目录'):
        log_export_path(directory, 'file.log')


def test_tilde_expansion():
    assert log_export_path('~/导出', '记录.log') == Path.home() / '导出' / '记录.log'


def test_concurrent_exports_never_overwrite_files_or_symlinks(tmp_path):
    path = tmp_path / 'same.log'
    original = tmp_path / 'original.log'
    original.write_text('original')
    path.symlink_to(original)
    (tmp_path / 'same-1.log').mkdir()
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda text: write_log_export(path, text), ['甲', '乙', '丙', '丁']))
    assert len(set(paths)) == 4
    assert {p.read_text(encoding='utf-8') for p in paths} == {'甲', '乙', '丙', '丁'}
    assert original.read_text() == 'original'
    assert path.is_symlink()
    assert (tmp_path / 'same-1.log').is_dir()
    assert {p.name for p in paths} == {f'same-{i}.log' for i in range(2, 6)}


@pytest.mark.parametrize('stage', ['mkdir', 'open', 'write', 'close'])
def test_failed_export_reports_io_error_and_removes_partial_file(tmp_path, stage):
    path = tmp_path / 'new' / 'log.txt'
    original_open = Path.open
    failure = OSError(errno.ENOSPC, 'No space left on device')

    class BrokenStream:
        def __init__(self, destination, *args, **kwargs):
            self.stream = original_open(destination, *args, **kwargs)

        def __enter__(self):
            return self

        def write(self, text):
            self.stream.write(text[:2])
            if stage == 'write':
                raise failure

        def __exit__(self, *args):
            self.stream.close()
            if stage == 'close':
                raise failure

    if stage in {'mkdir', 'open'}:
        mocked = patch.object(Path, stage, side_effect=failure)
    else:
        mocked = patch.object(Path, 'open', lambda destination, *args, **kwargs: BrokenStream(destination, *args, **kwargs))
    with mocked, pytest.raises(OSError, match='No space left'):
        write_log_export(path, 'partial data')
    assert not path.exists()


def test_export_keeps_experiment_log_layout_and_contents(tmp_path):
    layout = ArtifactLayout.for_new_run(tmp_path / 'experiment')
    layout.initialize()
    native_log = layout.path('terminal.log')
    native_log.write_text('native experiment evidence')
    before = {p.relative_to(layout.root): p.read_bytes() for p in layout.root.rglob('*') if p.is_file()}
    exported = write_log_export(tmp_path / 'exports' / 'history.log', 'manual history')
    assert native_log == tmp_path / 'experiment' / 'logs' / 'terminal.log'
    assert exported.read_text() == 'manual history'
    assert {p.relative_to(layout.root): p.read_bytes() for p in layout.root.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('language', ['zh', 'en'])
async def test_save_enter_remembers_directory_and_exposes_path_actions(tmp_path, language):
    settings_path = tmp_path / 'tui.json'
    app = AcprofTui(settings_path=settings_path)
    initial = TuiSettings(ui=UiPreferences(language=language), last_model='example/original',
                          last_result_dir='/saved/experiment', last_result_csv='/saved/experiment/result_all.csv')
    save_settings(settings_path, initial, app.project_dir)
    app.configuration.saved_settings = initial
    app.configuration.preferences = replace(initial.ui, log_wrap=False)
    target = tmp_path / '中文 dir' / 'new'
    raw = '[WARN] retry\n[WARN] retry\n#1 hidden 构建\n[INFO] complete'
    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab('monitor-tab')
        app.logs.toggle_tools()
        log = app.query_one('#run-log', SelectableLog)
        log.write(raw)
        log.set_view(level='ERROR', query='missing')
        await pilot.pause()
        assert await pilot.click('#export-log')
        await pilot.pause()
        assert app.screen.id == 'log-export-screen'
        assert app.focused.id == 'log-export-filename'
        for selector in ('#log-export-browse', '#log-export-cancel', '#log-export-save'):
            button = app.screen.query_one(selector, Button)
            assert button.region.width > 0
            assert button.region.height > 0
            assert app.screen.region.contains_region(button.region)
        host = str(app.screen.query_one('#log-export-host', Static).content)
        assert 'SSH' in host
        assert ('主机' if language == 'zh' else 'host') in host
        app.screen.query_one('#log-export-directory', Input).value = str(target)
        app.screen.query_one('#log-export-filename', Input).value = 'history.log'
        with patch.object(app, 'notify') as notify:
            await pilot.press('enter')
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any(call.args == (('日志已导出' if language == 'zh' else 'Logs exported'),)
                       for call in notify.call_args_list)
        exported = target / 'history.log'
        assert exported.read_bytes() == raw.encode('utf-8')
        assert not app.read_jobs
        assert await pilot.click('#copy-log-export-path')
        assert app.clipboard == str(exported)
        assert await pilot.click('#view-log-export-path')
        await pilot.pause()
        assert str(exported) in str(app.screen.query_one('#confirm-message', Static).content)
        await pilot.press('escape')
        restored, warning = load_settings(settings_path, app.project_dir)
        assert not warning
        assert restored == replace(initial, last_log_export_dir=str(target))
        before = settings_path.read_bytes()
        app.logs.export()
        await pilot.pause()
        app.screen.query_one('#log-export-directory', Input).value = str(tmp_path / 'cancelled')
        await pilot.click('#log-export-cancel')
        await pilot.pause()
        assert settings_path.read_bytes() == before
        assert not (tmp_path / 'cancelled').exists()
    restarted = AcprofTui(settings_path=settings_path)
    async with restarted.run_test(size=(80, 24)) as pilot:
        restarted.query_one('#run-log', SelectableLog).write('next launch')
        restarted.logs.export()
        await pilot.pause()
        assert restarted.screen.query_one('#log-export-directory', Input).value == str(target)
        await pilot.press('escape')


@pytest.mark.parametrize(('error', 'expected'), [
    (PermissionError(errno.EACCES, 'Permission denied'), '权限不足'),
    (OSError(errno.ENOSPC, 'No space left on device'), '磁盘空间不足'),
    (NotADirectoryError(errno.ENOTDIR, 'Not a directory'), 'Not a directory'),
])
async def test_export_failure_preserves_settings_and_allows_retry(tmp_path, error, expected):
    app = AcprofTui(settings_path=tmp_path / 'tui.json')
    initial = TuiSettings(last_log_export_dir=str(tmp_path / 'previous'))
    save_settings(app.configuration.settings_path, initial, app.project_dir)
    app.configuration.saved_settings = initial
    original = app.configuration.settings_path.read_bytes()
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one('#run-log', SelectableLog).write('data')
        app.logs.export()
        await pilot.pause()
        app.screen.query_one('#log-export-directory', Input).value = str(tmp_path / 'failed')
        with patch('acprof.tui.log_controller.write_log_export', side_effect=error), patch.object(app, 'notify') as notify:
            await pilot.click('#log-export-save')
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any(expected in str(call.args[0]) for call in notify.call_args_list)
        assert app.configuration.settings_path.read_bytes() == original
        assert app.configuration.saved_settings == initial
        assert not app.read_jobs
        assert app.operation_state().allows('run')
        assert app.logs.last_export_path is None
        app.logs.export()
        await pilot.pause()
        await pilot.click('#log-export-save')
        await app.workers.wait_for_complete()
        assert len(list((tmp_path / 'previous').glob('*.log'))) == 1


async def test_worker_keeps_measurement_gate_until_write_completes(tmp_path):
    app = AcprofTui(settings_path=tmp_path / 'tui.json')
    entered, release = threading.Event(), threading.Event()
    ui_thread = threading.get_ident()

    def delayed_write(path, text):
        assert threading.get_ident() != ui_thread
        entered.set()
        assert release.wait(10)
        return write_log_export(path, text)

    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one('#run-log', SelectableLog).write('retained')
        app.logs.export()
        await pilot.pause()
        app.screen.query_one('#log-export-directory', Input).value = str(tmp_path / 'out')
        with patch('acprof.tui.log_controller.write_log_export', side_effect=delayed_write):
            try:
                await pilot.press('enter')
                assert await asyncio.to_thread(entered.wait, 5)
                assert app.read_jobs
                assert app.query_one('#start-run', Button).disabled
                assert not app.check_action('request_run', ())
                with patch.object(app.recovery, 'review_run_destination') as launch:
                    app.action_request_run()
                    launch.assert_not_called()
                app.logs.export()  # Repeated clicks cannot queue a second write.
                assert app.screen.id == '_default'
            finally:
                release.set()
            await app.workers.wait_for_complete()
        await pilot.pause()
        assert not app.read_jobs
        assert app.operation_state().allows('run')


@pytest.mark.parametrize('language', ['zh', 'en'])
async def test_directory_browser_keyboard_navigation_and_resize(tmp_path, language):
    root = tmp_path / 'server'
    child = root / '子目录'
    child.mkdir(parents=True)
    (root / 'ordinary.log').write_text('not a directory')
    app = AcprofTui(settings_path=tmp_path / 'tui.json')
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one('#run-log', SelectableLog).write('data')
        app.logs.export()
        await pilot.pause()
        app.screen.query_one('#log-export-directory', Input).value = str(root / 'missing' / 'logs')
        assert await pilot.click('#log-export-browse')
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.screen.current == root
        assert app.screen.query_one(OptionList).option_count == 1
        assert not (root / 'missing').exists()
        assert app.focused.id == 'export-directory-list'
        await pilot.press('home', 'enter')
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.screen.current == child
        for size in [(120, 30), (150, 45), (80, 24)]:
            await pilot.resize_terminal(*size)
            await pilot.pause()
            button = app.screen.query_one('#export-directory-use', Button)
            assert app.screen.region.contains_region(button.region)
        assert await pilot.click('#export-directory-parent')
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.screen.current == root
        assert await pilot.click('#export-directory-use')
        await pilot.pause()
        assert app.screen.query_one('#log-export-directory', Input).value == str(root)
        await pilot.press('escape')
        assert not list(root.rglob('acprof-tui-*.log'))
        assert not (tmp_path / 'tui.json').exists()


async def test_cancelled_directory_read_still_blocks_measurement_until_thread_finishes(tmp_path):
    app = AcprofTui(settings_path=tmp_path / 'tui.json')
    entered, release = threading.Event(), threading.Event()

    def delayed_read(directory, **kwargs):
        entered.set()
        assert release.wait(10)
        return list_export_directories(directory, **kwargs)

    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one('#run-log', SelectableLog).write('data')
        app.logs.export()
        await pilot.pause()
        app.screen.query_one('#log-export-directory', Input).value = str(tmp_path)
        with patch('acprof.tui.log_controller.list_export_directories', side_effect=delayed_read):
            try:
                await pilot.click('#log-export-browse')
                assert await asyncio.to_thread(entered.wait, 5)
                await pilot.press('escape', 'escape')
                assert app.screen.id == '_default'
                assert app.read_jobs
                assert not app.operation_state().allows('run')
            finally:
                release.set()
            await app.workers.wait_for_complete()
        await pilot.pause()
        assert not app.read_jobs
        assert app.operation_state().allows('run')
        assert not list(tmp_path.rglob('*.log'))


async def test_directory_permission_error_is_visible_and_does_not_select_a_stale_path(tmp_path):
    app = AcprofTui(settings_path=tmp_path / 'tui.json')
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one('#run-log', SelectableLog).write('data')
        app.logs.export()
        await pilot.pause()
        app.screen.query_one('#log-export-directory', Input).value = str(tmp_path)
        await pilot.click('#log-export-browse')
        await app.workers.wait_for_complete()
        await pilot.pause()
        location = app.screen.query_one('#export-directory-location', Input)
        location.value = str(tmp_path / 'denied')
        location.focus()
        with patch('acprof.tui.log_controller.list_export_directories',
                   side_effect=PermissionError(errno.EACCES, 'Permission denied', str(tmp_path / 'denied'))):
            await pilot.press('enter')
            await app.workers.wait_for_complete()
        await pilot.pause()
        assert '权限不足' in str(app.screen.query_one('#export-directory-status', Static).content)
        assert location.value == str(tmp_path)
        assert await pilot.click('#export-directory-use')
        await pilot.pause()
        assert app.screen.query_one('#log-export-directory', Input).value == str(tmp_path)
        await pilot.press('escape')


async def test_invalid_filename_stays_in_dialog_without_side_effects(tmp_path):
    app = AcprofTui(settings_path=tmp_path / 'tui.json')
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one('#run-log', SelectableLog).write('data')
        app.logs.export()
        await pilot.pause()
        app.screen.query_one('#log-export-directory', Input).value = str(tmp_path / 'new')
        app.screen.query_one('#log-export-filename', Input).value = '../escape.log'
        await pilot.press('enter')
        await pilot.pause()
        assert app.screen.id == 'log-export-screen'
        assert '文件名' in str(app.screen.query_one('#log-export-error', Static).content)
        assert not (tmp_path / 'new').exists()
        assert app.read_jobs
        await pilot.press('escape')
        assert not app.read_jobs


@pytest.mark.parametrize('failure', ['unreadable_settings', 'settings_write'])
async def test_settings_failure_keeps_successful_export_available(tmp_path, failure):
    path = tmp_path / 'tui.json'
    path.write_text('{invalid' if failure == 'unreadable_settings' else '{"version": 4, "last_model": "old/model"}')
    original = path.read_bytes()
    app = AcprofTui(settings_path=path)
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one('#run-log', SelectableLog).write('data')
        app.logs.export()
        await pilot.pause()
        app.screen.query_one('#log-export-directory', Input).value = str(tmp_path / 'exports')
        with patch('acprof.tui.configuration_controller.save_settings', side_effect=PermissionError('settings read-only')), \
             patch.object(app, 'notify') as notify:
            await pilot.press('enter')
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any(call.kwargs.get('title') == '自动记忆未保存' for call in notify.call_args_list)
            assert any(call.args == ('日志已导出',) for call in notify.call_args_list)
        assert path.read_bytes() == original
        assert app.logs.last_export_path.is_file()
        assert not app.read_jobs
        app.logs.copy_export_path()
        assert app.clipboard == str(app.logs.last_export_path)
        assert app.configuration.saved_settings.last_log_export_dir == ''
