"""The local search picker supports narrow layouts and cancellation."""
import asyncio
import json
import os.path
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from rich.text import Text
from textual.app import App
from textual.widgets import Button, Input, Static

from acprof.platform import is_linux
from acprof.tui.experiments.catalog import scan_experiments
from acprof.tui.experiments.picker import PickerChoice, SearchPickerScreen, experiment_choice
from acprof.tui.i18n import translate


class PickerHarness(App):
    def __init__(self, language='zh'):
        super().__init__()
        self.language = language

    def tr(self, text):
        return translate(text, self.language)


@pytest.mark.parametrize('size', [(80, 24), (120, 30)])
async def test_search_select_actions_and_disabled_resume(size):
    from acprof.tui.experiments.picker import PickerChoice, SearchPickerScreen
    class Harness(App):
        def tr(self, text):
            return str(text)
    app, selections = Harness(), []
    choices = tuple(PickerChoice(i, (f'demo/model-{i}', 'failed'), f'reason-{i}',
        f'model-{i} failed', frozenset({'view', 'reuse'})) for i in range(40))
    screen = SearchPickerScreen('选择实验', ('Model', 'Status'),
        lambda cancelled: (choices, ()), actions=('view', 'reuse', 'resume'))
    async with app.run_test(size=size) as pilot:
        app.push_screen(screen, selections.append)
        await app.workers.wait_for_complete()
        await pilot.pause()
        screen.query_one('#picker-search', Input).value = 'model-18 failed'
        await pilot.pause()
        assert (screen.query_one('#picker-resume', Button).disabled)
        assert (await pilot.click('#picker-reuse'))
        await pilot.pause()
        assert (selections[0]) == (('reuse', 18))

async def test_close_cancels_loading_and_releases_guard_after_worker_finishes():
    from acprof.host.run_state import RunStateError, offline_analysis_lock
    from tests.support.tui_waits import wait_for_condition

    started = threading.Event()
    finish_read = threading.Event()
    busy = []

    def loader(cancelled):
        started.set()
        assert finish_read.wait(10)
        assert cancelled()
        return (), ()

    app = PickerHarness()
    screen = SearchPickerScreen('选择实验', ('Model',), loader, loading_changed=busy.append)
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(screen)
        try:
            await wait_for_condition(pilot, started.is_set, description="catalog read started", timeout=5)
            await pilot.press('escape')
            assert screen not in app.screen_stack
            assert busy == [True]
            assert screen.cancelled.is_set()
            if is_linux():
                with pytest.raises(RunStateError, match="测量锁"), offline_analysis_lock():
                    pytest.fail("dismissed picker released the measurement lock before its worker returned")
        finally:
            finish_read.set()
        await wait_for_condition(pilot, lambda: busy == [True, False], description="catalog read finished", timeout=5)
        with offline_analysis_lock():
            pass


@pytest.mark.parametrize('choices_case', range(2), ids=['()', '(candidate,)'])
@pytest.mark.parametrize(('language', 'empty', 'unmatched'), [('zh', '尚无本地实验记录，请检查搜索目录。', '没有匹配项，请修改搜索词。'), ('en', 'No local experiment records yet. Check the search directories.', 'No matches. Change the search terms.')])
async def test_empty_catalog_and_unmatched_query_have_distinct_messages(choices_case, language, empty, unmatched):
    candidate = PickerChoice('fixture', ('demo/model',), 'fixture details', 'demo/model', frozenset({'view'}))
    choices = tuple(((), (candidate,)))[choices_case]
    app = PickerHarness(language)
    screen = SearchPickerScreen('选择实验', ('模型',),
        lambda cancelled: (choices, ()), query='asdf', actions=('view',))
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(screen)
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert (screen.query_one('#picker-detail', Static).content) == (unmatched if choices else empty)
        assert (screen.query_one('#picker-status', Static).content) == (f'0 / {len(choices)}')
        assert (screen.query_one('#picker-view', Button).disabled)
        if choices:
            screen.query_one('#picker-search', Input).value = ''
            await pilot.pause()
            assert (screen.query_one('#picker-detail', Static).content) == ('fixture details')
            assert not (screen.query_one('#picker-view', Button).disabled)

async def test_query_changes_keep_loading_feedback_until_records_arrive():
    release = threading.Event()

    def loader(_cancelled):
        if not release.wait(timeout=5):
            raise TimeoutError('test did not release picker loader')
        return (), ()

    app = PickerHarness('en')
    screen = SearchPickerScreen('选择实验', ('模型',), loader)
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await app.push_screen(screen)
            screen.query_one('#picker-search', Input).value = 'asdf'
            await pilot.pause()
            assert (screen.query_one('#picker-status', Static).content) == ('Reading known experiment roots…')
            assert (screen.query_one('#picker-detail', Static).content) == ('Reading known experiment roots…')
            release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert (screen.query_one('#picker-detail', Static).content) == ('No local experiment records yet. Check the search directories.')
    finally:
        release.set()

async def test_read_failure_preserves_diagnostics_instead_of_claiming_no_history():
    def loader(_cancelled):
        raise OSError('fixture catalog read failure')

    app = PickerHarness('en')
    screen = SearchPickerScreen('选择实验', ('模型',), loader)
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(screen)
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert (screen.query_one('#picker-status', Static).content) == ('0 / 0 · fixture catalog read failure')
        assert (screen.query_one('#picker-detail', Static).content) == ('No local records were read. See the messages above.')


async def test_bad_manifest_warning_keeps_healthy_experiment_selectable(tmp_path):
    good, bad = tmp_path / 'good', tmp_path / 'bad'
    good.mkdir()
    bad.mkdir()
    (good / 'static_meta.json').write_text('{"model_name":"demo/healthy"}', encoding='utf-8')
    (bad / 'result_manifest.json').write_text('{broken', encoding='utf-8')

    def loader(cancelled):
        catalog = scan_experiments([tmp_path], cancelled=cancelled)
        return tuple(experiment_choice(record) for record in catalog.records), catalog.warnings

    app, selections = PickerHarness(), []
    screen = SearchPickerScreen('选择实验', ('模型', '日期', '设备', '状态', 'Run ID'),
                                loader, actions=('select',))
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(screen, selections.append)
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert 'result_manifest.json' in str(screen.query_one('#picker-status', Static).content)
        assert not screen.query_one('#picker-select', Button).disabled
        assert await pilot.click('#picker-select')
        await pilot.pause()
    assert selections[0][0] == 'select'
    assert selections[0][1].directory == good

async def test_scope_omits_stale_paths_but_keeps_explicit_deep_scan_roots():
    scratch = Path.cwd() / 'internal-testing' / 'shared'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='picker-scope-', dir=scratch) as temporary:
        root = Path(temporary)
        results = root / 'results'
        deep = results.joinpath(*('nested' for _ in range(8)), 'demo--deep')
        deep.mkdir(parents=True)
        (deep / 'static_meta.json').write_text(json.dumps({'model_name': 'demo/deep'}))
        stale = results / 'google--gemma-4-26B-A4B-it'
        alias = root / 'alias'
        alias.symlink_to(results, target_is_directory=True)
        regular_file = root / 'not-a-directory'
        regular_file.touch()
        roots = (deep, stale, results, alias, results, regular_file)

        def loader(cancelled):
            catalog = scan_experiments(roots, cancelled=cancelled)
            return tuple(map(experiment_choice, catalog.records)), catalog.warnings

        app = PickerHarness()
        screen = SearchPickerScreen('选择实验', ('模型', '日期', '设备', '状态', 'Run ID'),
            loader, scope=roots)
        async with app.run_test(size=(80, 24)) as pilot:
            await app.push_screen(screen)
            await app.workers.wait_for_complete()
            await pilot.pause()
            scope = screen.query_one('#picker-scope', Static)
            assert (scope.content) == (f'搜索范围: {results.relative_to(Path.cwd())}')
            tooltip = cast(Text, scope.tooltip).plain
            assert (str(deep)) in (tooltip)
            assert (str(stale)) in (tooltip)
            assert (str(regular_file)) in (tooltip)
            assert (tooltip.count(str(results) + '\n')) == (1)
            note = screen.query_one('#picker-scope-note', Static)
            assert (note.content) == ('已跳过 2 个不可用目录；悬停查看详情。')
            assert (note.display)
            assert ([choice.cells[0] for choice in screen.choices]) == (['demo/deep'])
            for size in ((120, 30), (150, 45), (80, 24)):
                await pilot.resize_terminal(*size)
                await pilot.pause()
                assert (screen.query_one('#picker-table').size.height) > (0)
                close = screen.query_one('#picker-close', Button)
                assert (close.region.width) > (0)
                assert (close.region.right) <= (size[0])
                assert (close.region.bottom) <= (size[1])
            assert (await pilot.click('#picker-close'))

@pytest.mark.parametrize(('language', 'expected'), [('zh', '搜索范围: 无可用目录'), ('en', 'Search roots: No available directories')])
async def test_scope_reports_when_no_search_directory_exists(language, expected):
    with tempfile.TemporaryDirectory() as temporary:
        missing = Path(temporary) / 'deleted-results'
        app = PickerHarness(language)
        screen = SearchPickerScreen('选择实验', ('模型',), lambda cancelled: ((), ()), scope=(missing,))
        async with app.run_test(size=(80, 24)) as pilot:
            await app.push_screen(screen)
            await app.workers.wait_for_complete()
            await pilot.pause()
            scope = screen.query_one('#picker-scope', Static)
            assert (scope.content) == (expected)
            assert (str(missing)) in (cast(Text, scope.tooltip).plain)

async def test_scope_inspection_runs_in_worker_and_keeps_path_errors_in_tooltip():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        denied = root / '[denied]'
        denied.mkdir()
        inspection_threads = []

        def is_dir(path: Path) -> bool:
            if path == denied:
                inspection_threads.append(threading.get_ident())
                raise PermissionError('fixture permission denied')
            return os.path.isdir(path)

        app = PickerHarness('en')
        screen = SearchPickerScreen('选择实验', ('模型',), lambda cancelled: ((), ()), scope=(root, denied))
        with patch.object(Path, 'is_dir', is_dir):
            async with app.run_test(size=(80, 24)) as pilot:
                await app.push_screen(screen)
                await app.workers.wait_for_complete()
                await pilot.pause()
                scope = screen.query_one('#picker-scope', Static)
                assert (scope.content) == (f'Search roots: {root}')
                assert (f'Cannot inspect directory {denied}: fixture permission denied') in (cast(Text, scope.tooltip).plain)
                assert (len(inspection_threads)) == (1)
                assert (inspection_threads[0]) != (threading.get_ident())


async def test_picker_wide_split_layout_and_quick_status_filters():
    choices = (
        PickerChoice('ok-1', ('demo/ok', 'complete'), 'ok detail', 'demo/ok complete', frozenset({'view'})),
        PickerChoice('fail-1', ('demo/fail', 'failed'), 'fail detail', 'demo/fail failed', frozenset({'view', 'resume'})),
    )
    quick_filters = (
        ('all', '全部', lambda _choice: True),
        ('resumable', '可续跑', lambda choice: 'resume' in choice.actions),
    )
    app = PickerHarness('zh')
    screen = SearchPickerScreen(
        '选择实验', ('模型', '状态'), lambda _cancelled: (choices, ()),
        actions=('view', 'resume'), quick_filters=quick_filters,
    )
    async with app.run_test(size=(130, 32)) as pilot:
        await app.push_screen(screen)
        await app.workers.wait_for_complete()
        await pilot.pause()
        dialog = screen.query_one('#picker-dialog')
        assert dialog.has_class('wide-picker')
        assert screen.query_one('#picker-table').region.right <= screen.query_one('#picker-detail-scroll').region.x
        assert screen.query_one('#picker-view', Button).variant == 'primary'
        assert len(screen.filtered) == 2

        assert await pilot.click('#picker-filter-resumable')
        await pilot.pause()
        assert [choice.value for choice in screen.filtered] == ['fail-1']

        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert not dialog.has_class('wide-picker')


@pytest.mark.skipif(not is_linux(), reason="The cross-process measurement lock is Linux-specific")
@pytest.mark.parametrize(("picker", "language", "hint"), [
    ("model", "zh", "测量锁"), ("experiment", "en", "measurement lock"),
])
async def test_external_measurement_blocks_picker_reads_until_reopened(tmp_path, picker, language, hint):
    import sys

    from acprof.experiments.configuration import RunConfig
    from acprof.host import run_state
    from acprof.tui.experiments.catalog import ExperimentCatalog
    from tests.support.tui_fixtures import AcprofTui

    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    controller = "models" if picker == "model" else "experiments"
    open_picker = (app.models.open_model_candidates if picker == "model" else
                   lambda: app.catalog.open_experiment_picker("experiment-source"))
    lock_holder = """
import sys
from pathlib import Path
from acprof.host import run_state
run_state.MEASUREMENT_LOCK_ROOT = Path(sys.argv[1])
run_state.MEASUREMENT_LOCK_NAMESPACE = bytes.fromhex(sys.argv[2])
with run_state.MeasurementLock():
    print("locked", flush=True)
    sys.stdin.readline()
"""
    async with app.run_test(size=(120, 30)) as pilot:
        with patch(f"acprof.tui.{controller}.controller.scan_experiments", return_value=ExperimentCatalog(())) as scan, \
             patch("acprof.tui.models.candidates.cached_candidates", return_value=((), ())) as cache, \
             patch.object(SearchPickerScreen, "read_scope", return_value=((tmp_path,), ())) as scope:
            owner = await asyncio.create_subprocess_exec(
                sys.executable, "-c", lock_holder, str(run_state.MEASUREMENT_LOCK_ROOT),
                run_state.MEASUREMENT_LOCK_NAMESPACE.hex(),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
            assert owner.stdin is not None
            assert owner.stdout is not None
            try:
                assert (await asyncio.wait_for(owner.stdout.readline(), 10)).strip() == b"locked"
                open_picker()
                await app.workers.wait_for_complete()
                await pilot.pause()
                scope.assert_not_called()
                scan.assert_not_called()
                cache.assert_not_called()
                assert hint in str(app.screen.query_one("#picker-status", Static).content)
                await pilot.press("escape")
            finally:
                owner.stdin.close()
                await asyncio.wait_for(owner.wait(), 10)
            assert owner.returncode == 0
            open_picker()
            await app.workers.wait_for_complete()
            await pilot.pause()
            scope.assert_called_once()
            scan.assert_called_once()
            assert cache.call_count == (1 if picker == "model" else 0)
