"""The local search picker supports narrow layouts and cancellation."""
import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Callable, cast
from unittest.mock import patch

from rich.text import Text
from textual.app import App
from textual.widgets import Button, Input, Static

from acprof.tui.experiment_catalog import scan_experiments
from acprof.tui.experiment_picker import SearchPickerScreen, experiment_choice
from acprof.tui.i18n import translate


class PickerHarness(App):
    def __init__(self, language='zh'):
        super().__init__()
        self.language = language

    def tr(self, text):
        return translate(text, self.language)


class PickerTests(unittest.IsolatedAsyncioTestCase):
    async def test_scope_omits_stale_paths_but_keeps_explicit_deep_scan_roots(self):
        scratch = Path.cwd() / 'internal-testing'
        scratch.mkdir(exist_ok=True)
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
                self.assertEqual(scope.content, f'搜索范围: {results.relative_to(Path.cwd())}')
                tooltip = cast(Text, scope.tooltip).plain
                self.assertIn(str(deep), tooltip)
                self.assertIn(str(stale), tooltip)
                self.assertIn(str(regular_file), tooltip)
                self.assertEqual(tooltip.count(str(results) + '\n'), 1)
                note = screen.query_one('#picker-scope-note', Static)
                self.assertEqual(note.content, '已跳过 2 个不可用目录；悬停查看详情。')
                self.assertTrue(note.display)
                self.assertEqual([choice.cells[0] for choice in screen.choices], ['demo/deep'])
                for size in ((120, 30), (150, 45), (80, 24)):
                    await pilot.resize_terminal(*size)
                    await pilot.pause()
                    self.assertGreater(screen.query_one('#picker-table').size.height, 0)
                    close = screen.query_one('#picker-close', Button)
                    self.assertGreater(close.region.width, 0)
                    self.assertLessEqual(close.region.right, size[0])
                    self.assertLessEqual(close.region.bottom, size[1])
                self.assertTrue(await pilot.click('#picker-close'))

    async def test_scope_reports_when_no_search_directory_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / 'deleted-results'
            for language, expected in (('zh', '搜索范围: 无可用目录'),
                                       ('en', 'Search roots: No available directories')):
                with self.subTest(language=language):
                    app = PickerHarness(language)
                    screen = SearchPickerScreen('选择实验', ('模型',), lambda cancelled: ((), ()), scope=(missing,))
                    async with app.run_test(size=(80, 24)) as pilot:
                        await app.push_screen(screen)
                        await app.workers.wait_for_complete()
                        await pilot.pause()
                        scope = screen.query_one('#picker-scope', Static)
                        self.assertEqual(scope.content, expected)
                        self.assertIn(str(missing), cast(Text, scope.tooltip).plain)

    async def test_scope_inspection_runs_in_worker_and_keeps_path_errors_in_tooltip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            denied = root / '[denied]'
            denied.mkdir()
            original_is_dir: Callable[[Path], bool] = Path.is_dir
            inspection_threads = []

            def is_dir(path: Path) -> bool:
                if path == denied:
                    inspection_threads.append(threading.get_ident())
                    raise PermissionError('fixture permission denied')
                return original_is_dir(path)

            app = PickerHarness('en')
            screen = SearchPickerScreen('选择实验', ('模型',), lambda cancelled: ((), ()), scope=(root, denied))
            with patch.object(Path, 'is_dir', is_dir):
                async with app.run_test(size=(80, 24)) as pilot:
                    await app.push_screen(screen)
                    await app.workers.wait_for_complete()
                    await pilot.pause()
                    scope = screen.query_one('#picker-scope', Static)
                    self.assertEqual(scope.content, f'Search roots: {root}')
                    self.assertIn(f'Cannot inspect directory {denied}: fixture permission denied', cast(Text, scope.tooltip).plain)
                    self.assertEqual(len(inspection_threads), 1)
                    self.assertNotEqual(inspection_threads[0], threading.get_ident())

    async def test_search_select_actions_and_disabled_resume(self):
        from acprof.tui.experiment_picker import PickerChoice, SearchPickerScreen
        class Harness(App):
            def tr(self, text):
                return str(text)
        for size in ((80, 24), (120, 30)):
            with self.subTest(size=size):
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
                    self.assertTrue(screen.query_one('#picker-resume', Button).disabled)
                    self.assertTrue(await pilot.click('#picker-reuse'))
                    await pilot.pause()
                    self.assertEqual(selections[0], ('reuse', 18))

    async def test_close_cancels_loading_and_releases_guard_after_worker_finishes(self):
        from acprof.tui.experiment_picker import SearchPickerScreen
        class Harness(App):
            def tr(self, text):
                return str(text)
        started = threading.Event()
        busy = []
        def loader(cancelled):
            started.set()
            while not cancelled():
                threading.Event().wait(0.005)
            return (), ()
        app = Harness()
        screen = SearchPickerScreen('选择实验', ('Model',), loader, loading_changed=busy.append)
        async with app.run_test(size=(80, 24)) as pilot:
            app.push_screen(screen)
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(started.is_set())
            await pilot.press('escape')
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertNotIn(screen, app.screen_stack)
            self.assertEqual(busy, [True, False])
