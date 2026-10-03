"""The local search picker supports narrow layouts and cancellation."""
import asyncio
import threading
import unittest

from textual.app import App
from textual.widgets import Button, Input


class PickerTests(unittest.IsolatedAsyncioTestCase):
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
