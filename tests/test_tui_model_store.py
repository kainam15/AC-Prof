"""Storage previews remain dismissible while another process owns the cache lock."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from test_model_store_gc import locked_store
from textual.widgets import Button, Input, Select, Static
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.tui.model_store import ModelStoreScreen


class ModelStoreCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_locked_preview_shows_capacity_then_closes_before_unlock(self):
        for language, size in (('zh', (80, 24)), ('en', (120, 40))):
            with self.subTest(language=language), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                app = AcprofTui(RunConfig.smoke('example/model'), settings_path=root / 'settings.json')
                with locked_store(root) as holder:
                    async with app.run_test(size=size) as pilot:
                        try:
                            app.query_one('#ui-language', Select).value = language
                            app.query_one('#model-store', Input).value = str(root)
                            app.open_model_store()
                            await pilot.pause()
                            self.assertIsInstance(app.screen, ModelStoreScreen)
                            screen = app.screen
                            content = str(screen.query_one('#store-report', Static).content)
                            self.assertIn('total_bytes', content)
                            self.assertIn(app.tr('缓存正在由另一任务处理；可以关闭窗口取消等待。'),
                                          str(screen.query_one('#store-status', Static).content))
                            self.assertFalse(screen.query_one('#store-close', Button).disabled)
                            status = screen.query_one('#store-status', Static)
                            self.assertGreaterEqual(status.region.y, 0)
                            self.assertLessEqual(status.region.bottom, screen.query_one('#store-close', Button).region.y)
                            self.assertTrue(app._is_busy())
                            if language == 'zh':
                                self.assertTrue(await pilot.click('#store-close'))
                            else:
                                await pilot.press('escape')
                            await app.workers.wait_for_complete()
                            await pilot.pause()
                            self.assertIsNone(holder.poll(), 'cache owner must still be holding the lock')
                            self.assertNotIsInstance(app.screen, ModelStoreScreen)
                            self.assertFalse(app._is_busy())
                        finally:
                            holder.stdin.close()
                            await app.workers.wait_for_complete()

    async def test_closing_retains_busy_state_until_the_reader_releases_resources(self):
        started, acknowledged, release = threading.Event(), threading.Event(), threading.Event()

        def scan(root, *, cancel):
            started.set()
            if not cancel.wait(5):
                raise AssertionError('close did not signal cancellation')
            acknowledged.set()
            if not release.wait(5):
                raise AssertionError('test did not release its simulated reader')
            return {'path': str(root), 'total_bytes': 0, 'free_bytes': 1000, 'models': []}

        with tempfile.TemporaryDirectory() as temporary, patch('acprof.tui.model_store.disk_report', side_effect=scan):
            root = Path(temporary)
            app = AcprofTui(RunConfig.smoke('example/model'), settings_path=root / 'settings.json')
            async with app.run_test(size=(80, 24)) as pilot:
                try:
                    app.query_one('#model-store', Input).value = str(root)
                    app.open_model_store()
                    await pilot.pause()
                    self.assertTrue(started.is_set())
                    self.assertTrue(await pilot.click('#store-close'))
                    await pilot.pause()
                    self.assertTrue(acknowledged.is_set())
                    self.assertIsInstance(app.screen, ModelStoreScreen)
                    self.assertTrue(app._is_busy())
                finally:
                    release.set()
                    await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertNotIsInstance(app.screen, ModelStoreScreen)
                self.assertFalse(app._is_busy())
