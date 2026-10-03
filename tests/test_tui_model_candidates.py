"""One model input selects local, condition-bound evidence without losing free text."""
import json
import unittest
from dataclasses import replace
from typing import cast
from unittest.mock import patch

import test_model_candidates as candidate_fixture
from rich.text import Text
from textual.widgets import Button, Input, Static
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.tui.experiment_picker import SearchPickerScreen
from acprof.tui.model_candidates import record_conditions
from acprof.tui.settings import TuiSettings, save_settings


class ModelInputTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = candidate_fixture.ModelCandidateTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    async def test_saved_missing_result_is_hidden_without_rewriting_settings(self):
        root = self.fixture.fixture.root
        results, store = root / 'results', root / 'store'
        results.mkdir()
        store.mkdir()
        stale = results / 'google--gemma-4-26B-A4B-it'
        settings_path = root / 'settings.json'
        save_settings(settings_path, TuiSettings(last_result_dir=str(stale),
            last_result_csv=str(stale / 'result_all.csv')), root)
        original_settings = settings_path.read_bytes()
        config = replace(RunConfig.smoke('asdf'), output_dir=str(results), model_store=str(store))
        with patch('acprof.tui.app.PROJECT_DIR', root):
            app = AcprofTui(config, settings_path=settings_path)
            async with app.run_test(size=(120, 30)) as pilot:
                app.query_one('#model', Input).focus()
                await pilot.press('f4')
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertIsInstance(app.screen, SearchPickerScreen)
                screen = cast(SearchPickerScreen, app.screen)
                scope = screen.query_one('#picker-scope', Static)
                self.assertEqual(scope.content, '搜索范围: results · store')
                self.assertIn(str(stale), cast(Text, scope.tooltip).plain)
                self.assertEqual(screen.query_one('#picker-scope-note', Static).content,
                    '已跳过 1 个不可用目录；悬停查看详情。')
                self.assertEqual(screen.choices, ())
                self.assertEqual(screen.query_one('#picker-detail', Static).content,
                    '尚无本地模型记录，可直接输入 Hugging Face 模型 ID。')
                await pilot.press('escape')
                await pilot.pause()
                self.assertEqual(app.query_one('#model', Input).value, 'asdf')
                app.ui_preferences = replace(app.ui_preferences, language='en')
                app.query_one('#model', Input).focus()
                await pilot.press('f4')
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertEqual(app.screen.query_one('#picker-scope', Static).content, 'Search roots: results · store')
                self.assertEqual(app.screen.query_one('#picker-detail', Static).content,
                    'No local model records yet. Enter a Hugging Face model ID directly.')
                await pilot.press('escape')
        self.assertEqual(settings_path.read_bytes(), original_settings)

    async def test_same_input_searches_pins_revision_and_changes_conditions(self):
        record = self.fixture.recorded()
        root = self.fixture.fixture.root
        broken = self.fixture.fixture.record('broken', run_id='broken', model='demo/bad')
        metadata_path = broken / 'static_meta.json'
        metadata = json.loads(metadata_path.read_text())
        metadata['runtime_environment'] = ['malformed']
        metadata_path.write_text(json.dumps(metadata))
        config = replace(RunConfig.smoke('demo/model'), output_dir=str(root), model_store=str(root / 'store'))
        app = AcprofTui(config, settings_path=root / 'settings.json')
        def conditions(record, config, *args):
            return {**record_conditions(record), 'mems': config.mems}
        with patch('acprof.tui.model_candidates.current_context', return_value={'warnings': []}), \
             patch('acprof.tui.model_candidates.current_conditions', side_effect=conditions):
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                self.assertTrue(await pilot.click('#model-candidates'))
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertIsInstance(app.screen, SearchPickerScreen)
                self.assertIn('demo--bad', ' '.join(app.screen.warnings))
                self.assertEqual(app.screen.filtered[0].value.status, 'verified_current')
                self.assertTrue(await pilot.click('#picker-use'))
                await pilot.pause()
                self.assertEqual(app.query_one('#model', Input).value, record.model_id)
                self.assertEqual(app.query_one('#revision', Input).value, record.revision)
                self.assertIsNotNone(app.query_one('#model', Input).suggester)
                app.query_one('#mems', Input).value = '8'
                await pilot.pause()
                app.query_one('#model', Input).focus()
                await pilot.press('f4')
                await app.workers.wait_for_complete()
                await pilot.pause()
                candidate = app.screen.filtered[0].value
                self.assertEqual(candidate.status, 'revalidate')
                self.assertIn('mems: changed', candidate.reason)
                self.assertFalse(app.screen.query_one('#picker-use', Button).disabled)
                await pilot.press('escape')
                await pilot.pause()
                app.query_one('#model', Input).value = 'unlisted/new-model'
                await pilot.pause()
                self.assertEqual(app._collect_config().model, 'unlisted/new-model')
                self.assertEqual(app._collect_config().revision, '')

    async def test_measurement_blocks_candidate_io(self):
        root = self.fixture.fixture.root
        app = AcprofTui(RunConfig.smoke('demo/model'), settings_path=root / 'settings.json')
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            app._latest_snapshot = replace(app._latest_snapshot, measurement_active=True)
            with patch('acprof.tui.model_actions.scan_experiments') as scan:
                app.open_model_candidates()
                await pilot.pause()
                scan.assert_not_called()
            self.assertNotIsInstance(app.screen, SearchPickerScreen)
