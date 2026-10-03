"""Existing result pages discover records, reuse full options and review exact resume commands."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import test_experiment_catalog as catalog_fixture
from textual.widgets import Button, Input, Select, Static, TabbedContent

from acprof.experiment import RunConfig, RunConfigError, build_run_command
from acprof.run_args import build_parser
from acprof.tui.app import AcprofTui
from acprof.tui.experiment_catalog import scan_experiments
from acprof.tui.experiment_picker import SearchPickerScreen
from acprof.tui.settings import TuiSettings, load_settings, save_settings


class FrozenOptionTests(unittest.TestCase):
    def test_extra_options_are_typed_cannot_override_fields_and_survive_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = replace(RunConfig.smoke('demo/model'), revision='a' * 40,
                extra_options={'matrix_seed': 71, 'matrix_order': 'declared', 'latency_slo': ['default=3'],
                               'keep_compute_profiles': False})
            save_settings(root / 'settings.json', TuiSettings(run_defaults=config), root)
            restored, error = load_settings(root / 'settings.json', root)
            self.assertFalse(error)
            self.assertEqual(restored.run_defaults, config.validate(project_dir=root))
            command = build_run_command(restored.run_defaults, project_dir=root)
            parsed = build_parser().parse_args(command[command.index('--model'):])
            self.assertEqual(parsed.latency_slo, ['default=3'])
            self.assertFalse(parsed.keep_compute_profiles)
            from acprof.tui.commands import build_probe_command
            probe = build_probe_command(config, project_dir=root)
            self.assertEqual(probe[probe.index('--revision') + 1], config.revision)
            from acprof.tui.run_form import matches_preset
            self.assertFalse(matches_preset(config, 'smoke'))
            self.assertEqual(config.with_preset('main').revision, 'a' * 40)
            for extra in ({'model': 'other/model'}, {'output_dir': 'elsewhere'}, {'resume': True},
                          {'matrix_order': 'bad'}, {'latency_slo': 'not-list'}, {'matrix_seed': True}):
                with self.subTest(extra=extra), self.assertRaises(RunConfigError):
                    replace(config, extra_options=extra).validate(project_dir=root)


class CatalogWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = catalog_fixture.ExperimentCatalogTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    async def test_search_failed_run_reuse_full_config_and_confirm_frozen_resume(self):
        root = self.fixture.root
        path = self.fixture.record('batch-18', status='failed')
        (path / 'result_all.csv').unlink()
        (path / 'runtime_failures.json').write_text(json.dumps({'failures': [{'reason_code': 'resource_limit', 'detail': '4 GiB'}]}))
        for i in range(24):
            self.fixture.record(f'batch-{i}', model=f'demo/other-{i}', run_id=f'run-{i}', status='complete')
        config = replace(RunConfig.smoke('demo/model'), output_dir=str(root))
        app = AcprofTui(config, settings_path=root / 'settings.json')
        app.ui_preferences = replace(app.ui_preferences, language='en')
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app._activate_tab('plot-tab')
            await pilot.pause()
            button = app.query_one('#choose-result-csv', Button)
            self.assertTrue(await pilot.click(button))
            await app.workers.wait_for_complete()
            await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, SearchPickerScreen)
            for button in screen.query(Button):
                self.assertGreaterEqual(button.region.x, 0)
                self.assertLessEqual(button.region.right, 80)
                self.assertLessEqual(button.region.bottom, 24)
            screen.query_one('#picker-search', Input).value = 'demo/model failed 2026-10 cpu'
            await pilot.pause()
            self.assertEqual(len(screen.filtered), 1)
            self.assertTrue(await pilot.click('#picker-view'))
            await pilot.pause()
            self.assertIn('resource_limit', str(app.query_one('#result-summary', Static).content))
            record = scan_experiments([path]).records[0]
            app._experiment_selected('result-csv', ('reuse', record))
            await pilot.pause()
            reused = app._collect_config()
            self.assertFalse(reused.resume)
            self.assertEqual(reused.extra_options['matrix_seed'], 71)
            self.assertEqual(reused.revision, 'a' * 40)
            self.assertNotEqual(reused.output_dir, str(path.parent))
            self.assertEqual(app.query_one('#run-preset', Select).value, 'custom')
            self.assertIn('--matrix-seed 71', str(app.query_one('#command-preview', Static).content))
            with patch.object(app, '_launch') as launch:
                app._experiment_selected('result-csv', ('resume', record))
                await pilot.pause()
                launch.assert_not_called()
                self.assertIn('--resume', app._pending_launch.command)
                self.assertTrue(await pilot.click('#confirm-yes'))
                await pilot.pause()
                pending = launch.call_args.args[0]
                parsed = build_parser().parse_args(pending.command[pending.command.index('--model'):])
                self.assertEqual(parsed.matrix_seed, 71)
                self.assertEqual(parsed.output_dir, str(path.parent))
                self.assertTrue(parsed.resume)

    async def test_paths_are_selected_in_existing_pages_and_measurement_blocks_discovery(self):
        root = self.fixture.root
        path = self.fixture.record('batch')
        app = AcprofTui(replace(RunConfig.smoke('demo/model'), output_dir=str(root)), settings_path=root / 'settings.json')
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            record = scan_experiments([path]).records[0]
            for target in ('result-csv', 'result-dir', 'report-source', 'comparison-left', 'comparison-right'):
                app._experiment_selected(target, ('select', record))
                expected = path / 'result_all.csv' if target == 'result-csv' else path
                self.assertEqual(app.query_one('#' + target, Input).value, str(expected))
            app._latest_snapshot = replace(app._latest_snapshot, measurement_active=True)
            with patch('acprof.tui.catalog_actions.scan_experiments') as scan:
                app._open_experiment_picker('result-csv')
                await pilot.pause()
                scan.assert_not_called()
            self.assertNotIsInstance(app.screen, SearchPickerScreen)
            self.assertEqual(app.query_one('#main-tabs', TabbedContent).active, 'run-tab')
