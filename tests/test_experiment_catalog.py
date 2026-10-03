"""Rebuildable local discovery and exact frozen-argument recovery."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from acprof.run_args import build_parser


class ExperimentCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def record(self, name, *, run_id='run-a', status='interrupted', model='demo/model', revision='a' * 40):
        path = self.root / name / model.replace('/', '--')
        path.mkdir(parents=True)
        options = vars(build_parser().parse_args(['--model', model, '--cpus', '1', '--mems', '4', '--gpus', 'off',
            '--matrix-order', 'declared', '--matrix-seed', '71', '--latency-slo', 'default=3', '--no-prune-startup-oom']))
        for key in ('resume', 'output_dir', 'skip_build', 'notify'):
            options.pop(key)
        options['revision'] = revision
        options['measurement_environment'] = {'execution_environment': 'native_linux'}
        state = {'schema_version': 1, 'run_id': run_id, 'status': status, 'created_at': '2026-10-01T12:00:00Z',
            'options': options, 'artifacts': {}, 'runtime': {'task': {'model_id': model, 'model_revision': revision,
                'runtime_profile_id': 'transformers-cpu'}, 'image': {'runtime_environment': {'environment_id': 'env-a'}}}}
        (path / 'run_state.json').write_text(json.dumps(state))
        (path / 'static_meta.json').write_text(json.dumps({'model_name': model, 'model_revision': revision, 'cpu_model': 'Fixture CPU'}))
        (path / 'result_all.csv').write_text('fixture\n')
        return path

    def test_search_many_records_by_model_date_device_status_and_keep_run_identity(self):
        from acprof.tui.experiment_catalog import scan_experiments
        for i in range(25):
            self.record(f'batch-{i}', run_id=f'run-{i}', status='failed' if i == 18 else 'complete', model=f'demo/model-{i}')
        report = scan_experiments([self.root])
        self.assertEqual(len(report.records), 25)
        found = report.search('model-18 2026-10 failed cpu')
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].run_id, 'run-18')
        renamed = self.root / 'moved' / 'renamed'
        renamed.parent.mkdir()
        found[0].directory.rename(renamed)
        selected = scan_experiments([self.root]).search('model-18')[0]
        self.assertEqual(selected.run_id, 'run-18')
        self.assertEqual(selected.directory, renamed)

    def test_duplicate_run_collapses_aliases_and_content_conflict_is_explicit(self):
        from acprof.tui.experiment_catalog import scan_experiments
        original = self.record('a')
        copy = self.root / 'b' / original.name
        shutil.copytree(original, copy)
        report = scan_experiments([self.root])
        self.assertEqual(len(report.records), 1)
        self.assertEqual(set(report.records[0].aliases), {original, copy})
        (copy / 'result_all.csv').write_text('changed\n')
        report = scan_experiments([self.root])
        self.assertIn('run_id_content_conflict', report.records[0].issues)
        self.assertFalse(report.records[0].can_resume)

    def test_scan_is_bounded_skips_symlink_and_reports_bad_json(self):
        from acprof.tui.experiment_catalog import scan_experiments
        path = self.record('known')
        outside = self.root / 'outside'
        outside.mkdir()
        (path.parent / 'escape').symlink_to(outside, target_is_directory=True)
        (outside / 'static_meta.json').write_text('{}')
        (path / 'runtime_failures.json').write_text('{broken')
        report = scan_experiments([path.parent])
        self.assertEqual(len(report.records), 1)
        self.assertIn('runtime_failures.json', ' '.join(report.warnings))
        limited = scan_experiments([self.root], max_directories=1)
        self.assertTrue(any('scan_limit' in warning for warning in limited.warnings))

    def test_resume_restores_nonform_options_and_reuse_gets_new_output(self):
        from acprof.tui.experiment_catalog import (
            config_from_record,
            resume_command,
            scan_experiments,
        )
        path = self.record('batch')
        record = scan_experiments([path]).records[0]
        command = resume_command(record, python_executable=Path('/fixed/python'))
        parsed = build_parser().parse_args(command[command.index('--model'):])
        self.assertTrue(parsed.resume)
        self.assertEqual(parsed.matrix_seed, 71)
        self.assertEqual(parsed.matrix_order, 'declared')
        self.assertEqual(parsed.latency_slo, ['default=3'])
        self.assertFalse(parsed.prune_startup_oom)
        self.assertEqual(parsed.revision, 'a' * 40)
        self.assertEqual(Path(parsed.output_dir), path.parent)
        first, second = config_from_record(record, reuse=True), config_from_record(record, reuse=True)
        self.assertFalse(first.resume)
        self.assertNotEqual(first.output_dir, second.output_dir)
        self.assertNotEqual(first.output_dir, str(path.parent))
        self.assertEqual(first.model, 'demo/model')
        self.assertEqual(first.revision, 'a' * 40)
        from acprof.experiment import build_run_command
        reused = build_run_command(first, project_dir=Path.cwd())
        parsed_reuse = build_parser().parse_args(reused[reused.index('--model'):])
        self.assertEqual(parsed_reuse.matrix_seed, 71)
        self.assertEqual(parsed_reuse.matrix_order, 'declared')
        self.assertEqual(parsed_reuse.latency_slo, ['default=3'])
        self.assertFalse(parsed_reuse.prune_startup_oom)

    def test_legacy_missing_identity_is_unknown_and_cannot_resume(self):
        from acprof.tui.experiment_catalog import resume_command, scan_experiments
        path = self.root / 'legacy'
        path.mkdir()
        (path / 'static_meta.json').write_text(json.dumps({'model_name': 'old/model'}))
        record = scan_experiments([path]).records[0]
        self.assertIsNone(record.run_id)
        with self.assertRaisesRegex(ValueError, 'run_id'):
            resume_command(record, python_executable=Path('/fixed/python'))

    def test_directory_enumeration_and_duplicate_hashing_have_budgets(self):
        from acprof.tui.experiment_catalog import scan_experiments
        original = self.record('a')
        shutil.copytree(original, self.root / 'b' / original.name)
        report = scan_experiments([self.root], max_duplicate_bytes=3)
        self.assertEqual(report.records[0].issues, ('duplicate_evidence_unverified',))
        self.assertFalse(report.records[0].can_resume)
        for index in range(30):
            (self.root / f'file-{index}').touch()
        report = scan_experiments([self.root], max_entries=5)
        self.assertIn('scan_entry_limit', ' '.join(report.warnings))
        calls = []
        def cancelled():
            calls.append(True)
            return len(calls) > 4
        with self.assertRaises(InterruptedError):
            scan_experiments([self.root], cancelled=cancelled)
