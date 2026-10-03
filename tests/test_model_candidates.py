"""Model suggestions keep cache presence separate from conditional success."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import test_experiment_catalog as fixtures


class ModelCandidateTests(unittest.TestCase):
    def test_coverage_candidate_retains_attempt_resources_and_failed_access_evidence(self):
        from acprof.tui.model_candidates import coverage_candidates
        root = self.fixture.root
        report = root / 'coverage.json'
        report.write_text(json.dumps({'schema_version': 2, 'scope': 'selected_sample_only; no_formal_measurement',
            'attempts': [{'attempt_id': 2, 'configuration': {'resources': {'cpus': 2, 'memory_gb': 8, 'gpu': False},
                'budgets': {'max_download_bytes': 5000}, 'host': {'machine_id_sha256': 'host-a'}}}],
            'rows': [{'model_id': 'demo/gated', 'revision': 'b' * 40, 'attempt_id': 2,
                'failure': {'reason_code': 'access_denied', 'detail': 'accept model terms'}}]}))
        candidates = coverage_candidates(report)
        self.assertEqual(candidates[0].status, 'revalidate')
        self.assertEqual(candidates[0].history_status, 'access_required')
        self.assertIn('access_denied', candidates[0].reason)
        self.assertIn('current conditions', candidates[0].reason)
        from acprof.tui.model_candidates import model_choice
        self.assertIn('use', model_choice(candidates[0]).actions)
        self.assertEqual(candidates[0].conditions['mems'], '8')
        self.assertEqual(candidates[0].report, report)
        self.assertIn('max_download_bytes', candidates[0].detail)
        self.assertIn('accept model terms', candidates[0].detail)
        payload = json.loads(report.read_text())
        payload['rows'][0]['failure'] = {'reason_code': 'resource_limit', 'detail': 'original 8 GiB limit'}
        report.write_text(json.dumps(payload))
        limited = coverage_candidates(report)[0]
        self.assertEqual(limited.status, 'revalidate')
        self.assertEqual(limited.history_status, 'resource_limited')
        self.assertEqual(limited.conditions['mems'], '8')
        self.assertIn('use', model_choice(limited).actions)

    def setUp(self):
        self.fixture = fixtures.ExperimentCatalogTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def recorded(self, *, code=None):
        from acprof.tui.experiment_catalog import scan_experiments
        path = self.fixture.record('history')
        state = json.loads((path / 'run_state.json').read_text())
        from acprof.host.execution_conditions import measurement_environment
        state['options']['measurement_environment'] = measurement_environment()
        state['host'] = {'machine_id_sha256': 'host-a', 'source_sha256': 'source-a'}
        (path / 'run_state.json').write_text(json.dumps(state))
        (path / 'runtime_validation.json').write_text(json.dumps({'status': 'ok' if code is None else 'error',
            'scope': 'isolated_minimum_scale_predict_and_postprocess_before_measurement'}))
        if code:
            (path / 'runtime_failures.json').write_text(json.dumps({'failures': [{
                'stage': 'runtime', 'reason_code': code, 'detail': 'original failure', 'evidence': {'memory_gb': 4}}]}))
        return scan_experiments([path]).records[0]

    def test_success_is_bound_to_revision_runtime_device_and_resources(self):
        from acprof.tui.model_candidates import candidate_from_record, record_conditions
        record = self.recorded()
        conditions = record_conditions(record)
        candidate = candidate_from_record(record, conditions)
        self.assertEqual(candidate.status, 'verified_current')
        for key, changed in (('revision', 'b' * 40), ('runtime_environment', 'env-b'), ('device', 'GPU-other'), ('mems', '8'), ('host_id', 'host-b'),
                             ('execution_options', {'cpuset_cpus': '0-1'}), ('execution_options', {'input_scales': '128'})):
            with self.subTest(key=key):
                current = {**conditions, key: changed}
                candidate = candidate_from_record(record, current)
                self.assertEqual(candidate.status, 'revalidate')
                self.assertIn(key, candidate.reason)
        candidate = candidate_from_record(record, {**conditions, 'revision': ''})
        self.assertEqual(candidate.status, 'revalidate')
        self.assertIn('revision', candidate.reason)

    def test_failures_remain_selectable_with_original_resources_and_source(self):
        from acprof.tui.model_candidates import candidate_from_record, record_conditions
        record = self.recorded(code='resource_limit')
        candidate = candidate_from_record(record, record_conditions(record))
        self.assertEqual(candidate.status, 'resource_limited')
        self.assertEqual(candidate.experiment, record.directory)
        changed = candidate_from_record(record, {**record_conditions(record), 'mems': '8'})
        self.assertEqual(changed.status, 'revalidate')
        self.assertEqual(changed.history_status, 'resource_limited')
        self.assertIn('original failure', candidate.detail)
        self.assertIn('memory_gb', candidate.detail)
        access = replace(record, failures=({'reason_code': 'access_denied', 'detail': 'gated'},))
        self.assertEqual(candidate_from_record(access, record_conditions(access)).status, 'access_required')

    def test_cache_is_not_inference_evidence_and_missing_blobs_are_not_cached(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from acprof.tui.model_candidates import cached_candidates
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = root / 'entries' / ('a' * 64)
            entry.mkdir(parents=True)
            (entry / 'model_download_plan.json').write_text('{}')
            plan = {'model_id': 'demo/cached', 'model_revision': 'a' * 40}
            with patch('acprof.host.model_store.read_entry', return_value=plan), patch(
                'acprof.host.model_store.model_sources', return_value=[SimpleNamespace(cache_status='hit')]
            ):
                values, warnings = cached_candidates(root)
                self.assertEqual(len(values), 1)
                self.assertEqual(values[0].status, 'cached_unverified')
                self.assertFalse(warnings)
            with patch('acprof.host.model_store.read_entry', return_value=plan), patch(
                'acprof.host.model_store.model_sources', return_value=[SimpleNamespace(cache_status='partial')]
            ):
                values, warnings = cached_candidates(root)
                self.assertFalse(values)

    def test_actual_runtime_routing_and_driver_query_are_shared_and_bounded(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from acprof.experiment import RunConfig
        from acprof.host.runtime_images import select_nlp_torch_index_url
        from acprof.tui.model_candidates import current_conditions, current_context
        record = self.recorded()
        task = {'model_id': record.model_id, 'pipeline_tag': 'fill-mask', 'task_family': 'nlp',
                'runtime_backend': 'transformers_model', 'library_name': 'transformers',
                'model_revision': record.revision, 'detection_method': 'config_infer',
                'model_config': {'model_type': 'bert'}, 'model_resolution': {'runtime_profile': 'historical', 'requested_devices': ['cpu']}}
        record = replace(record, state={**record.state, 'runtime': {'task': task}})
        config = RunConfig.smoke(record.model_id)
        with patch.dict('os.environ', {}, clear=True), patch('acprof.host.runtime_images.shutil.which', return_value='/usr/bin/nvidia-smi'), patch(
                'acprof.host.command.run_command', return_value=SimpleNamespace(returncode=0, stdout='CUDA Version: 12.4')) as command:
            self.assertTrue(select_nlp_torch_index_url().endswith('/cu124'))
            self.assertEqual(command.call_args.kwargs['timeout'], 10)
            command.reset_mock()
            with patch('acprof.host.run_state.host_identity', return_value={'machine_id_sha256': 'host-a', 'source_sha256': 'source-a'}), \
                 patch('acprof.host.gpu_device.resolve_gpu_device') as device, \
                 patch('acprof.host.task_support.require_task_support'):
                context = current_context(config, Path.cwd())
                frozen = json.dumps(record.state, sort_keys=True)
                first = current_conditions(record, config, context, Path.cwd())
                second = current_conditions(record, config, context, Path.cwd())
                self.assertEqual(first['runtime_profile'], 'nlp-cu124')
                self.assertEqual(first['runtime_environment'], second['runtime_environment'])
                self.assertEqual(json.dumps(record.state, sort_keys=True), frozen)
                command.assert_called_once()
                device.assert_not_called()

    def test_missing_thread_evidence_and_changed_thread_or_input_options_require_revalidation(self):
        from acprof.tui.model_candidates import candidate_from_record, record_conditions
        record = self.recorded()
        conditions = record_conditions(record)
        for changes in ({'input_scales': '128'}, {'input_scale_policy': 'minimal'}, {'cpuset_cpus': '0-1'},
                        {'measurement_environment': {'OMP_NUM_THREADS': '3'}}):
            current = {**conditions, 'execution_options': {**conditions['execution_options'], **changes}}
            self.assertEqual(candidate_from_record(record, current).status, 'revalidate')
        legacy = replace(record, options={**record.options, 'measurement_environment': {}})
        self.assertEqual(candidate_from_record(legacy, record_conditions(legacy)).status, 'revalidate')

    def test_driver_timeout_stays_unknown_and_non_torch_cpu_avoids_probe(self):
        from subprocess import TimeoutExpired
        from unittest.mock import patch

        from acprof.experiment import RunConfig
        from acprof.tui.model_candidates import current_context
        config = RunConfig.smoke('demo/model')
        with patch('acprof.host.run_state.host_identity', return_value={'machine_id_sha256': 'host-a'}), \
             patch('acprof.host.runtime_images.select_nlp_torch_index_url', side_effect=TimeoutExpired('nvidia-smi', 10)) as select, \
             patch('acprof.host.gpu_device.resolve_gpu_device') as gpu:
            timed_out = current_context(config, Path.cwd())
            self.assertIsNone(timed_out['torch_index_url'])
            self.assertIn('10', timed_out['warnings'][0])
            select.assert_called_once()
            select.reset_mock()
            current_context(config, Path.cwd(), needs_torch_platform=False)
            select.assert_not_called()
            gpu.assert_not_called()
