import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acprof.host import (
    container_state,
    docker_runtime as docker,
    orchestrator,
    runtime_images,
    startup_probe,
)
from acprof.host.detect import TaskInfo
from acprof.host.matrix_plan import matrix_identity


class StartupProbeTests(unittest.TestCase):
    def setUp(self):
        self.task = TaskInfo('org/model', 'fill-mask', 'nlp', 'transformers_pipeline',
                             'transformers', 'a' * 40, 'manual')
        self.session = docker.RunningContainer('probe-owned', 'http://localhost', 1234, 0.1, container_id='b' * 64)
        self.image = runtime_images.ImageInfo('sha256:' + 'b' * 64)
        self.identity = matrix_identity(self.task, self.image, [2, 1], [8, 2, 4], ['off'],
                                        [64.], order='declared', seed=0, prune=True)

    def test_probe_only_starts_waits_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch("acprof.host.docker_runtime.start_container_session", return_value=self.session), \
             patch("acprof.host.container_state.inspect_container_state", return_value={'Running': True}), \
             patch.object(docker, 'stop_container_session') as stop, \
             patch.object(orchestrator, 'run_single_case', side_effect=AssertionError('formal collection')):
            report = startup_probe.run_startup_probes(tmp, self.identity, self.task, self.image,
                                                      request_timeout_seconds=45)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ['startup_oom_pruning.json'])
            self.assertEqual(report['attempts'][0]['outcome'], 'startup_feasible')
            self.assertEqual(len(report['attempts']), 1)
            stop.assert_called_once()

    def test_only_contiguous_explicit_docker_oom_can_prune(self):
        oom = container_state.ContainerStartupError('OOM', state={'OOMKilled': True, 'Running': False})
        failures = [RuntimeError('container_oom_killed during startup'),
                    RuntimeError('CUDA out of memory'), RuntimeError('runtime OOM'),
                    container_state.ContainerStartupError('timeout', state={'OOMKilled': False}, timed_out=True),
                    container_state.ContainerStartupError('timeout with ambiguous late OOM',
                        state={'OOMKilled': True, 'Running': False}, timed_out=True),
                    container_state.ContainerStartupError('restarting',
                        state={'OOMKilled': True, 'Running': False, 'Restarting': True}),
                    container_state.ContainerStartupError('still running',
                        state={'OOMKilled': True, 'Running': True}),
                    container_state.ContainerStartupError('exit 137', state={'OOMKilled': False, 'ExitCode': 137}),
                    container_state.ContainerStartupError('unknown state')]
        for failure in failures:
            with self.subTest(error=str(failure)), tempfile.TemporaryDirectory() as tmp, \
                 patch("acprof.host.docker_runtime.start_container_session", side_effect=[oom, failure, oom]) as start, \
                 patch.object(docker, 'stop_container_session'):
                report = startup_probe.run_startup_probes(tmp, self.identity, self.task, self.image,
                                                          request_timeout_seconds=45)
                self.assertEqual(startup_probe.startup_oom_prefixes(report), {'off': [2]})
                self.assertEqual(start.call_count, 2)
                self.assertTrue(all(r['cpu_cores'] == 1 for r in report['attempts']))

    def test_probe_precedes_frozen_plan_and_formal_rows(self):
        def start(*args, **kwargs):
            self.assertFalse((Path(directory) / 'matrix_plan.json').exists())
            self.assertEqual(list(Path(directory).glob('result*.csv')), [])
            if kwargs['mem'] == 2:
                raise container_state.ContainerStartupError('OOM', state={'OOMKilled': True, 'Running': False})
            return self.session

        def formal(**kwargs):
            self.assertTrue((Path(directory) / 'matrix_plan.json').is_file())
            self.assertEqual(kwargs['mem'], 4)
            return ''

        with tempfile.TemporaryDirectory() as directory, \
             patch("acprof.host.docker_runtime.start_container_session", side_effect=start), \
             patch("acprof.host.container_state.inspect_container_state", return_value={'Running': True}), \
             patch.object(docker, 'stop_container_session'), \
             patch.object(orchestrator, 'run_single_case', side_effect=formal) as case:
            result = orchestrator.run_matrix(self.task, self.image, [2, 1], [4, 2], ['off'],
                directory, directory, warmup=0, repeat=1, input_scales='64',
                prune_startup_oom=True, matrix_order='declared')
            self.assertEqual(case.call_count, 2)
            self.assertEqual(len(result), 2)
            for path in result:
                with open(path) as stream:
                    row = next(csv.DictReader(stream))
                self.assertIn('result_origin=inferred_not_measured', row['error'])
                self.assertEqual(row['result_origin'], 'inferred_not_measured')
            report = json.loads((Path(directory) / 'startup_oom_pruning.json').read_text())
            self.assertEqual(report['status'], 'complete')
            with patch("acprof.host.docker_runtime.start_container_session", side_effect=AssertionError('reprobe')):
                orchestrator.run_matrix(self.task, self.image, [2, 1], [4, 2], ['off'],
                    directory, directory, warmup=0, repeat=1, input_scales='64',
                    prune_startup_oom=True, matrix_order='declared')

    def test_interrupted_probe_resumes_completed_attempts(self):
        oom = container_state.ContainerStartupError('OOM', state={'OOMKilled': True, 'Running': False})
        with tempfile.TemporaryDirectory() as tmp, patch.object(docker, 'stop_container_session'), \
             patch("acprof.host.container_state.inspect_container_state", return_value={'Running': True}):
            with patch("acprof.host.docker_runtime.start_container_session", side_effect=[oom, KeyboardInterrupt()]):
                with self.assertRaises(KeyboardInterrupt):
                    startup_probe.run_startup_probes(tmp, self.identity, self.task, self.image,
                                                     request_timeout_seconds=45)
            with patch("acprof.host.docker_runtime.start_container_session", return_value=self.session) as start:
                report = startup_probe.run_startup_probes(tmp, self.identity, self.task, self.image,
                                                          request_timeout_seconds=45)
                self.assertEqual(start.call_args.kwargs['mem'], 4)
                start.assert_called_once()
                self.assertEqual(startup_probe.startup_oom_prefixes(report), {'off': [2]})
