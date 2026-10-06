import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.host import (
    container_state,
    docker_runtime as docker,
    orchestrator,
    runtime_images,
    startup_probe,
)
from acprof.host.detect import TaskInfo
from acprof.host.matrix_plan import freeze_matrix_plan, matrix_identity


class TestStartupProbe:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
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
            assert ([p.name for p in Path(tmp).iterdir()]) == (['startup_oom_pruning.json'])
            assert (report['attempts'][0]['outcome']) == ('startup_feasible')
            assert (len(report['attempts'])) == (1)
            stop.assert_called_once()

    @pytest.mark.parametrize('failure_case', range(9))
    def test_only_contiguous_explicit_docker_oom_can_prune(self, failure_case):
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
        failure = tuple(failures)[failure_case]
        with tempfile.TemporaryDirectory() as tmp, patch("acprof.host.docker_runtime.start_container_session", side_effect=[oom, failure, oom]) as start, patch.object(docker, 'stop_container_session'):
            report = startup_probe.run_startup_probes(tmp, self.identity, self.task, self.image,
                                                      request_timeout_seconds=45)
            assert (startup_probe.startup_oom_prefixes(report)) == ({'off': [2]})
            assert (start.call_count) == (2)
            assert (all(r['cpu_cores'] == 1 for r in report['attempts']))

    def test_probe_precedes_frozen_plan_and_formal_rows(self):
        def start(*args, **kwargs):
            assert not ((Path(directory) / 'matrix_plan.json').exists())
            assert (list(Path(directory).glob('result*.csv'))) == ([])
            if kwargs['mem'] == 2:
                raise container_state.ContainerStartupError('OOM', state={'OOMKilled': True, 'Running': False})
            return self.session

        def formal(**kwargs):
            assert ((Path(directory) / 'matrix_plan.json').is_file())
            assert (kwargs['mem']) == (4)
            return ''

        with tempfile.TemporaryDirectory() as directory, \
             patch("acprof.host.docker_runtime.start_container_session", side_effect=start), \
             patch("acprof.host.container_state.inspect_container_state", return_value={'Running': True}), \
             patch.object(docker, 'stop_container_session'), \
             patch.object(orchestrator, 'run_single_case', side_effect=formal) as case:
            result = orchestrator.run_matrix(self.task, self.image, [2, 1], [4, 2], ['off'],
                directory, directory, warmup=0, repeat=1, input_scales='64',
                prune_startup_oom=True, matrix_order='declared')
            assert (case.call_count) == (2)
            assert (len(result)) == (2)
            for path in result:
                with open(path) as stream:
                    row = next(csv.DictReader(stream))
                assert ('result_origin=inferred_not_measured') in (row['error'])
                assert (row['result_origin']) == ('inferred_not_measured')
            report = json.loads((Path(directory) / 'startup_oom_pruning.json').read_text())
            assert (report['status']) == ('complete')
            with patch("acprof.host.docker_runtime.start_container_session", side_effect=AssertionError('reprobe')):
                orchestrator.run_matrix(self.task, self.image, [2, 1], [4, 2], ['off'],
                    directory, directory, warmup=0, repeat=1, input_scales='64',
                    prune_startup_oom=True, matrix_order='declared')

    def test_frozen_plan_rejects_nonfinite_probe_evidence_before_formal_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            freeze_matrix_plan(root / "matrix_plan.json", self.identity, {"off": [2]})
            (root / startup_probe.PROBE_NAME).write_text(json.dumps({
                "schema_version": 2,
                "identity": self.identity,
                "status": "complete",
                "attempts": [{
                    "cpu_cores": 1,
                    "mem_cap_gb": 2,
                    "gpu_mode": "off",
                    "ready": False,
                    "outcome": "startup_oom",
                    "docker_state": {
                        "OOMKilled": True,
                        "Running": False,
                        "Restarting": False,
                    },
                }],
                "corrupt_metric": float("nan"),
            }))
            with patch.object(
                orchestrator,
                "run_single_case",
                side_effect=AssertionError("formal collection must not start"),
            ), pytest.raises(ValueError, match="invalid startup probe report JSON"):
                orchestrator.run_matrix(
                    self.task,
                    self.image,
                    [2, 1],
                    [8, 2, 4],
                    ["off"],
                    tmp,
                    tmp,
                    warmup=0,
                    repeat=1,
                    input_scales="64",
                    prune_startup_oom=True,
                    matrix_order="declared",
                )

    def test_interrupted_probe_resumes_completed_attempts(self):
        oom = container_state.ContainerStartupError('OOM', state={'OOMKilled': True, 'Running': False})
        with tempfile.TemporaryDirectory() as tmp, patch.object(docker, 'stop_container_session'), \
             patch("acprof.host.container_state.inspect_container_state", return_value={'Running': True}):
            with patch("acprof.host.docker_runtime.start_container_session", side_effect=[oom, KeyboardInterrupt()]):
                with pytest.raises(KeyboardInterrupt):
                    startup_probe.run_startup_probes(tmp, self.identity, self.task, self.image,
                                                     request_timeout_seconds=45)
            with patch("acprof.host.docker_runtime.start_container_session", return_value=self.session) as start:
                report = startup_probe.run_startup_probes(tmp, self.identity, self.task, self.image,
                                                          request_timeout_seconds=45)
                assert (start.call_args.kwargs['mem']) == (4)
                start.assert_called_once()
                assert (startup_probe.startup_oom_prefixes(report)) == ({'off': [2]})

    def test_resume_rejects_nonfinite_probe_report_before_starting_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / startup_probe.PROBE_NAME
            path.write_text(json.dumps({
                "schema_version": 2,
                "identity": self.identity,
                "status": "running",
                "attempts": [],
                "corrupt_metric": float("nan"),
            }))
            with patch(
                "acprof.host.docker_runtime.start_container_session",
                side_effect=AssertionError("resume evidence must fail before probing"),
            ), pytest.raises(ValueError, match="invalid startup probe report JSON"):
                startup_probe.run_startup_probes(
                    tmp,
                    self.identity,
                    self.task,
                    self.image,
                    request_timeout_seconds=45,
                )
