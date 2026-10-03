"""Only positive evidence of a dead owner permits crash recovery."""
import json
import subprocess
from unittest.mock import patch

import pytest

from acprof.host.container_lifecycle import (
    ContainerCleanupError,
    container_owner_labels,
    recover_abandoned_containers,
)


class TestContainerRecovery:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.owner = container_owner_labels()
        self.other_owner = {**self.owner, 'org.acprof.owner.pid': str(int(self.owner['org.acprof.owner.pid']) + 100000)}
        self.identifier = 'a' * 64
        self.removed = []
        self.commands = []

    def recover(self, labels, *, inspect_code=0, inspect_error='', remove_code=0, state=None):
        def execute(command, **_kwargs):
            self.commands.append(command)
            if command[:2] == ['docker', 'ps']:
                return subprocess.CompletedProcess(command, 0, self.identifier, '')
            if command[:2] == ['docker', 'inspect']:
                data = [{'Id': self.identifier, 'Config': {'Labels': labels},
                         'State': {'Running': True, 'Status': 'running'} if state is None else state}]
                return subprocess.CompletedProcess(command, inspect_code, json.dumps(data), inspect_error)
            self.removed.append(command[-1])
            return subprocess.CompletedProcess(command, remove_code, '', 'permission denied' if remove_code else '')
        return recover_abandoned_containers(self.owner, execute)

    @pytest.mark.parametrize('running', (True, False))
    def test_current_owner_residue_blocks_new_start_without_removing_it(self, running):
        state = {'Running': running, 'Status': 'running' if running else 'exited'}
        with pytest.raises(ContainerCleanupError) as raised:
            self.recover(self.owner, state=state)
        evidence = raised.value.to_dict()
        assert (evidence['container_id']) == (self.identifier)
        assert (evidence['final_state']) == ('present')
        assert (evidence['docker_state']) == (state)
        assert (evidence['operations'][0]['operation']) == ('preflight')
        assert not (self.removed)
        assert (all(command[1] in ('ps', 'inspect') for command in self.commands))

    def test_other_live_owner_is_preserved(self):
        with patch('acprof.host.container_lifecycle._process_identity',
                   return_value=(self.other_owner['org.acprof.owner.start'], 'S')):
            assert (self.recover(self.other_owner)) == (())
        assert not (self.removed)

    def test_reused_pid_is_recovered_when_start_time_differs(self):
        previous = {**self.owner, 'org.acprof.owner.start': str(int(self.owner['org.acprof.owner.start']) + 1)}
        assert (self.recover(previous)) == ((self.identifier,))

    def test_missing_owner_and_previous_boot_are_recovered(self):
        with patch('acprof.host.container_lifecycle._process_identity', side_effect=FileNotFoundError):
            assert (self.recover(self.other_owner)) == ((self.identifier,))
        labels = {**self.owner, 'org.acprof.owner.boot': '00000000-0000-0000-0000-000000000000'}
        assert (self.recover(labels)) == ((self.identifier,))

    @pytest.mark.parametrize('error_case', range(3), ids=['PermissionError', 'ValueError', 'IndexError'])
    def test_unreadable_or_malformed_procfs_does_not_prove_owner_exited(self, error_case):
        error = tuple((PermissionError, ValueError, IndexError))[error_case]
        with patch('acprof.host.container_lifecycle._process_identity', side_effect=error):
            assert (self.recover(self.other_owner)) == (())
        assert not (self.removed)

    @pytest.mark.parametrize('labels_case', range(4), ids=['{}', "{**owner, 'org.acprof.container.lifecycle': '2'}", "{**owner, 'org.acprof.owner.pid': '-1'}", "{**owner, 'org.acprof.owner.start': 'unknown'}"])
    def test_unknown_lifecycle_and_incomplete_labels_are_preserved(self, labels_case):
        labels = tuple(({}, {**self.owner, 'org.acprof.container.lifecycle': '2'}, {**self.owner, 'org.acprof.owner.pid': '-1'}, {**self.owner, 'org.acprof.owner.start': 'unknown'}))[labels_case]
        assert (self.recover(labels)) == (())
        assert not (self.removed)

    def test_concurrent_removal_is_safe_but_inspection_failure_is_visible(self):
        assert (self.recover(self.owner, inspect_code=1, inspect_error='No such object: ' + self.identifier)) == (())
        with pytest.raises(RuntimeError, match='cleanup unknown'):
            self.recover(self.owner, inspect_code=1, inspect_error='permission denied')
        assert not (self.removed)

    def test_cleanup_failure_aborts_before_starting_another_container(self):
        with patch('acprof.host.container_lifecycle._process_identity', side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match='Cannot remove'):
                self.recover(self.other_owner, remove_code=1)


class TestContainerCleanup:
    def stop(self, replies):
        from acprof.host.docker_runtime import RunningContainer, stop_container_session
        self.calls = []
        def execute(command, **kwargs):
            self.calls.append((command, kwargs))
            reply = replies[command[1]]
            if isinstance(reply, BaseException):
                raise reply
            code, output, error = reply
            return subprocess.CompletedProcess(command, code, output, error)
        session = RunningContainer('owned', 'http://localhost', 8002, 0, container_id='a' * 64)
        with patch('acprof.host.command.run_command', side_effect=execute):
            stop_container_session(session)

    def test_failed_stop_then_successful_remove_is_complete_and_bounded(self):
        self.stop({'stop': (1, '', 'cannot stop'), 'rm': (0, 'a' * 64, '')})
        assert (all(0 < kwargs.get('timeout', 0) <= 30 for _, kwargs in self.calls))

    def test_failed_remove_requires_inspect_confirmation_of_absence(self):
        self.stop({'stop': (1, '', 'No such container'), 'rm': (1, '', 'transport interrupted'),
                   'inspect': (1, '[]', 'Error: No such object: ' + 'a' * 64)})
        assert ([c[0][1] for c in self.calls]) == (['stop', 'rm', 'inspect'])

    def test_surviving_container_aborts_with_structured_evidence(self):
        state = json.dumps([{'Id': 'a' * 64, 'State': {'Running': True, 'Status': 'running'}}])
        with pytest.raises(RuntimeError) as raised:
            self.stop({'stop': (1, '', 'cannot stop'), 'rm': (1, '', 'cannot remove'),
                       'inspect': (0, state, '')})
        evidence = raised.value.to_dict()
        assert (evidence['container_id']) == ('a' * 64)
        assert (evidence['final_state']) == ('present')
        assert (evidence['docker_state']['Running'])
        assert (len(evidence['operations'])) == (3)

    def test_docker_timeouts_remain_unknown_and_preserve_original_error(self):
        original = ValueError('inference failed first')
        try:
            raise original
        except ValueError:
            with pytest.raises(RuntimeError) as raised:
                self.stop({name: subprocess.TimeoutExpired(['docker', name], 1)
                           for name in ('stop', 'rm', 'inspect')})
        assert (raised.value.to_dict()['final_state']) == ('unknown')
        assert (raised.value.to_dict()['run_error']['detail']) == (str(original))
        assert ([c[0][1] for c in self.calls]) == (['stop', 'rm', 'inspect'])

    def test_no_such_in_remove_error_does_not_prove_absence(self):
        with pytest.raises(RuntimeError):
            self.stop({'stop': (1, '', 'stop failed'), 'rm': (1, '', 'No such backend'),
                       'inspect': (1, '', 'Docker daemon unavailable')})


    def test_matrix_stops_after_cleanup_failure_and_saves_both_errors(self):
        import tempfile
        from pathlib import Path

        from acprof.host import orchestrator
        from acprof.host.detect import TaskInfo
        from acprof.host.docker_runtime import RunningContainer
        from acprof.host.runtime_images import ImageInfo
        session = RunningContainer('owned', 'http://localhost', 8002, 0, container_id='a' * 64)
        task = TaskInfo('fixture/model', 'tabular-classification', 'structured', 'onnxruntime', 'onnx', 'a' * 40, 'manual')
        with tempfile.TemporaryDirectory() as directory, patch.object(
            orchestrator, 'start_container_session', return_value=session,
        ) as start, patch.object(orchestrator, 'record_case_conditions', side_effect=ValueError('original runtime failure')), patch(
            'acprof.host.command.run_command', side_effect=subprocess.TimeoutExpired(['docker'], 1),
        ):
            with pytest.raises(RuntimeError):
                orchestrator.run_matrix(task, ImageInfo('fixture'), [1, 2], [2], ['off'], directory,
                                        directory, profiling_mode='basic', input_scales='1', prune_startup_oom=False)
            assert (start.call_count) == (1)
            paths = list(Path(directory).rglob('*cleanup_error.json'))
            assert (len(paths)) == (1)
            evidence = json.loads(paths[0].read_text())
            assert (evidence['run_error']['detail']) == ('original runtime failure')
            assert (evidence['final_state']) == ('unknown')
