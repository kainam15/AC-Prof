"""A retry must prove previous owned-container cleanup complete before new work."""
import json
import subprocess
from unittest.mock import patch

import pytest

from acprof.host import container_lifecycle as lifecycle


class TestCleanupDebt:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.owner = lifecycle.container_owner_labels()
        self.identifier = 'a' * 64
        self.other_owner = {**self.owner, 'org.acprof.owner.pid': '123456789'}
        self.commands = []
        self.present = {}
        self.listed = []

    def execute_docker(self, command, **kwargs):
        self.commands.append((command, kwargs))
        if command[1] == 'ps':
            return subprocess.CompletedProcess(command, 0, '\n'.join(self.listed), '')
        if command[1] == 'inspect':
            item = self.present.get(command[-1])
            return subprocess.CompletedProcess(command, 0 if item else 1,
                json.dumps([item]) if item else '[]', '' if item else 'Error: No such object: ' + command[-1])
        if command[1] == 'rm':
            self.present.pop(command[-1], None)
            self.listed.remove(command[-1])
            return subprocess.CompletedProcess(command, 0, command[-1], '')
        pytest.fail('Unexpected Docker operation: ' + repr(command))

    def recover(self, errors, run=None):
        recover = getattr(lifecycle, 'recover_cleanup_debt', None)
        assert (callable(recover)), 'cleanup retries need a public debt verification boundary'
        return recover(errors, run or self.execute_docker)

    def debt(self, identifier=None):
        return {'schema_version': 1, 'status': 'incomplete',
                'container_id': self.identifier if identifier is None else identifier, 'final_state': 'unknown'}

    def add_container(self, *, labels=None):
        self.listed.append(self.identifier)
        self.present[self.identifier] = {'Id': self.identifier, 'Config': {'Labels': labels or self.other_owner},
                                        'State': {'Running': True, 'Status': 'running'}}

    def test_absent_ids_are_verified_once_and_evidence_is_bounded(self):
        evidence = self.recover([self.debt(), self.debt()])
        assert (evidence['status']) == ('complete')
        assert (evidence['verified_container_ids']) == ([self.identifier])
        assert (evidence['recovered_container_ids']) == ([])
        assert (evidence['scoped_inventory_empty']) is None
        assert ([command[1] for command, _ in self.commands]) == (['ps', 'inspect'])
        assert (all(0 < kwargs.get('timeout', 0) <= 30 for _, kwargs in self.commands))

    def test_another_live_owner_does_not_clear_debt_or_get_removed(self):
        self.add_container()
        with patch.object(lifecycle, '_process_identity', return_value=(self.other_owner['org.acprof.owner.start'], 'S')):
            with pytest.raises(lifecycle.ContainerCleanupError) as raised:
                self.recover([self.debt()])
        assert (raised.value.to_dict()['final_state']) == ('present')
        assert (raised.value.docker_state['Running'])
        assert (self.identifier) in (self.present)
        assert (all(command[1] in {'ps', 'inspect'} for command, _ in self.commands))

    def test_abandoned_owner_is_recovered_then_absence_is_verified(self):
        self.add_container(labels={**self.other_owner, 'org.acprof.owner.boot': 'previous-boot'})
        evidence = self.recover([self.debt()])
        assert (evidence['recovered_container_ids']) == ([self.identifier])
        assert (evidence['verified_container_ids']) == ([self.identifier])
        assert ([command[1] for command, _ in self.commands]) == (['ps', 'inspect', 'rm', 'inspect'])
        assert not (self.present)

    @pytest.mark.parametrize('errors_case', range(5), ids=['[]', '[{}]', "[debt('')]", "[debt('mutable-name')]", '[None]'])
    def test_missing_or_invalid_ids_require_fresh_empty_scoped_inventory(self, errors_case):
        errors = tuple(([], [{}], [self.debt('')], [self.debt('mutable-name')], [None]))[errors_case]
        self.commands = []
        evidence = self.recover(errors)
        assert (evidence['scoped_inventory_empty'])
        assert (evidence['verified_container_ids']) == ([])
        queries = [command for command, _ in self.commands]
        assert (len(queries)) == (2)
        for command in queries:
            assert (command[:4]) == (['docker', 'ps', '-aq', '--no-trunc'])
            for key in (lifecycle.LIFECYCLE_LABEL, lifecycle.OWNER_PREFIX + 'host', lifecycle.OWNER_PREFIX + 'uid'):
                assert ('label=' + key + '=' + self.owner[key]) in (command)
            assert ('label=' + lifecycle.OWNER_PREFIX + 'pid=' + self.owner[lifecycle.OWNER_PREFIX + 'pid']) not in (command)

    def test_unknown_id_cannot_be_cleared_while_live_scoped_container_remains(self):
        self.add_container()
        with patch.object(lifecycle, '_process_identity', return_value=(self.other_owner['org.acprof.owner.start'], 'S')):
            with pytest.raises(lifecycle.ContainerCleanupError) as raised:
                self.recover([self.debt('')])
        assert (raised.value.final_state) == ('unknown')
        assert (raised.value.operations[-1]['container_ids']) == ([self.identifier])
        assert (all(command[1] in {'ps', 'inspect'} for command, _ in self.commands))

    @pytest.mark.parametrize('reply_case', range(6))
    def test_inspection_errors_never_prove_absence(self, reply_case):
        replies = [
            subprocess.TimeoutExpired(['docker', 'inspect'], 15),
            PermissionError('Docker unavailable'),
            (1, '', 'No such object: ' + 'b' * 64),
            (1, '', 'No such object: ' + self.identifier + '0'),
            (0, 'not JSON', ''),
            (0, json.dumps([{'Id': 'b' * 64, 'State': {'Running': True}}]), ''),
        ]
        reply = tuple(replies)[reply_case]
        def execute(command, **kwargs):
            if command[1] != 'inspect':
                return self.execute_docker(command, **kwargs)
            if isinstance(reply, BaseException):
                raise reply
            return subprocess.CompletedProcess(command, *reply)
        with pytest.raises(lifecycle.ContainerCleanupError) as raised:
            self.recover([self.debt()], execute)
        assert (raised.value.final_state) == ('unknown')
        assert (raised.value.container_id) == (self.identifier)

    @pytest.mark.parametrize('reply_case', range(3), ids=["subprocess.TimeoutExpired(['docker', 'ps'], 15)", "(1, '', 'permission denied')", "(0, 'shortid', '')"])
    def test_failed_final_inventory_cannot_clear_unknown_debt(self, reply_case):
        reply = tuple((subprocess.TimeoutExpired(['docker', 'ps'], 15), (1, '', 'permission denied'), (0, 'shortid', '')))[reply_case]
        calls = []
        def execute(command, **kwargs):
            calls.append(command)
            if len(calls) == 1:
                return self.execute_docker(command, **kwargs)
            if isinstance(reply, BaseException):
                raise reply
            return subprocess.CompletedProcess(command, *reply)
        with pytest.raises(lifecycle.ContainerCleanupError) as raised:
            self.recover([self.debt('')], execute)
        assert (raised.value.final_state) == ('unknown')

    def test_successful_remove_without_absence_does_not_clear_debt(self):
        self.add_container(labels={**self.other_owner, 'org.acprof.owner.boot': 'previous-boot'})
        def execute(command, **kwargs):
            if command[1] == 'rm':
                return subprocess.CompletedProcess(command, 0, self.identifier, '')
            return self.execute_docker(command, **kwargs)
        with pytest.raises(lifecycle.ContainerCleanupError) as raised:
            self.recover([self.debt()], execute)
        assert (raised.value.final_state) == ('present')
