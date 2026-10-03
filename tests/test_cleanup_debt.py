"""A retry must prove previous owned-container cleanup complete before new work."""
import json
import subprocess
import unittest
from unittest.mock import patch

from acprof.host import container_lifecycle as lifecycle


class CleanupDebtTests(unittest.TestCase):
    def setUp(self):
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
        self.fail('Unexpected Docker operation: ' + repr(command))

    def recover(self, errors, run=None):
        recover = getattr(lifecycle, 'recover_cleanup_debt', None)
        self.assertTrue(callable(recover), 'cleanup retries need a public debt verification boundary')
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
        self.assertEqual(evidence['status'], 'complete')
        self.assertEqual(evidence['verified_container_ids'], [self.identifier])
        self.assertEqual(evidence['recovered_container_ids'], [])
        self.assertIsNone(evidence['scoped_inventory_empty'])
        self.assertEqual([command[1] for command, _ in self.commands], ['ps', 'inspect'])
        self.assertTrue(all(0 < kwargs.get('timeout', 0) <= 30 for _, kwargs in self.commands))

    def test_another_live_owner_does_not_clear_debt_or_get_removed(self):
        self.add_container()
        with patch.object(lifecycle, '_process_identity', return_value=(self.other_owner['org.acprof.owner.start'], 'S')):
            with self.assertRaises(lifecycle.ContainerCleanupError) as raised:
                self.recover([self.debt()])
        self.assertEqual(raised.exception.to_dict()['final_state'], 'present')
        self.assertTrue(raised.exception.docker_state['Running'])
        self.assertIn(self.identifier, self.present)
        self.assertTrue(all(command[1] in {'ps', 'inspect'} for command, _ in self.commands))

    def test_abandoned_owner_is_recovered_then_absence_is_verified(self):
        self.add_container(labels={**self.other_owner, 'org.acprof.owner.boot': 'previous-boot'})
        evidence = self.recover([self.debt()])
        self.assertEqual(evidence['recovered_container_ids'], [self.identifier])
        self.assertEqual(evidence['verified_container_ids'], [self.identifier])
        self.assertEqual([command[1] for command, _ in self.commands], ['ps', 'inspect', 'rm', 'inspect'])
        self.assertFalse(self.present)

    def test_missing_or_invalid_ids_require_fresh_empty_scoped_inventory(self):
        for errors in ([], [{}], [self.debt('')], [self.debt('mutable-name')], [None]):
            with self.subTest(errors=errors):
                self.commands = []
                evidence = self.recover(errors)
                self.assertTrue(evidence['scoped_inventory_empty'])
                self.assertEqual(evidence['verified_container_ids'], [])
                queries = [command for command, _ in self.commands]
                self.assertEqual(len(queries), 2)
                for command in queries:
                    self.assertEqual(command[:4], ['docker', 'ps', '-aq', '--no-trunc'])
                    for key in (lifecycle.LIFECYCLE_LABEL, lifecycle.OWNER_PREFIX + 'host', lifecycle.OWNER_PREFIX + 'uid'):
                        self.assertIn('label=' + key + '=' + self.owner[key], command)
                    self.assertNotIn('label=' + lifecycle.OWNER_PREFIX + 'pid=' + self.owner[lifecycle.OWNER_PREFIX + 'pid'], command)

    def test_unknown_id_cannot_be_cleared_while_live_scoped_container_remains(self):
        self.add_container()
        with patch.object(lifecycle, '_process_identity', return_value=(self.other_owner['org.acprof.owner.start'], 'S')):
            with self.assertRaises(lifecycle.ContainerCleanupError) as raised:
                self.recover([self.debt('')])
        self.assertEqual(raised.exception.final_state, 'unknown')
        self.assertEqual(raised.exception.operations[-1]['container_ids'], [self.identifier])
        self.assertTrue(all(command[1] in {'ps', 'inspect'} for command, _ in self.commands))

    def test_inspection_errors_never_prove_absence(self):
        replies = [
            subprocess.TimeoutExpired(['docker', 'inspect'], 15),
            PermissionError('Docker unavailable'),
            (1, '', 'No such object: ' + 'b' * 64),
            (1, '', 'No such object: ' + self.identifier + '0'),
            (0, 'not JSON', ''),
            (0, json.dumps([{'Id': 'b' * 64, 'State': {'Running': True}}]), ''),
        ]
        for reply in replies:
            with self.subTest(reply=reply):
                def execute(command, **kwargs):
                    if command[1] != 'inspect':
                        return self.execute_docker(command, **kwargs)
                    if isinstance(reply, BaseException):
                        raise reply
                    return subprocess.CompletedProcess(command, *reply)
                with self.assertRaises(lifecycle.ContainerCleanupError) as raised:
                    self.recover([self.debt()], execute)
                self.assertEqual(raised.exception.final_state, 'unknown')
                self.assertEqual(raised.exception.container_id, self.identifier)

    def test_failed_final_inventory_cannot_clear_unknown_debt(self):
        for reply in (subprocess.TimeoutExpired(['docker', 'ps'], 15), (1, '', 'permission denied'), (0, 'shortid', '')):
            with self.subTest(reply=reply):
                calls = []
                def execute(command, **kwargs):
                    calls.append(command)
                    if len(calls) == 1:
                        return self.execute_docker(command, **kwargs)
                    if isinstance(reply, BaseException):
                        raise reply
                    return subprocess.CompletedProcess(command, *reply)
                with self.assertRaises(lifecycle.ContainerCleanupError) as raised:
                    self.recover([self.debt('')], execute)
                self.assertEqual(raised.exception.final_state, 'unknown')

    def test_successful_remove_without_absence_does_not_clear_debt(self):
        self.add_container(labels={**self.other_owner, 'org.acprof.owner.boot': 'previous-boot'})
        def execute(command, **kwargs):
            if command[1] == 'rm':
                return subprocess.CompletedProcess(command, 0, self.identifier, '')
            return self.execute_docker(command, **kwargs)
        with self.assertRaises(lifecycle.ContainerCleanupError) as raised:
            self.recover([self.debt()], execute)
        self.assertEqual(raised.exception.final_state, 'present')
