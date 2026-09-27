"""Only positive evidence of a dead owner permits crash recovery."""
import json
import subprocess
import unittest
from unittest.mock import patch

from acprof.host.container_lifecycle import container_owner_labels, recover_abandoned_containers


class ContainerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.owner = container_owner_labels()
        self.identifier = 'a' * 64
        self.removed = []

    def recover(self, labels, *, inspect_code=0, inspect_error='', remove_code=0):
        def execute(command, **_kwargs):
            if command[:2] == ['docker', 'ps']:
                return subprocess.CompletedProcess(command, 0, self.identifier, '')
            if command[:2] == ['docker', 'inspect']:
                data = [{'Id': self.identifier, 'Config': {'Labels': labels}}]
                return subprocess.CompletedProcess(command, inspect_code, json.dumps(data), inspect_error)
            self.removed.append(command[-1])
            return subprocess.CompletedProcess(command, remove_code, '', 'permission denied' if remove_code else '')
        return recover_abandoned_containers(self.owner, execute)

    def test_missing_owner_and_previous_boot_are_recovered(self):
        with patch('acprof.host.container_lifecycle._process_identity', side_effect=FileNotFoundError):
            self.assertEqual(self.recover(self.owner), (self.identifier,))
        labels = {**self.owner, 'org.acprof.owner.boot': '00000000-0000-0000-0000-000000000000'}
        self.assertEqual(self.recover(labels), (self.identifier,))

    def test_unreadable_or_malformed_procfs_does_not_prove_owner_exited(self):
        for error in (PermissionError, ValueError, IndexError):
            with self.subTest(error=error), patch('acprof.host.container_lifecycle._process_identity', side_effect=error):
                self.assertEqual(self.recover(self.owner), ())
        self.assertFalse(self.removed)

    def test_unknown_lifecycle_and_incomplete_labels_are_preserved(self):
        for labels in ({}, {**self.owner, 'org.acprof.container.lifecycle': '2'},
                       {**self.owner, 'org.acprof.owner.pid': '-1'},
                       {**self.owner, 'org.acprof.owner.start': 'unknown'}):
            with self.subTest(labels=labels):
                self.assertEqual(self.recover(labels), ())
        self.assertFalse(self.removed)

    def test_concurrent_removal_is_safe_but_inspection_failure_is_visible(self):
        self.assertEqual(self.recover(self.owner, inspect_code=1, inspect_error='No such object'), ())
        with self.assertRaisesRegex(RuntimeError, 'Cannot inspect'):
            self.recover(self.owner, inspect_code=1, inspect_error='permission denied')
        self.assertFalse(self.removed)

    def test_cleanup_failure_aborts_before_starting_another_container(self):
        with patch('acprof.host.container_lifecycle._process_identity', side_effect=FileNotFoundError):
            with self.assertRaisesRegex(RuntimeError, 'Cannot remove'):
                self.recover(self.owner, remove_code=1)
