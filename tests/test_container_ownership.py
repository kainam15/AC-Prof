import hashlib
import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acprof.host import container_state, docker_runtime, runtime_images
from acprof.host.detect import TaskInfo


class ContainerOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.task = TaskInfo("fixture/model", "fill-mask", "nlp", "transformers_pipeline",
                             "transformers", "a" * 40, "fixture")
        self.commands = []
        self.identifier = "b" * 64

    def start(self):
        return docker_runtime.start_container_session(self.task, 1, 2, "off", runtime_images.ImageInfo("fixture"),
                                               "same-case", "[test]")

    def command(self, args, **kwargs):
        self.commands.append(args)
        return SimpleNamespace(returncode=0, stdout=self.identifier if args[:2] == ["docker", "run"] else "", stderr="")

    def test_failed_launch_never_removes_an_existing_container(self):
        def failed(args, **kwargs):
            self.commands.append(args)
            return SimpleNamespace(returncode=1, stdout="", stderr="container name already in use")
        with patch("acprof.host.command.run_command", side_effect=failed), self.assertRaises(RuntimeError):
            self.start()
        self.assertFalse(any(args[:2] == ["docker", "rm"] for args in self.commands))

    def test_sessions_have_unique_names_and_cleanup_uses_the_owned_immutable_id(self):
        response = SimpleNamespace(status_code=200, text="", json=lambda: {"status": "ok"})
        with patch("acprof.host.command.run_command", side_effect=self.command), patch("requests.get", return_value=response), redirect_stdout(io.StringIO()):
            first, second = self.start(), self.start()
            docker_runtime.stop_container_session(first)
        self.assertNotEqual(first.name, second.name)
        removed = [args for args in self.commands if args[:2] in (["docker", "rm"], ["docker", "stop"])]
        self.assertEqual([args[-1] for args in removed], [self.identifier, self.identifier])

    def test_launched_container_records_process_ownership_for_crash_recovery(self):
        response = SimpleNamespace(status_code=200, text='', json=lambda: {'status': 'ok'})
        with patch("acprof.host.command.run_command", side_effect=self.command), patch('requests.get', return_value=response), redirect_stdout(io.StringIO()):
            self.start()
        command = next(args for args in self.commands if args[:2] == ['docker', 'run'])
        labels = dict(command[i + 1].split('=', 1) for i, arg in enumerate(command) if arg == '--label')
        self.assertEqual(labels.get('org.acprof.container.lifecycle'), '1')
        for key in ('host', 'uid', 'boot', 'pid', 'start'):
            self.assertTrue(labels.get('org.acprof.owner.' + key))

    def test_only_proven_abandoned_local_containers_are_reclaimed_before_launch(self):
        stat = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()
        labels = {'org.acprof.container.lifecycle': '1',
                  'org.acprof.owner.host': hashlib.sha256(Path('/etc/machine-id').read_bytes()).hexdigest(),
                  'org.acprof.owner.uid': str(os.getuid()),
                  'org.acprof.owner.boot': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  'org.acprof.owner.pid': str(os.getpid()),
                  'org.acprof.owner.start': stat[19]}
        snapshots = {}
        for index, overrides in enumerate((
            {'org.acprof.owner.start': str(int(stat[19]) - 1)},  # reused PID
            {},  # live owner
            {'org.acprof.owner.host': 'another-host'},
            {'org.acprof.owner.uid': str(os.getuid() + 1)},
            {'org.acprof.owner.start': ''},  # incomplete identity
        )):
            identifier = str(index + 1) * 64
            snapshots[identifier] = {'Id': identifier, 'Config': {'Labels': {**labels, **overrides}}}

        def execute(args, **kwargs):
            self.commands.append(args)
            if args[:2] == ['docker', 'ps']:
                return SimpleNamespace(returncode=0, stdout='\n'.join(snapshots), stderr='')
            if args[:2] == ['docker', 'inspect']:
                return SimpleNamespace(returncode=0, stdout=json.dumps([snapshots[args[-1]]]), stderr='')
            return SimpleNamespace(returncode=0, stdout=self.identifier if args[:2] == ['docker', 'run'] else '', stderr='')

        response = SimpleNamespace(status_code=200, text='', json=lambda: {'status': 'ok'})
        with patch("acprof.host.command.run_command", side_effect=execute), patch('requests.get', return_value=response), redirect_stdout(io.StringIO()):
            self.start()
        removed = [args[-1] for args in self.commands if args[:2] == ['docker', 'rm']]
        self.assertEqual(removed, ['1' * 64])
        self.assertLess(next(i for i, c in enumerate(self.commands) if c[:2] == ['docker', 'rm']),
                        next(i for i, c in enumerate(self.commands) if c[:2] == ['docker', 'run']))

    def test_startup_failure_removes_only_the_created_id(self):
        state = {"Running": False, "Restarting": False, "OOMKilled": True, "ExitCode": 137}
        with patch("acprof.host.command.run_command", side_effect=self.command), patch("acprof.host.container_state.inspect_container_state", return_value=state), patch(
            "requests.get", side_effect=ConnectionError("not ready")
        ), redirect_stdout(io.StringIO()), self.assertRaises(container_state.ContainerStartupError):
            self.start()
        self.assertEqual([args[-1] for args in self.commands if args[:2] == ["docker", "rm"]], [self.identifier])

    def test_failed_docker_start_cleans_its_cidfile_id(self):
        def failed(args, **kwargs):
            if args[:2] == ["docker", "run"]:
                Path(args[args.index("--cidfile") + 1]).write_text(self.identifier)
                self.commands.append(args)
                return SimpleNamespace(returncode=125, stdout="", stderr="port already allocated")
            return self.command(args, **kwargs)
        with patch("acprof.host.command.run_command", side_effect=failed), self.assertRaisesRegex(RuntimeError, "port already allocated"):
            self.start()
        self.assertEqual([args[-1] for args in self.commands if args[:2] == ["docker", "rm"]], [self.identifier])

    def test_cancelled_readiness_cleans_only_owned_container(self):
        with patch("acprof.host.command.run_command", side_effect=self.command), patch(
            "requests.get", side_effect=KeyboardInterrupt
        ), self.assertRaises(KeyboardInterrupt):
            self.start()
        self.assertEqual([args[-1] for args in self.commands if args[:2] == ["docker", "rm"]], [self.identifier])


if __name__ == "__main__":
    unittest.main()
