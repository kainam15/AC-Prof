from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from acprof.host import docker_runtime as docker
from acprof.host.detect import TaskInfo


class ContainerOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.task = TaskInfo("fixture/model", "fill-mask", "nlp", "transformers_pipeline",
                             "transformers", "a" * 40, "fixture")
        self.commands = []
        self.identifier = "b" * 64

    def start(self):
        return docker._start_container_session(self.task, 1, 2, "off", docker.ImageInfo("fixture"),
                                               "same-case", "[test]")

    def command(self, args, **kwargs):
        self.commands.append(args)
        return SimpleNamespace(returncode=0, stdout=self.identifier if args[:2] == ["docker", "run"] else "", stderr="")

    def test_failed_launch_never_removes_an_existing_container(self):
        def failed(args, **kwargs):
            self.commands.append(args)
            return SimpleNamespace(returncode=1, stdout="", stderr="container name already in use")
        with patch.object(docker, "_run", side_effect=failed), self.assertRaises(RuntimeError):
            self.start()
        self.assertFalse(any(args[:2] == ["docker", "rm"] for args in self.commands))

    def test_sessions_have_unique_names_and_cleanup_uses_the_owned_immutable_id(self):
        response = SimpleNamespace(status_code=200, text="", json=lambda: {"status": "ok"})
        with patch.object(docker, "_run", side_effect=self.command), patch("requests.get", return_value=response), redirect_stdout(io.StringIO()):
            first, second = self.start(), self.start()
            docker._stop_container_session(first)
        self.assertNotEqual(first.name, second.name)
        removed = [args for args in self.commands if args[:2] in (["docker", "rm"], ["docker", "stop"])]
        self.assertEqual([args[-1] for args in removed], [self.identifier, self.identifier])

    def test_startup_failure_removes_only_the_created_id(self):
        state = {"Running": False, "Restarting": False, "OOMKilled": True, "ExitCode": 137}
        with patch.object(docker, "_run", side_effect=self.command), patch.object(docker, "_inspect_container_state", return_value=state), patch(
            "requests.get", side_effect=ConnectionError("not ready")
        ), redirect_stdout(io.StringIO()), self.assertRaises(docker.ContainerStartupError):
            self.start()
        self.assertEqual([args[-1] for args in self.commands if args[:2] == ["docker", "rm"]], [self.identifier])

    def test_failed_docker_start_cleans_its_cidfile_id(self):
        def failed(args, **kwargs):
            if args[:2] == ["docker", "run"]:
                Path(args[args.index("--cidfile") + 1]).write_text(self.identifier)
                self.commands.append(args)
                return SimpleNamespace(returncode=125, stdout="", stderr="port already allocated")
            return self.command(args, **kwargs)
        with patch.object(docker, "_run", side_effect=failed), self.assertRaisesRegex(RuntimeError, "port already allocated"):
            self.start()
        self.assertEqual([args[-1] for args in self.commands if args[:2] == ["docker", "rm"]], [self.identifier])

    def test_cancelled_readiness_cleans_only_owned_container(self):
        with patch.object(docker, "_run", side_effect=self.command), patch(
            "requests.get", side_effect=KeyboardInterrupt
        ), self.assertRaises(KeyboardInterrupt):
            self.start()
        self.assertEqual([args[-1] for args in self.commands if args[:2] == ["docker", "rm"]], [self.identifier])


if __name__ == "__main__":
    unittest.main()
