"""工具入口必须可以脱离源码工作目录运行。"""
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.experiment import RunConfig, build_run_command

ROOT = Path(__file__).resolve().parents[1]


class TestDistribution:
    def invoke(self, arguments, cwd):
        return subprocess.run(arguments, cwd=cwd, text=True, capture_output=True,
                              env={**os.environ, "PYTHONPATH": str(ROOT)}, timeout=30)

    def test_module_help_from_an_empty_workspace(self):
        with tempfile.TemporaryDirectory() as workspace:
            result = self.invoke([sys.executable, "-m", "acprof", "--help"], workspace)
            assert (result.returncode) == (0), result.stderr
            for command in ("run", "tui", "probe", "plot", "doctor"):
                assert (command) in (result.stdout)
            assert (list(Path(workspace).iterdir())) == ([])

    def test_unknown_command_is_an_argument_error(self):
        with tempfile.TemporaryDirectory() as workspace:
            result = self.invoke([sys.executable, "-m", "acprof", "unknown"], workspace)
            assert (result.returncode) == (2), result.stderr
            assert ("Traceback") not in (result.stderr)

    @pytest.mark.parametrize('command', ('run', 'probe', 'profile', 'plot', 'audit', 'stats', 'tui'))
    def test_public_help_uses_only_the_console_command(self, command):
        with tempfile.TemporaryDirectory() as workspace:
            result = self.invoke(
                [sys.executable, "-m", "acprof.cli.main", command, "--help"], workspace,
            )
            assert (result.returncode) == (0), result.stderr
            assert (f"usage: acprof {command}") in (result.stdout)
            assert not re.search(r"\b(?:run|probe|profile|plot|audit|stats|tui)\.py\b", result.stdout)
            assert (list(Path(workspace).iterdir())) == ([])

    def test_doctor_json_is_machine_readable_even_when_prerequisites_are_missing(self):
        with tempfile.TemporaryDirectory() as workspace:
            result = self.invoke([sys.executable, "-m", "acprof", "doctor", "--json",
                                  "--profiling-mode", "basic"], workspace)
            assert (result.returncode) in ((0, 1)), result.stderr
            assert (result.stdout.startswith("{")), result.stderr
            report = json.loads(result.stdout)
            assert (report["profiling_mode"]) == ("basic")
            assert (report["ready"]) == (result.returncode == 0)
            assert (report["checks"])

    def test_tui_run_command_works_without_root_scripts(self):
        with tempfile.TemporaryDirectory() as workspace:
            command = build_run_command(RunConfig.smoke("example/model"),
                                        project_dir=Path(workspace))
            result = self.invoke([*command, "--help"], workspace)
            assert (result.returncode) == (0), result.stderr
            assert ("--profiling-mode") in (result.stdout)

    def test_recorded_command_uses_the_public_cli_and_preserves_quoting(self):
        from acprof.cli.run import format_run_command
        arguments = ["--model", "example/model", "--output-dir", "/a path/results;literal"]
        assert (shlex.split(format_run_command(["acprof run", *arguments]))) == (["acprof", "run", *arguments])
        assert (format_run_command([])) == ("acprof run")
        with patch("sys.frozen", True, create=True), patch("sys.executable", "/opt/AC Prof/acprof"):
            command = format_run_command(["acprof run", *arguments])
        assert (shlex.split(command)) == (["acprof", "run", *arguments])

    @pytest.mark.parametrize('prefix', (['/opt/python', '/opt/bin/acprof', 'run'], ['/opt/acprof-linux-x86_64', 'run'], ['/opt/python', '-m', 'acprof', 'run'], ['/opt/python', '-u', '-m', 'acprof', 'run']))
    def test_posthoc_recognizes_installed_runs_in_their_own_workspace(self, prefix):
        from acprof.host.posthoc.storage import find_active_processes
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = root / "proc" / "999999999"
            process.mkdir(parents=True)
            (process / "cwd").symlink_to(root, target_is_directory=True)
            (process / "cmdline").write_bytes(b"\0".join(value.encode() for value in
                [*prefix, "--model", "org/model", "--output-dir", "results"]))
            with patch("acprof.host.posthoc.storage.Path", side_effect=lambda value:
                       root / "proc" if value == "/proc" else Path(value)):
                matches = find_active_processes(root / "results" / "org--model", model_id="org/model")
            assert ([pid for pid, _ in matches]) == ([999999999])

    @pytest.mark.parametrize('prefix', (['/opt/python', '/opt/bin/acprof', 'profile'], ['/opt/acprof-linux-x86_64', 'profile'], ['/opt/python', '-u', '-m', 'acprof', 'profile']))
    def test_posthoc_recognizes_installed_profile_processes(self, prefix):
        from acprof.host.posthoc.storage import find_active_processes
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_dir = root / "results" / "org--model"
            process = root / "proc" / "999999999"
            process.mkdir(parents=True)
            (process / "cmdline").write_bytes(b"\0".join(value.encode() for value in
                [*prefix, str(result_dir)]))
            with patch("acprof.host.posthoc.storage.Path", side_effect=lambda value:
                       root / "proc" if value == "/proc" else Path(value)):
                matches = find_active_processes(result_dir)
            assert ([pid for pid, _ in matches]) == ([999999999])

    def test_posthoc_recognizes_profiler_output_below_result_directory(self):
        from acprof.host.posthoc.storage import find_active_processes

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_dir = root / "results" / "org--model"
            process = root / "proc" / "999999999"
            process.mkdir(parents=True)
            output = result_dir / "raw" / "posthoc_profiles" / "trace"
            (process / "cmdline").write_bytes(
                b"\0".join(value.encode() for value in ["/usr/bin/nsys", "profile", f"--output={output}"])
            )
            with patch(
                "acprof.host.posthoc.storage.Path",
                side_effect=lambda value: root / "proc" if value == "/proc" else Path(value),
            ):
                matches = find_active_processes(result_dir)

        assert [pid for pid, _ in matches] == [999999999]

    def test_posthoc_does_not_treat_sibling_result_prefix_as_active(self):
        from acprof.host.posthoc.storage import find_active_processes

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "results" / "org--model"
            sibling = root / "results" / "org--model-old"
            process = root / "proc" / "999999999"
            process.mkdir(parents=True)
            (process / "cmdline").write_bytes(
                b"\0".join(value.encode() for value in ["/opt/bin/acprof", "profile", str(sibling)])
            )
            with patch(
                "acprof.host.posthoc.storage.Path",
                side_effect=lambda value: root / "proc" if value == "/proc" else Path(value),
            ):
                matches = find_active_processes(expected)

        assert matches == []
