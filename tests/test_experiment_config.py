"""Headless tools and CLI conversions share configuration without importing TUI."""
import subprocess
import sys
import unittest


class ExperimentConfigTests(unittest.TestCase):
    def test_cli_namespace_and_command_share_normalization(self):
        from pathlib import Path

        from acprof.experiment import RunConfig, build_run_command
        from acprof.run_args import build_parser
        args = build_parser().parse_args(["--model", "demo/model", "--cpuset-cpus", "3,1-2",
                                         "--gpus", "off", "--cpus", "1"])
        config = RunConfig.from_namespace(args).validate()
        self.assertEqual(config.cpuset_cpus, "1-3")
        command = build_run_command(config, project_dir=Path.cwd())
        self.assertEqual(command[command.index("--cpuset-cpus") + 1], "1-3")

    def test_hardware_tool_does_not_import_ui_package(self):
        result = subprocess.run([sys.executable, "-c", "import sys; import scripts.check_hardware; "
                                 "assert not any(n == 'acprof.tui' or n.startswith('acprof.tui.') for n in sys.modules)"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
