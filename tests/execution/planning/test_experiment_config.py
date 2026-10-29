"""Headless tools and CLI conversions share configuration without importing TUI."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize("entrypoint", ["run", "probe"])
def test_image_policy_defaults_to_reuse_and_allows_explicit_rebuild(entrypoint):
    from pathlib import Path

    from acprof.cli.probe import _build_parser
    from acprof.experiment import RunConfig, build_run_command
    from acprof.run_args import build_parser
    from acprof.tui.commands import build_probe_command

    parser = build_parser() if entrypoint == "run" else _build_parser()
    builder = build_run_command if entrypoint == "run" else build_probe_command
    for rebuild in (False, True):
        argv = ["--model", "demo/model", *(["--rebuild-image"] if rebuild else [])]
        config = RunConfig.from_namespace(parser.parse_args(argv))
        command = builder(config, project_dir=Path.cwd())
        assert ("--rebuild-image" in command) is rebuild
        assert "--skip-build" not in command
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--model", "demo/model", "--skip-build"])
    assert error.value.code == 2


def test_rebuild_cannot_change_a_resumed_experiment():
    from acprof.experiment import RunConfig, RunConfigError

    with pytest.raises(RunConfigError, match="重建"):
        RunConfig(model="demo/model", resume=True, rebuild_image=True).validate()


def test_image_execution_policy_does_not_change_frozen_measurement_options():
    from acprof.host.run_state import run_options
    from acprof.run_args import build_parser, flatten_run_options

    parser = build_parser()
    normal = parser.parse_args(["--model", "demo/model"])
    rebuild = parser.parse_args(["--model", "demo/model", "--rebuild-image"])
    assert run_options(normal) == run_options(rebuild)
    assert flatten_run_options({"model": "demo/model", "skip_build": False}) == {"model": "demo/model"}


def test_cli_namespace_and_command_share_normalization():
    from pathlib import Path

    from acprof.experiment import RunConfig, build_run_command
    from acprof.run_args import build_parser
    args = build_parser().parse_args(["--model", "demo/model", "--cpuset-cpus", "3,1-2",
                                     "--gpus", "off", "--cpus", "1"])
    config = RunConfig.from_namespace(args).validate()
    assert (config.cpuset_cpus) == ("1-3")
    command = build_run_command(config, project_dir=Path.cwd())
    assert (command[command.index("--cpuset-cpus") + 1]) == ("1-3")

def test_hardware_tool_does_not_import_ui_package():
    result = subprocess.run([sys.executable, "-c", "import sys; import scripts.check_hardware; "
                             "assert not any(n == 'acprof.tui' or n.startswith('acprof.tui.') for n in sys.modules)"],
                            capture_output=True, text=True)
    assert (result.returncode) == (0), result.stderr
