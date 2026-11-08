"""CLI help and argument errors must work without loading the TUI runtime."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def invoke_cli(arguments: list[str], workspace: Path) -> subprocess.CompletedProcess[str]:
    # Exclude site-packages, editable-install hooks and ambient PYTHONPATH.
    return subprocess.run(
        [sys.executable, "-I", "-S", "-c", """
import runpy
import sys

sys.path.insert(0, sys.argv[1])
sys.argv = ["acprof", *sys.argv[2:]]
runpy.run_module("acprof", run_name="__main__")
""", str(ROOT), *arguments],
        cwd=workspace, capture_output=True, text=True, check=False, timeout=10,
    )


@pytest.mark.parametrize("help_flag", ["--help", "-h"])
def test_tui_help_without_runtime_dependencies(tmp_path, help_flag):
    result = invoke_cli(["tui", help_flag], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "usage: acprof tui" in result.stdout
    for option in ("--model", "--preset", "--output-dir", "--color-system"):
        assert option in result.stdout
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("arguments", "error"), [
    (["--preset", "unknown"], "invalid choice"),
    (["--color-system", "unknown"], "invalid choice"),
    (["--model"], "expected one argument"),
    (["--output-dir"], "expected one argument"),
    (["--unknown-option"], "unrecognized arguments"),
])
def test_tui_argument_errors_without_runtime_dependencies(tmp_path, arguments, error):
    result = invoke_cli(["tui", *arguments], tmp_path)
    assert result.returncode == 2, result.stderr
    assert "usage: acprof tui" in result.stderr
    assert error in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("arguments", "output"), [
    (["--help"], "usage: acprof"),
    (["--version"], "AC-Prof "),
])
def test_top_level_help_and_version_without_runtime_dependencies(tmp_path, arguments, output):
    result = invoke_cli(arguments, tmp_path)
    assert result.returncode == 0, result.stderr
    assert output in result.stdout
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []


def test_tui_start_does_not_hide_missing_runtime_dependencies(tmp_path):
    result = invoke_cli(["tui"], tmp_path)
    assert result.returncode == 1
    assert "ModuleNotFoundError" in result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("terminal_env", "version", "expected_version", "expected_width"), [
    ({"TERM_PROGRAM": "vscode"}, None, "11.0.0", 1),
    ({"TERM_PROGRAM": "vscode"}, "17.0.0", "17.0.0", 2),
    ({"WT_SESSION": "fixture-session"}, None, "11.0.0", 1),
    ({"WT_SESSION": "fixture-session", "WSL_DISTRO_NAME": "Ubuntu"}, None, "11.0.0", 1),
    ({"WT_SESSION": "fixture-session"}, "17.0.0", "17.0.0", 2),
    ({"TERM_PROGRAM": "xterm"}, None, None, 2),
    ({"WT_SESSION": "", "WSL_DISTRO_NAME": "Ubuntu"}, None, None, 2),
])
def test_tui_start_matches_terminal_unicode_before_rendering(
    terminal_env, version, expected_version, expected_width,
):
    # Rich caches its first auto-selected width table; each launch needs a fresh process.
    env = {key: value for key, value in os.environ.items()
           if key not in {"TERM_PROGRAM", "WT_SESSION", "WSL_DISTRO_NAME", "UNICODE_VERSION"}}
    env.update(terminal_env)
    if version is not None:
        env["UNICODE_VERSION"] = version
    result = subprocess.run(
        [sys.executable, "-c", """
import json
import os
import sys
from types import SimpleNamespace

class WidthProbe:
    def __init__(self, **kwargs):
        from rich.cells import cell_len
        from acprof.tui.ui.navigation import PageNavigation
        icon = PageNavigation.ICONS['run-tab']
        print(json.dumps([os.environ.get('UNICODE_VERSION'), icon,
                          cell_len(icon), cell_len('实验配置')]))
        raise SystemExit(0)

sys.modules['acprof.tui.app'] = SimpleNamespace(AcprofTui=WidthProbe)
sys.modules['acprof.tui.run.form'] = SimpleNamespace(infer_preset=None)
from acprof.cli.main import main
main(['tui'])
"""],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [expected_version, "☰", expected_width, 8]
