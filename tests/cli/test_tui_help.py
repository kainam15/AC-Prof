"""CLI help and argument errors must work without loading the TUI runtime."""
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
