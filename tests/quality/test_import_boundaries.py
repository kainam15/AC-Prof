"""Exercise declared dependency contracts against real isolated Python packages."""
from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[2]
CHECKER = PROJECT_DIR / "scripts" / "check_import_boundaries.py"
CONFIGURATION = (PROJECT_DIR / "pyproject.toml").read_text(encoding="utf-8")


def write_module(root: Path, module: str, source: str = "") -> None:
    parts = module.split(".")
    for length in range(1, len(parts)):
        package = root.joinpath(*parts[:length])
        package.mkdir(exist_ok=True)
        (package / "__init__.py").touch()
    root.joinpath(*parts).with_suffix(".py").write_text(source, encoding="utf-8")


@pytest.fixture
def package_root(tmp_path):
    (tmp_path / "pyproject.toml").write_text(CONFIGURATION, encoding="utf-8")
    contracts = tomllib.loads(CONFIGURATION)["tool"]["importlinter"]["contracts"]
    for contract in contracts:
        for module in [*contract["source_modules"], *contract["forbidden_modules"]]:
            package = tmp_path.joinpath(*module.split("."))
            package.mkdir(parents=True, exist_ok=True)
            (package / "__init__.py").touch()
    (tmp_path / "acprof" / "__init__.py").touch()
    for module in (
        "analysis.report", "host.hardware_conditions", "host.orchestrator", "cli.run",
        "container.server", "tui.app", "monitors.common", "host.other", "models.contract",
        "host.detect", "models.inspection.source", "container.model_probe", "hub.repository",
        "host.model_store.store", "system.command", "events.progress", "tui.run.progress",
        "tasks.groups", "experiments.configuration", "runtime.profiles", "application.run",
        "results.csv", "host.run_state", "host.command", "monitors.resource_usage", "bridge",
    ):
        write_module(tmp_path, f"acprof.{module}")
    return tmp_path


def run_checker(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(root)],
        cwd=root, capture_output=True, text=True, check=False, timeout=20,
    )


@pytest.mark.parametrize(
    ("module", "source"),
    [
        ("acprof.analysis.report", "from acprof.host import hardware_conditions"),
        ("acprof.analysis.report", "from ..host.hardware_conditions import conditions_path"),
        ("acprof.host.orchestrator", "import acprof.cli.run"),
        ("acprof.container.server", "from acprof.tui.app import AcprofTui"),
        ("acprof.monitors.common", "from acprof.host.other import execute"),
        ("acprof.models.contract", "from acprof.host.detect import detect_task"),
        ("acprof.models.inspection.source", "from acprof.container.model_probe import validate"),
        ("acprof.hub.repository", "from acprof.host.model_store.store import store_root"),
        ("acprof.system.command", "from acprof.host.orchestrator import execute"),
        ("acprof.events.progress", "from acprof.tui.run.progress import RunProgressTracker"),
        ("acprof.tasks.groups", "from acprof.host.detect import detect_task"),
        ("acprof.experiments.configuration", "from acprof.cli.run import main"),
        ("acprof.runtime.profiles", "from acprof.application.run import RunApplication"),
        ("acprof.results.csv", "from acprof.host.run_state import load_state"),
        ("acprof.host.orchestrator", "from acprof.application.run import RunApplication"),
        ("acprof.runtime.profiles", "from acprof.host.detect import detect_task"),
    ],
)
def test_reverse_dependencies_are_rejected(package_root, module, source):
    write_module(package_root, module, source)
    result = run_checker(package_root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "BROKEN" in result.stdout
    assert module in result.stdout


@pytest.mark.parametrize("guard", ["TYPE_CHECKING", "typing.TYPE_CHECKING"])
def test_runtime_type_only_imports_are_excluded(package_root, guard):
    source = (
        "import typing\nfrom typing import TYPE_CHECKING\n"
        f"if {guard}:\n    from acprof.host.hardware_conditions import conditions_path\n"
    )
    write_module(package_root, "acprof.analysis.report", source)
    result = run_checker(package_root)
    assert result.returncode == 0, result.stdout + result.stderr


def test_runtime_else_branch_is_not_hidden_by_type_guard(package_root):
    source = (
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    pass\n"
        "else:\n    from acprof.host.hardware_conditions import conditions_path\n"
    )
    write_module(package_root, "acprof.analysis.report", source)
    result = run_checker(package_root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "BROKEN" in result.stdout


@pytest.mark.parametrize(
    ("module", "dependency", "allowed"),
    [
        ("acprof.monitors.common", "acprof.system.command", True),
        ("acprof.monitors.resource_usage", "acprof.system.command", True),
        ("acprof.monitors.common", "acprof.host.command", False),
    ],
)
def test_shared_runner_has_no_monitor_host_exception(package_root, module, dependency, allowed):
    write_module(package_root, module, f"from {dependency} import run_command")
    result = run_checker(package_root)
    assert result.returncode == (0 if allowed else 1), result.stdout + result.stderr


@pytest.mark.parametrize(
    ("module", "dependency"),
    [
        ("acprof.host.orchestrator", "acprof.application.run"),
        ("acprof.runtime.profiles", "acprof.host.detect"),
        ("acprof.runtime.profiles", "acprof.application.run"),
    ],
)
def test_orchestration_boundaries_reject_indirect_imports(package_root, module, dependency):
    write_module(package_root, module, "import acprof.bridge")
    write_module(package_root, "acprof.bridge", f"import {dependency}")
    result = run_checker(package_root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "BROKEN" in result.stdout
    assert "acprof.bridge" in result.stdout


def test_invalid_python_does_not_pass_boundary_check(package_root):
    write_module(package_root, "acprof.host.orchestrator", "def broken(\n")
    result = run_checker(package_root)
    assert result.returncode != 0, result.stdout + result.stderr


def test_actual_repository_does_not_reverse_import():
    result = run_checker(PROJECT_DIR)
    assert result.returncode == 0, result.stdout + result.stderr
