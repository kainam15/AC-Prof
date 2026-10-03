"""Import boundaries that keep application entry points out of core code."""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_private_api import read_baseline, scan_sources

PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_repository_root_has_no_python_files():
    assert (sorted(path.name for path in PROJECT_DIR.glob("*.py"))) == ([])
    assert not ((PROJECT_DIR / "acprof-tui").exists())

def test_cprofile_uses_the_standard_library_profile():
    result = subprocess.run(
        [sys.executable, "-c", (
            "import cProfile, pathlib, profile, sysconfig; "
            "assert pathlib.Path(profile.__file__).resolve() == "
            "pathlib.Path(sysconfig.get_path('stdlib'), 'profile.py').resolve(); "
            "assert cProfile.Profile().runcall(sum, [1, 2, 3]) == 6"
        )], cwd=PROJECT_DIR, capture_output=True, text=True, check=False,
    )
    assert (result.returncode) == (0), result.stderr

def test_run_orchestrator_does_not_own_tmux_subprocesses():
    tree = ast.parse((PROJECT_DIR / "acprof/cli/run.py").read_text())
    tmux_commands = [node for node in ast.walk(tree)
                     if isinstance(node, ast.List) and node.elts
                     and isinstance(node.elts[0], ast.Constant) and node.elts[0].value == "tmux"]
    assert (tmux_commands) == ([])

def test_private_dependencies_match_the_reviewed_baseline():
    actual = set(scan_sources(PROJECT_DIR))
    baseline = read_baseline(PROJECT_DIR / "tests/private_api_baseline.json")
    assert (actual) == (baseline), "Run scripts/check_private_api.py for dependency locations"

def test_short_host_commands_use_the_registered_runner():
    # Resolve import aliases too, so `from subprocess import run` cannot bypass it.
    violations = []
    for path in (PROJECT_DIR / "acprof").rglob("*.py"):
        relative = path.relative_to(PROJECT_DIR).as_posix()
        if relative.startswith("acprof/container/") or relative == "acprof/host/command.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                aliases.update((a.asname or a.name, a.name) for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module == "subprocess":
                aliases.update((a.asname or a.name, f"subprocess.{a.name}") for a in node.names)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                root, _, rest = name.partition(".")
                resolved = aliases.get(root, root) + ("." + rest if rest else "")
                if resolved in {f"subprocess.{method}" for method in ("run", "call", "check_call", "check_output", "getoutput", "getstatusoutput")}:
                    violations.append(f"{relative}:{node.lineno}: {resolved}")
    assert (violations) == ([])

@pytest.mark.parametrize('name', ('probe', 'posthoc'))
def test_shared_preflight_does_not_depend_on_run_cli(name):
    path = PROJECT_DIR / "acprof" / "cli" / f"{name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    dependencies = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert ("acprof.cli.run") not in (dependencies)

def implementation_paths():
    paths = [path for package in ("host", "container", "workloads", "monitors", "packet",
                                  "analysis", "plotting", "tui")
             for path in (PROJECT_DIR / "acprof" / package).rglob("*.py")]
    paths.extend(PROJECT_DIR / "acprof" / name for name in ("experiment.py", "run_args.py"))
    return sorted(paths)


@pytest.mark.parametrize("path", implementation_paths(),
                         ids=lambda path: path.relative_to(PROJECT_DIR).as_posix())
def test_implementation_packages_do_not_import_cli(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    dependencies = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            dependencies.append(node.module)
        elif isinstance(node, ast.Import):
            dependencies.extend(alias.name for alias in node.names)
    assert not (any(name == "acprof.cli" or name.startswith("acprof.cli.")
            for name in dependencies)), dependencies

def test_analysis_import_does_not_load_rendering():
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; "
            "import acprof.analysis.latency_report; "
            "import acprof.analysis.model; "
            "assert not any(name == 'matplotlib' or name.startswith('matplotlib.') "
            "for name in sys.modules); "
            "assert 'plotly' not in sys.modules; "
            "assert 'acprof.host.run_state' not in sys.modules; "
            "assert 'acprof.cli.plot' not in sys.modules"
        )],
        cwd=PROJECT_DIR, capture_output=True, text=True, check=False,
    )
    assert (result.returncode) == (0), result.stderr

def test_metrics_import_does_not_initialize_measurement():
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; "
            "import acprof.host.client_metrics; "
            "assert 'acprof.host.client' not in sys.modules; "
            "assert not any(name.startswith('acprof.monitors') "
            "for name in sys.modules)"
        )],
        cwd=PROJECT_DIR, capture_output=True, text=True, check=False,
    )
    assert (result.returncode) == (0), result.stderr

def test_tui_commands_import_does_not_load_textual():
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import acprof.tui.commands; "
            "assert not any(name == 'textual' or name.startswith('textual.') "
            "for name in sys.modules)"
        )],
        cwd=PROJECT_DIR, capture_output=True, text=True, check=False,
    )
    assert (result.returncode) == (0), result.stderr
