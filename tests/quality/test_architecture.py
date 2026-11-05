"""Import boundaries that keep application entry points out of core code."""
import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from scripts.check_critical_coverage import BRANCH_FLOORS
from scripts.check_private_api import read_baseline, scan_sources

PROJECT_DIR = Path(__file__).resolve().parents[2]


def _expand_regex_alternations(pattern: str) -> list[str]:
    match = re.search(r"\(([^()]+)\)", pattern)
    if not match:
        return [pattern]
    prefix, suffix = pattern[:match.start()], pattern[match.end():]
    expanded = []
    for option in match.group(1).split("|"):
        expanded.extend(_expand_regex_alternations(f"{prefix}{option}{suffix}"))
    return expanded


def test_shared_boundaries_gate_tracks_current_modules():
    config = yaml.safe_load((PROJECT_DIR / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    local_repo = next(repo for repo in config["repos"] if repo["repo"] == "local")
    hook = next(item for item in local_repo["hooks"] if item["id"] == "shared-boundaries")
    pattern = re.compile(hook["files"])
    expected = {
        "acprof/experiments/constraints.py",
        "acprof/input_budget.py",
        "acprof/image_inputs.py",
        "acprof/gpu_probe.py",
        "acprof/host/metadata/gpu.py",
    }
    assert all(pattern.fullmatch(path) for path in expected)
    assert all((PROJECT_DIR / path).is_file() for path in expected)


def test_quality_gate_module_targets_exist_in_repository():
    all_py_files = [
        path.relative_to(PROJECT_DIR).as_posix()
        for path in (PROJECT_DIR / "acprof").rglob("*.py")
    ]
    pre_commit = yaml.safe_load((PROJECT_DIR / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    local_hooks = {
        hook["id"]: hook
        for repo in pre_commit.get("repos", [])
        if repo.get("repo") == "local"
        for hook in repo.get("hooks", [])
    }
    for hook_id in ("shared-boundaries", "core-complexity"):
        pattern = local_hooks[hook_id]["files"]
        for branch in _expand_regex_alternations(pattern):
            matched = [path for path in all_py_files if re.search(branch, path)]
            assert matched, f"pre-commit hook {hook_id!r} branch {branch!r} matches no files"

    pyproject = tomllib.loads((PROJECT_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    for override in pyproject.get("tool", {}).get("mypy", {}).get("overrides", []):
        for module in override.get("module", []):
            module_rel = module.replace(".", "/")
            assert (
                (PROJECT_DIR / f"{module_rel}.py").is_file()
                or (PROJECT_DIR / module_rel / "__init__.py").is_file()
            ), f"mypy override module {module!r} does not exist"

    for path in BRANCH_FLOORS:
        assert (PROJECT_DIR / path).is_file(), f"coverage floor target {path!r} does not exist"


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
        if relative.startswith("acprof/container/") or relative == "acprof/system/command.py":
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

@pytest.mark.parametrize('name', ['probe', 'posthoc'])
def test_shared_preflight_does_not_depend_on_run_cli(name):
    path = PROJECT_DIR / "acprof" / "cli" / f"{name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    dependencies = {
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert ("acprof.cli.run") not in (dependencies)

def implementation_paths():
    packages = (
        "application", "host", "container", "workloads", "monitors", "packet",
        "analysis", "plotting", "tui", "events", "tasks", "experiments",
        "runtime", "results",
    )
    return sorted(
        path for package in packages
        for path in (PROJECT_DIR / "acprof" / package).rglob("*.py")
    )


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
            "import acprof.host.measurement.metrics; "
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
