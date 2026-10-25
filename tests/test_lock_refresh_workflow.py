"""Review-only scheduled dependency refresh must preserve lock coordination and CI."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def workflow():
    return yaml.load((ROOT / ".github/workflows/lock-refresh.yml").read_text(), Loader=yaml.BaseLoader)


def test_lock_refresh_runs_only_on_main_and_is_not_self_merging():
    data = workflow()
    assert data["on"]["schedule"]
    assert data["on"]["workflow_dispatch"]["inputs"]["scope"]["options"] == ["all", "host", "runtime"]
    job = data["jobs"]["refresh"]
    assert job["if"] == "github.ref == 'refs/heads/main'"
    assert job["permissions"] == {"contents": "write", "pull-requests": "write", "actions": "write"}
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "gh pr create" in commands
    assert "gh workflow run ci.yml" in commands
    assert "gh pr merge" not in commands
    assert "gh release" not in commands


def test_lock_refresh_generates_and_verifies_coordinated_artifacts():
    steps = workflow()["jobs"]["refresh"]["steps"]
    scripts = "\n".join(step.get("run", "") for step in steps)
    assert "uv lock --upgrade" in scripts
    assert "uv sync --locked" in scripts
    assert "scripts/compile_locks.py --upgrade" in scripts
    assert "uv pip compile" not in scripts
    assert "uv lock --check" in scripts
    assert "scripts/compile_locks.py --check" in scripts
    assert "tests/test_lock_compiler.py" in scripts
    assert "git add -A uv.lock dockerfiles/locks" in scripts
    assert "git diff --cached --quiet" in scripts
    assert "git push" in scripts
    assert "gh workflow run ci.yml" in scripts


def test_refresh_installs_locked_python_deps_before_runtime_compilation():
    steps = workflow()["jobs"]["refresh"]["steps"]
    positions = {step["name"]: i for i, step in enumerate(steps) if "name" in step}
    host = positions["Refresh the project lock and all dependency groups"]
    install = positions["Prepare the locked development toolchain"]
    runtime = positions["Refresh Docker runtime Python artifact locks"]
    verify = positions["Check regenerated hashes, profiles and metadata"]

    assert host < install < runtime < verify
    assert "if" not in steps[install]  # Also required for runtime-only refreshes.
    assert steps[install]["run"].strip() == "uv sync --locked"
    assert (
        "uv run --locked python scripts/compile_locks.py --upgrade"
        in steps[runtime]["run"]
    )
    assert "uv sync" not in steps[verify]["run"]


def test_refresh_scripts_restrict_publication_and_supply_input_defaults():
    data = workflow()
    steps = data["jobs"]["refresh"]["steps"]
    env = data["jobs"]["refresh"]["env"]
    assert "all" in env["REFRESH_SCOPE"]
    for step in steps:
        if step.get("name", "").startswith("Refresh"):
            assert "scope" in step["if"].lower() or "REFRESH_SCOPE" in step["if"]
    assert "pull_request" not in data["on"]
