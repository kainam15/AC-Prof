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
    assert "gh pr merge" not in commands and "gh release" not in commands


def test_lock_refresh_generates_and_verifies_coordinated_artifacts():
    steps = workflow()["jobs"]["refresh"]["steps"]
    scripts = "\n".join(step.get("run", "") for step in steps)
    assert "scripts/compile_locks.py --host-only --upgrade" in scripts
    assert "scripts/compile_locks.py --runtime-only --upgrade" in scripts
    assert "uv pip compile" in scripts and "--generate-hashes" in scripts
    assert "scripts/compile_locks.py --host-only --check" in scripts
    assert "scripts/compile_locks.py --check" in scripts
    assert "tests/test_lock_compiler.py" in scripts
    assert "git add -A requirements dockerfiles/locks" in scripts
    assert "git diff --cached --quiet" in scripts
    assert "git push" in scripts
    assert "gh workflow run ci.yml" in scripts


def test_refresh_scripts_restrict_publication_and_supply_input_defaults():
    data = workflow()
    steps = data["jobs"]["refresh"]["steps"]
    env = data["jobs"]["refresh"]["env"]
    assert "all" in env["REFRESH_SCOPE"]
    for step in steps:
        if step.get("name", "").startswith("Refresh"):
            assert "scope" in step["if"].lower() or "REFRESH_SCOPE" in step["if"]
    assert "pull_request" not in data["on"]
