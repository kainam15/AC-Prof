"""Security audit findings remain warnings, malformed reports fail closed."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_dependency_audit.py"


def test_workflow_covers_both_host_lock_interpreter_branches_without_artifact_collisions():
    workflow = (ROOT / '.github/workflows/dependency-audit.yml').read_text()
    rows = re.findall(r"\{lock: ([a-z-]+), python: '([0-9.]+)'\}", workflow)
    assert len(rows) == len(set(rows))
    assert {('host', '3.10'), ('host', '3.12'), ('test', '3.12'), ('dev', '3.12')} <= set(rows)
    assert 'name: dependency-audit-${{ matrix.lock }}-${{ matrix.python }}' in workflow


def run_check(tmp_path, data):
    path = tmp_path / "audit.json"
    if isinstance(data, str):
        path.write_text(data)
    elif data is not None:
        path.write_text(json.dumps(data))
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_STEP_SUMMARY"}
    env["GITHUB_STEP_SUMMARY"] = str(tmp_path / "summary.md")
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(path), "host.lock"],
        capture_output=True, text=True, env=env, check=False,
    )


def test_valid_empty_vulnerability_list_is_reported_as_clean(tmp_path):
    result = run_check(tmp_path, {
        "dependencies": [{"name": "example", "version": "1.0", "vulns": []}],
        "fixes": [],
    })
    assert result.returncode == 0
    assert "audited 1 locked packages; 0" in result.stdout
    assert "::warning::" not in result.stdout
    assert "No known vulnerabilities" in (tmp_path / "summary.md").read_text()


def test_known_vulnerabilities_are_reported_without_blocking(tmp_path):
    result = run_check(tmp_path, {
        "dependencies": [
            {"name": "example", "version": "1.0",
             "vulns": [{"id": "CVE-fixture", "fix_versions": ["1.1"]}]},
            {"name": "clean", "version": "1.0", "vulns": []},
        ],
        "fixes": [],
    })
    assert result.returncode == 0
    assert "audited 2 locked packages; 1 known vulnerability" in result.stdout
    assert "::warning::" in result.stdout
    assert "example: 1" in (tmp_path / "summary.md").read_text()


@pytest.mark.parametrize("data", [
    None, "", "{", "[]", "{}", {"dependencies": []},
    {"dependencies": [{"name": "example"}]},
    {"dependencies": [{"name": "example", "vulns": ["bad"]}]},
])
def test_missing_or_invalid_evidence_fails_closed(tmp_path, data):
    result = run_check(tmp_path, data)
    assert result.returncode == 2
    assert "audit evidence unavailable/invalid" in result.stderr
