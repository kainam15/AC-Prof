"""Security audit findings remain warnings, malformed reports fail closed."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "check_dependency_audit.py"


def test_workflow_scans_supported_host_python_and_tracks_artifact_identity():
    workflow = (ROOT / '.github/workflows/dependency-audit.yml').read_text()
    rows = re.findall(r"\{lock: ([a-z-]+), python: '([0-9.]+)'\}", workflow)
    assert len(rows) == len(set(rows))
    assert {('host', '3.12'), ('test', '3.12'), ('dev', '3.12')} <= set(rows)
    assert all(python != '3.10' for _, python in rows)
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
    assert "audited 1 packages, skipped 0 packages; 0" in result.stdout
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
    assert "audited 2 packages, skipped 0 packages; 1 known vulnerability" in result.stdout
    assert "::warning::" in result.stdout
    assert "example: 1" in (tmp_path / "summary.md").read_text()


def test_cuda_wheels_skipped_explicitly_without_claiming_clean(tmp_path):
    result = run_check(tmp_path, {
        "dependencies": [
            {"name": "torch", "skip_reason": "Not found on PyPI: 2.11.0+cu128"},
            {"name": "flask", "version": "3.1", "vulns": []},
        ],
    })
    assert result.returncode == 0
    assert "audited 1 packages, skipped 1 packages" in result.stdout
    assert "NOT ASSESSED: torch:" in result.stdout
    assert "INCOMPLETE" in result.stdout
    assert "No known vulnerabilities in **assessed** dependencies." in (tmp_path / "summary.md").read_text()
    assert "**NOT ASSESSED**: torch" in (tmp_path / "summary.md").read_text()


def test_known_findings_and_unassessed_packages_are_both_visible(tmp_path):
    result = run_check(tmp_path, {
        "dependencies": [
            {"name": "torch", "skip_reason": "Local CUDA wheel"},
            {"name": "flask", "version": "3.1", "vulns": [{"id": "CVE-fixture"}]},
        ],
    })
    assert result.returncode == 0
    assert "1 known vulnerability records" in result.stdout
    assert "INCOMPLETE" in result.stdout


def test_vulnerability_aliases_merge_transitively_across_packages(tmp_path):
    result = run_check(tmp_path, {"dependencies": [
        {"name": "first", "version": "1", "vulns": [
            {"id": "GHSA-first", "aliases": ["CVE-shared"]},
            {"id": "PYSEC-first", "aliases": ["GHSA-other"]},
            {"id": "CVE-shared", "aliases": ["GHSA-other"]},
        ]},
        {"name": "second", "version": "2", "vulns": [{"id": "GHSA-other"}]},
    ]})
    assert result.returncode == 0, result.stderr
    assert "audited 2 packages, skipped 0 packages; 1 known vulnerability" in result.stdout
    assert "first: 1" in result.stdout


@pytest.mark.parametrize("data", [
    None, "", "{", "[]", "{}", {"dependencies": []},
    {"dependencies": [{"name": "example"}]},
    {"dependencies": [{"name": "example", "vulns": ["bad"]}]},
    {"dependencies": [{"name": "torch", "skip_reason": "unknown"}]},
    {"dependencies": [{"name": "torch", "skip_reason": ""},
                      {"name": "flask", "vulns": []}]},
    {"dependencies": [{"name": "torch", "skip_reason": "unknown", "vulns": []},
                      {"name": "flask", "vulns": []}]},
    {"dependencies": [{"name": "flask", "vulns": []},
                      {"name": "flask", "vulns": []}]},
    {"dependencies": [{"name": "flask", "vulns": [{"id": "CVE-example", "aliases": "not-a-list"}]}]},
    {"dependencies": [{"name": "flask", "vulns": [{"id": "CVE-example", "aliases": [""]}]}]},
])
def test_missing_or_invalid_evidence_fails_closed(tmp_path, data):
    result = run_check(tmp_path, data)
    assert result.returncode == 2
    assert "audit evidence unavailable/invalid" in result.stderr
