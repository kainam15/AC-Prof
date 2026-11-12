"""The stable CI gate must not mistake skipped or missing evidence for success."""

from __future__ import annotations

import json

import pytest

from scripts.check_ci_gate import FULL_JOBS, check_gate, main


def jobs(*, full: str, readme: str = "false", metric: str = "false") -> dict:
    names = ("changes", "docs-check", "metric-docs", *FULL_JOBS)
    result = {name: {"result": "success"} for name in names}
    result["changes"]["outputs"] = {"full": full, "readme": readme, "metric": metric}
    return result


@pytest.mark.parametrize(("readme", "metric"), [
    ("false", "false"), ("true", "false"), ("false", "true"), ("true", "true"),
])
def test_docs_scope_requires_only_applicable_jobs(readme, metric):
    result = jobs(full="false", readme=readme, metric=metric)
    for name in FULL_JOBS:
        result[name]["result"] = "skipped"
    result["metric-docs"]["result"] = "skipped"
    if readme == "true":
        result["wheel"]["result"] = "success"
    if metric == "true":
        result["metric-docs"]["result"] = "success"
    assert check_gate(result) == []


def test_docs_scope_rejects_required_special_job_failure():
    result = jobs(full="false", readme="true")
    result["wheel"]["result"] = "failure"
    assert "wheel" in check_gate(result)[0]


@pytest.mark.parametrize("failed_job", ["changes", "docs-check", *FULL_JOBS])
@pytest.mark.parametrize("failure_status", ["failure", "cancelled", "skipped", None])
def test_full_scope_rejects_any_incomplete_job(failed_job, failure_status):
    result = jobs(full="true")
    result[failed_job]["result"] = failure_status
    assert check_gate(result) == [f"{failed_job}: expected success, got {failure_status!r}"]


def test_full_scope_requires_all_jobs():
    result = jobs(full="true")
    assert check_gate(result) == []
    del result["host-summary"]
    assert check_gate(result) == ["host-summary: expected success, got 'missing'"]


@pytest.mark.parametrize("invalid", [
    None, {}, {"changes": {"result": "success"}}, jobs(full="unknown"),
    jobs(full="false", metric="unknown"),
])
def test_missing_or_invalid_change_scope_fails_closed(invalid):
    assert check_gate(invalid)


def test_cli_returns_failure_on_missing_report(monkeypatch):
    monkeypatch.delenv("CI_NEEDS_JSON", raising=False)
    assert main() == 1


def test_cli_accepts_full_success(monkeypatch):
    monkeypatch.setenv("CI_NEEDS_JSON", json.dumps(jobs(full="true")))
    assert main() == 0
