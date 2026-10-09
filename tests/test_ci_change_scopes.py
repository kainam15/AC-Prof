"""The CI path-filter policy and expensive-job guards must stay in sync."""

from fnmatch import fnmatchcase
from pathlib import Path

import pytest
import yaml

CI = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
HEAVY = (
    "lint", "types", "contracts", "host", "host-summary",
    "report-browser", "tui-snapshots", "onnx-cpu", "runtime",
)


@pytest.fixture(scope="module")
def workflow():
    return yaml.load(CI.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


@pytest.fixture(scope="module")
def filters(workflow):
    steps = workflow["jobs"]["changes"]["steps"]
    action = next(step for step in steps if step.get("id") == "paths")
    assert action["with"]["predicate-quantifier"] == "some-with-excludes"
    return yaml.safe_load(action["with"]["filters"])


def matches(patterns: list[str], changed_path: str) -> bool:
    # Policy examples here deliberately use basic globs; the Action uses picomatch.
    positives = [pattern for pattern in patterns if not pattern.startswith("!")]
    negatives = [pattern[1:] for pattern in patterns if pattern.startswith("!")]
    return (any(fnmatchcase(changed_path, pattern) for pattern in positives)
            and not any(fnmatchcase(changed_path, pattern) for pattern in negatives))


@pytest.mark.parametrize(("file", "full", "readme", "metric"), [
    ("docs/README.md", False, False, False),
    ("docs/results/metric_reference.md", False, False, True),
    ("README.md", False, True, False),
    ("AGENTS.md", False, False, False),
    ("docs/AGENTS.md", False, False, False),
    (".agents/skills/acprof-ci-triage/SKILL.md", False, False, False),
    (".agents/skills/example/agents/openai.yaml", False, False, False),
    (".agents/skills/example/scripts/rebuild.py", True, False, False),
    ("acprof/cli/main.py", True, False, False),
    ("tests/test_architecture.py", True, False, False),
    ("requirements/dev.lock", True, False, False),
    ("pyproject.toml", True, False, False),
    ("scripts/check_docs.py", True, False, False),
    (".github/workflows/ci.yml", True, False, False),
    ("some-unknown-file.md", True, False, False),
])
def test_changed_path_classification(filters, file, full, readme, metric):
    assert matches(filters["full"], file) is full
    assert matches(filters["readme"], file) is readme
    assert matches(filters["metric"], file) is metric
    assert matches(filters["documentation"], file) is (not full)


def test_mixed_change_requires_full_ci(filters):
    paths = ["docs/README.md", "acprof/cli/main.py"]
    assert any(matches(filters["full"], path) for path in paths)


def test_manual_and_release_reusable_workflows_force_full(workflow):
    assert "workflow_dispatch" in workflow["on"]
    assert "workflow_call" in workflow["on"]
    assert workflow["on"]["workflow_call"]["inputs"]["force_full"]["default"] == "true"
    expr = workflow["jobs"]["changes"]["outputs"]["full"]
    assert "github.event_name != 'push'" in expr
    assert "github.event_name != 'pull_request'" in expr
    assert "steps.paths.outputs.documentation != 'true'" in expr
    assert "inputs.force_full" in expr
    assert "refs/tags/" in expr


def test_host_summary_skips_when_host_is_not_requested(workflow):
    summary = workflow["jobs"]["host-summary"]
    assert set(summary["needs"]) == {"changes", "host"}
    assert "always()" in summary["if"]
    assert "needs.changes.outputs.full == 'true'" in summary["if"]
    assert "needs.changes.result == 'success'" in summary["if"]


def test_job_policy(workflow):
    jobs = workflow["jobs"]
    assert not jobs["docs-check"].get("needs")
    assert "scripts/check_docs.py" in str(jobs["docs-check"]["steps"])
    for job in HEAVY:
        assert "needs.changes.outputs.full == 'true'" in jobs[job]["if"]
    assert "needs.changes.outputs.readme == 'true'" in jobs["wheel"]["if"]
    assert "needs.changes.outputs.metric == 'true'" in jobs["metric-docs"]["if"]
    assert "scripts/render_metric_reference.py --check" in str(jobs["metric-docs"]["steps"])
