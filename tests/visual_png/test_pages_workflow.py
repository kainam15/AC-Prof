"""Contract tests for the public TUI PNG report publishing workflow."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "tui-png.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _step(workflow: dict, name: str) -> dict:
    return next(
        step for step in workflow["jobs"]["png-visual"]["steps"] if step.get("name") == name
    )


def test_pages_deployment_is_main_only_and_does_not_require_green_pngs(workflow: dict) -> None:
    build = workflow["jobs"]["png-visual"]
    deploy = workflow["jobs"]["deploy-pages"]
    for step in (_step(workflow, "Prepare GitHub Pages report"), deploy):
        condition = step["if"]
        assert "github.ref == 'refs/heads/main'" in condition
        assert "github.event_name != 'pull_request'" in condition
        assert "!cancelled()" in condition
        assert "success()" not in condition
    assert deploy["needs"] == "png-visual"
    assert deploy["if"].find("needs.png-visual.outputs.pages_ready == 'true'") > 0
    assert build["outputs"]["pages_ready"] == "${{ steps.prepare_pages.outputs.ready }}"
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}
    assert deploy["environment"]["name"] == "github-pages"
    assert workflow["concurrency"]["cancel-in-progress"] is False


def test_pages_uses_pinned_actions_and_preserves_run_artifact(workflow: dict) -> None:
    steps = workflow["jobs"]["png-visual"]["steps"]
    report_step = _step(workflow, "Render standalone HTML comparison report")
    artifact_step = _step(workflow, "Upload HTML, actual, expected, diff and summary for review")
    prepare_step = _step(workflow, "Prepare GitHub Pages report")
    upload_step = _step(workflow, "Upload GitHub Pages artifact")
    assert steps.index(report_step) < steps.index(artifact_step) < steps.index(prepare_step)
    assert report_step["if"] == "always()"
    assert artifact_step["if"] == "always()"
    assert "report.html" in artifact_step["with"]["path"]
    assert upload_step["if"].find("steps.prepare_pages.outputs.ready == 'true'") > 0
    for step in (upload_step, workflow["jobs"]["deploy-pages"]["steps"][0]):
        assert re.fullmatch(
            r"actions/(?:upload-pages-artifact|deploy-pages)@[0-9a-f]{40}", step["uses"]
        )


@pytest.mark.skipif(shutil.which("bash") is None, reason="GitHub Actions staging uses bash")
@pytest.mark.parametrize("report", [None, "", "<h1>visual regression</h1>"])
def test_pages_stages_only_nonempty_html(
    workflow: dict, tmp_path: Path, report: str | None
) -> None:
    script = _step(workflow, "Prepare GitHub Pages report")["run"]
    html = tmp_path / "internal-testing" / "shared" / "tui-png" / "report.html"
    if report is not None:
        html.parent.mkdir(parents=True)
        html.write_text(report, encoding="utf-8")
    output = tmp_path / "job-output.txt"
    output.touch()
    env = dict(os.environ, RUNNER_TEMP=str(tmp_path / "runner"), GITHUB_OUTPUT=str(output))

    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script], cwd=tmp_path, env=env, check=True
    )

    published = tmp_path / "runner" / "tui-png-pages" / "index.html"
    if report:
        assert published.read_text(encoding="utf-8") == report
        assert output.read_text(encoding="utf-8") == "ready=true\n"
    else:
        assert not published.exists()
        assert output.read_text(encoding="utf-8") == ""
