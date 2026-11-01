"""Offline Golden approval tests: no network, browser, or GitHub credentials."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.approve_tui_png_goldens import (
    ISSUE_MARKER,
    ISSUE_TITLE,
    ApprovalRequest,
    parse_issue,
    stage_goldens,
    validate_run,
)
from scripts.render_tui_png_report import render_report
from scripts.tui_png_review_ui import REVIEW_SCRIPT, ReviewContext

REPO = "kainam15/AC-Prof"
SHA = "a" * 40


def issue_event(
    *,
    user: str = "kainam15",
    scenes: list[str] | None = None,
    run_id: int = 123,
    head_sha: str = SHA,
) -> dict:
    payload = {
        "version": 1,
        "run_id": run_id,
        "head_sha": head_sha,
        "scenes": ["form"] if scenes is None else scenes,
    }
    return {
        "action": "opened",
        "repository": {"full_name": REPO},
        "issue": {
            "number": 25,
            "title": ISSUE_TITLE,
            "user": {"login": user},
            "body": f"{ISSUE_MARKER}\n{json.dumps(payload)}\n-->\n\nApproved!",
        },
    }


def run_metadata() -> dict:
    return {
        "id": 123,
        "path": ".github/workflows/tui-png.yml",
        "head_repository": {"full_name": REPO},
        "head_branch": "main",
        "event": "push",
        "head_sha": SHA,
        "status": "completed",
        "conclusion": "failure",  # A failed visual comparison must still be reviewable.
    }


def png(path: Path, color: str = "#222222") -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 14), color).save(path)
    return path.read_bytes()


def test_owner_approval_parsing_and_failed_run_are_supported() -> None:
    request = parse_issue(issue_event(), REPO)
    assert request == ApprovalRequest(issue_number=25, run_id=123, head_sha=SHA, scenes=("form",))
    validate_run(run_metadata(), request, REPO)


@pytest.mark.parametrize(
    "bad_event",
    [
        lambda e: e["issue"]["user"].update(login="other-user"),
        lambda e: e["issue"].update(title="Normal issue"),
        lambda e: e.update(action="edited"),
        lambda e: e["repository"].update(full_name="someone/else"),
        lambda e: e["issue"].update(body="<script>fake</script>"),
        lambda e: e["issue"].update(number=0),
    ],
)
def test_untrusted_or_unrelated_issues_rejected(bad_event) -> None:
    event = issue_event()
    bad_event(event)
    with pytest.raises(ValueError, match=".+"):
        parse_issue(event, REPO)


@pytest.mark.parametrize(
    "scenes",
    [
        [],
        ["form", "form"],
        ["../../README"],
        ["form.png"],
        ["form", "/tmp/evil"],
        ["form"] * 21,
    ],
)
def test_reject_invalid_scene_selection(scenes: list[str]) -> None:
    with pytest.raises(ValueError, match=".+"):
        parse_issue(issue_event(scenes=scenes), REPO)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("path", ".github/workflows/ci.yml"),
        ("head_branch", "feature"),
        ("event", "pull_request"),
        ("head_sha", "b" * 40),
        ("status", "in_progress"),
        ("id", 124),
        ("head_repository", {"full_name": "someone/fork"}),
    ],
)
def test_reject_unauthorized_ci_provenance(key: str, value: object) -> None:
    run = run_metadata()
    run[key] = value
    with pytest.raises(ValueError, match="Untrusted source"):
        validate_run(run, parse_issue(issue_event(), REPO), REPO)


def test_stage_updates_only_selected_changed_png(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    checkout = tmp_path / "checkout"
    actual = artifact / "internal-testing/shared/tui-png"
    golden = checkout / "tests/visual_png/__snapshots__"
    expected_form = png(golden / "form.png", "#223344")
    expected_error = png(golden / "error-modal.png", "#223344")
    current_form = png(actual / "form.png", "#995533")
    png(actual / "error-modal.png", "#995533")
    (actual / "report.json").write_text(
        json.dumps(
            [
                {"scene": "form", "status": "visual-regression"},
                {"scene": "error-modal", "status": "visual-regression"},
            ]
        ),
        encoding="utf-8",
    )
    changed = stage_goldens(parse_issue(issue_event(), REPO), artifact=artifact, checkout=checkout)
    assert changed == [golden / "form.png"]
    assert (golden / "form.png").read_bytes() == current_form != expected_form
    assert (golden / "error-modal.png").read_bytes() == expected_error


@pytest.mark.parametrize(
    "status",
    ["passed", "missing-capture", "missing-baseline", "invalid-image"],
)
def test_never_approve_unfinished_comparisons(tmp_path: Path, status: str) -> None:
    artifact, checkout = tmp_path / "artifact", tmp_path / "checkout"
    actual = artifact / "internal-testing/shared/tui-png"
    png(actual / "form.png")
    png(checkout / "tests/visual_png/__snapshots__/form.png", "#ababab")
    (actual / "report.json").write_text(json.dumps([{"scene": "form", "status": status}]))
    with pytest.raises(ValueError, match="not a completed"):
        stage_goldens(parse_issue(issue_event(), REPO), artifact=artifact, checkout=checkout)


def test_reject_invalid_candidate_and_same_baseline(tmp_path: Path) -> None:
    artifact, checkout = tmp_path / "artifact", tmp_path / "checkout"
    actual = artifact / "internal-testing/shared/tui-png"
    baseline = checkout / "tests/visual_png/__snapshots__/form.png"
    png(baseline)
    candidate = actual / "form.png"
    candidate.parent.mkdir(parents=True)
    (actual / "report.json").write_text(
        json.dumps([{"scene": "form", "status": "visual-regression"}])
    )
    candidate.write_bytes(b"not a PNG")
    with pytest.raises(ValueError, match="Invalid PNG"):
        stage_goldens(parse_issue(issue_event(), REPO), artifact=artifact, checkout=checkout)
    candidate.write_bytes(baseline.read_bytes())
    with pytest.raises(ValueError, match="would change"):
        stage_goldens(parse_issue(issue_event(), REPO), artifact=artifact, checkout=checkout)


def test_html_review_controls_main_only_and_no_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual, baseline = tmp_path / "actual", tmp_path / "baselines"
    png(actual / "form.png")
    png(actual / "error-modal.png")
    png(baseline / "form.png", "#666666")
    png(baseline / "error-modal.png", "#666666")
    (actual / "report.json").write_text(
        json.dumps(
            [
                {"scene": "form", "status": "visual-regression"},
                {"scene": "error-modal", "status": "missing-capture"},
            ]
        )
    )
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_SHA", SHA)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    document = render_report(actual=actual, baselines=baseline).read_text()
    assert "data-golden-review-panel" in document
    assert 'data-run-id="123"' in document
    assert 'data-golden-scene="form"' in document
    assert 'data-golden-scene="error-modal"' not in document
    assert "Submit new issue" in document
    assert "GH_TOKEN" not in document
    assert "GITHUB_TOKEN" not in document
    monkeypatch.setenv("GITHUB_REF", "refs/pull/1/merge")
    assert (
        "data-golden-review-panel"
        not in render_report(actual=actual, baselines=baseline).read_text()
    )


def test_html_review_disabled_on_missing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in {
        "GITHUB_REPOSITORY": REPO,
        "GITHUB_RUN_ID": "123",
        "GITHUB_SHA": SHA,
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_EVENT_NAME": "push",
    }.items():
        monkeypatch.setenv(key, value)
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "report.json").write_text(
        json.dumps([{"scene": "form", "status": "missing-capture"}])
    )
    assert "data-golden-review-panel" not in render_report(actual=actual).read_text()


def test_review_source_contains_only_github_issue_request_not_write_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "/issues/new" in REVIEW_SCRIPT
    assert "fetch(" not in REVIEW_SCRIPT
    assert "/git/refs" not in REVIEW_SCRIPT
    assert "Bearer " not in REVIEW_SCRIPT
    monkeypatch.delenv("GITHUB_REF", raising=False)
    assert ReviewContext.from_environment() is None


def test_review_workflow_is_owner_gated_and_only_changes_selected_png() -> None:
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[2]
    doc = yaml.safe_load(
        (root / ".github/workflows/tui-png-golden-review.yml").read_text(encoding="utf-8")
    )
    # PyYAML treats 'on' as a boolean in YAML 1.1.
    events = doc.get("on", doc.get(True))
    assert events == {"issues": {"types": ["opened"]}}
    job = doc["jobs"]["update-golden"]
    assert "github.event.issue.user.login == github.repository_owner" in job["if"]
    assert "pull_request_target" not in str(doc)
    assert job["permissions"]["contents"] == "write"
    assert job["permissions"]["pull-requests"] == "write"
    assert doc["permissions"]["contents"] == "read"
    assert "python3 scripts/approve_tui_png_goldens.py" in job["steps"][-1]["run"]


def test_browser_review_request_matches_backend_contract() -> None:
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("Node unavailable for static browser-control simulation")
    harness = r"""
const input = {checked:true, dataset:{goldenScene:'form'}, addEventListener() {}};
const submit = {disabled:true, listeners:{}, addEventListener(k, cb){this.listeners[k]=cb}};
const noop = {addEventListener(){}};
const count = {textContent:''};
const controls = {
  '[data-golden-submit]': submit,
  '[data-golden-count]': count,
  '[data-golden-select-all]': noop,
  '[data-golden-clear]': noop
};
const panel = {dataset:{repository:'kainam15/AC-Prof', runId:'123', headSha:'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'}, querySelector(k){return controls[k]}};
const document = {
  querySelector(k){return k === '[data-golden-review-panel]' ? panel : null},
  querySelectorAll(){return [input]}
};
let link = null;
const window = {open(url){link=url}};
"""
    harness += (
        REVIEW_SCRIPT
        + "\nsubmit.listeners.click();\nconsole.log(new URL(link).searchParams.get('body'));\n"
    )
    result = subprocess.run(["node", "-e", harness], check=True, text=True, capture_output=True)
    body = result.stdout.strip()
    event = issue_event()
    event["issue"]["body"] = body
    assert parse_issue(event, REPO).scenes == ("form",)
