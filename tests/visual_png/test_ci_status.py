"""Same-run CI pass/fail claims require validated SVG, PNG and Pilot evidence."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from scripts.render_tui_png_report import render_report
from scripts.tui_ci_status import ROOT, load_status, main, record_status
from tests.visual_png.scenarios import SCENARIOS
from tests.visual_png.test_journey_report import make_synthetic_manifest


@pytest.fixture(autouse=True)
def _isolate_synthetic_ci_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests create fake run IDs; only the explicit CI-context test opts into CI matching."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)


SHA = "a" * 40
REPO = "kainam15/AC-Prof"
RUN = "123456"


def _fixtures(actual: Path) -> None:
    actual.mkdir(parents=True, exist_ok=True)
    repo = actual / "repo"
    (repo / "tests/visual").mkdir(parents=True)
    (repo / "tests/visual/pilot_cases.txt").write_text(
        "tests/tui/workflows/test_journey.py::test_launch\n"
        "tests/tui/widgets/test_settings.py::TestSettings::test_save\n",
        encoding="utf-8",
    )
    (actual / "svg-tests.xml").write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="tests.visual.test_settings" name="test_settings_page"/>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    (actual / "pilot-tests.xml").write_text(
        '<testsuite tests="3" failures="0" errors="0" skipped="0">'
        '<testcase classname="tests.tui.workflows.test_journey" name="test_launch[en]"/>'
        '<testcase classname="tests.tui.workflows.test_journey" name="test_launch[zh]"/>'
        '<testcase classname="tests.tui.widgets.test_settings.TestSettings" name="test_save"/>'
        "</testsuite>",
        encoding="utf-8",
    )
    baselines = repo / "tests/visual_png/__snapshots__"
    baselines.mkdir(parents=True)
    rows = []
    for name in SCENARIOS:
        Image.new("RGB", (4, 4), "white").save(actual / f"{name}.png")
        (baselines / f"{name}.png").hardlink_to(actual / f"{name}.png")
        rows.append({"scene": name, "status": "passed"})
    (actual / "report.json").write_text(json.dumps(rows), encoding="utf-8")
    make_synthetic_manifest(actual)


def _record(actual: Path, **changes) -> dict:
    outcomes = {"svg": "success", "png": "success", "pilot": "success"}
    outcomes.update(changes)
    return record_status(
        actual=actual,
        root=actual / "repo",
        outcomes=outcomes,
        commit=SHA,
        repository=REPO,
        run_id=RUN,
        run_attempt="1",
        event="push",
    )


def _write(actual: Path, value: dict) -> None:
    (actual / "ci-status.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_all_passed_requires_independent_same_run_evidence(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    status = _record(tmp_path)
    assert {name: row["status"] for name, row in status["checks"].items()} == {
        "svg": "passed",
        "png": "passed",
        "pilot": "passed",
    }
    assert {name: row["count"] for name, row in status["checks"].items()} == {
        "svg": 1,
        "png": len(SCENARIOS),
        "pilot": 21,
    }
    pilot_detail = status["checks"]["pilot"]["detail"]
    assert "3 项断言用例通过" in pilot_detail
    assert "21 帧诊断截图" in pilot_detail
    assert "真实模型采集未执行" in pilot_detail
    _write(tmp_path, status)
    document = render_report(actual=tmp_path).read_text(encoding="utf-8")
    assert "本次 TUI CI 验收 · 3/3 项通过" in document
    for name in ("svg", "png", "pilot"):
        assert f'data-ci-check="{name}" data-ci-status="passed"' in document
    assert "GitHub Actions 本次运行" in document
    assert "/actions/runs/123456/attempts/1" in document
    assert SHA[:12] in document


@pytest.mark.parametrize(
    ("check", "missing"),
    [
        ("svg", "svg-tests.xml"),
        ("png", "report.json"),
        ("pilot", "journey/manifest.json"),
        ("pilot", "pilot-tests.xml"),
    ],
)
def test_missing_evidence_refuses_to_claim_a_success(
    tmp_path: Path, check: str, missing: str
) -> None:
    _fixtures(tmp_path)
    (tmp_path / missing).unlink()
    result = _record(tmp_path)
    assert result["checks"][check]["status"] == "failed"
    assert result["checks"][check]["count"] == 0
    assert sum(row["status"] == "passed" for row in result["checks"].values()) == 2


@pytest.mark.parametrize("check", ["svg", "pilot"])
@pytest.mark.parametrize(
    "defect",
    [
        "summary-only",
        "skipped",
        "failure",
        "error",
        "count",
        "wrong-module",
        "missing-name",
        "duplicate",
        "malformed",
        "unreported-skip",
    ],
)
def test_junit_requires_complete_passing_case_identities(
    tmp_path: Path, check: str, defect: str
) -> None:
    _fixtures(tmp_path)
    source = tmp_path / f"{check}-tests.xml"
    doc = ET.parse(source)
    suite = next(doc.iter("testsuite"))
    case = suite.find("testcase")
    assert case is not None
    if defect == "summary-only":
        for child in suite.findall("testcase"):
            suite.remove(child)
    elif defect in {"skipped", "failure", "error"}:
        key = {"skipped": "skipped", "failure": "failures", "error": "errors"}[defect]
        suite.set(key, "1")
    elif defect == "unreported-skip":
        ET.SubElement(case, "skipped")
    elif defect == "count":
        suite.set("tests", "999")
    elif defect == "wrong-module":
        case.set("classname", "tests.cli.test_unrelated")
    elif defect == "missing-name":
        del case.attrib["name"]
    elif defect == "duplicate":
        suite.append(deepcopy(case))
        suite.set("tests", str(len(suite.findall("testcase"))))
    doc.write(source)
    if defect == "malformed":
        source.write_text("<incomplete")
    result = _record(tmp_path)["checks"][check]
    assert result["status"] == "failed"
    assert result["count"] == 0


def test_pilot_requires_every_selected_function(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    source = tmp_path / "pilot-tests.xml"
    doc = ET.parse(source)
    suite = doc.getroot()
    suite.remove(suite.findall("testcase")[-1])
    suite.set("tests", "2")
    doc.write(source)
    assert _record(tmp_path)["checks"]["pilot"]["status"] == "failed"


def test_successful_step_with_missing_evidence_writes_status_and_exits_nonzero(
    tmp_path, monkeypatch
):
    _fixtures(tmp_path)
    (tmp_path / "pilot-tests.xml").unlink()
    output = tmp_path / "ci-status.json"
    for name, value in {
        "GITHUB_SHA": SHA,
        "GITHUB_REPOSITORY": REPO,
        "GITHUB_RUN_ID": RUN,
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_EVENT_NAME": "push",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("scripts.tui_ci_status.ROOT", tmp_path / "repo")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tui_ci_status",
            "--actual",
            str(tmp_path),
            "--output",
            str(output),
            "--svg-outcome",
            "success",
            "--png-outcome",
            "success",
            "--pilot-outcome",
            "success",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert load_status(output)["checks"]["pilot"]["status"] == "failed"


def test_failed_or_skipped_action_step_is_not_green_even_with_good_artifacts(
    tmp_path: Path,
) -> None:
    _fixtures(tmp_path)
    status = _record(tmp_path, svg="failure", png="skipped", pilot="cancelled")
    assert [status["checks"][name]["status"] for name in ("svg", "png", "pilot")] == [
        "failed",
        "not_run",
        "failed",
    ]
    _write(tmp_path, status)
    report = render_report(actual=tmp_path).read_text()
    assert "本次 TUI CI 验收 · 0/3 项通过" in report
    assert 'data-ci-check="png" data-ci-status="not_run"' in report


def test_incomplete_or_failed_png_and_svg_files_are_not_a_pass(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    payload = json.loads((tmp_path / "report.json").read_text())
    payload.pop()
    (tmp_path / "report.json").write_text(json.dumps(payload))
    (tmp_path / "svg-tests.xml").write_text(
        '<testsuites><testsuite tests="21" failures="1" errors="0" skipped="0" /></testsuites>'
    )
    outcome = _record(tmp_path)
    assert outcome["checks"]["svg"]["status"] == "failed"
    assert outcome["checks"]["png"]["status"] == "failed"
    assert outcome["checks"]["pilot"]["status"] == "passed"


def test_invalid_ci_status_does_not_leak_a_false_green_indicator(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    original = _record(tmp_path)
    for corruption in ("bad-status", "bad-event", "bad-count"):
        record = deepcopy(original)
        if corruption == "bad-status":
            record["checks"]["svg"]["status"] = ["passed"]
        elif corruption == "bad-event":
            record["event"] = {"injected": "object"}
        else:
            record["checks"]["svg"]["count"] = 0
        _write(tmp_path, record)
        assert load_status(tmp_path / "ci-status.json", environ={}) is None
        page = render_report(actual=tmp_path).read_text()
        assert "本次 TUI CI 验收 · 未核验" in page
        assert 'data-ci-check="svg" data-ci-status="not_run"' in page


def test_ci_renderer_refuses_stale_commit_or_different_run(tmp_path: Path, monkeypatch) -> None:
    _fixtures(tmp_path)
    record = _record(tmp_path)
    _write(tmp_path, record)
    env = {
        "GITHUB_ACTIONS": "true",
        "GITHUB_SHA": SHA,
        "GITHUB_RUN_ID": RUN,
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_REPOSITORY": REPO,
    }
    assert load_status(tmp_path / "ci-status.json", environ=env) is not None
    assert load_status(tmp_path / "ci-status.json", environ={**env, "GITHUB_RUN_ATTEMPT": "2"}) is None
    env["GITHUB_SHA"] = "b" * 40
    assert load_status(tmp_path / "ci-status.json", environ=env) is None
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    monkeypatch.setenv("GITHUB_RUN_ID", RUN)
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    page = render_report(actual=tmp_path).read_text()
    assert "本次 TUI CI 验收 · 未核验" in page


def test_no_record_in_local_report_remains_explicitly_unverified(tmp_path: Path) -> None:
    page = render_report(actual=tmp_path).read_text()
    assert "未找到与本次 CI 对应的状态记录" in page
    assert "本次 TUI CI 验收 · 未核验" in page
    assert "/3 项通过" not in page
    assert all(
        f'data-ci-check="{name}" data-ci-status="not_run"' in page
        for name in ("svg", "png", "pilot")
    )


def test_workflow_uses_same_run_evidence_before_html_and_retains_private_pr_reports() -> None:
    import yaml

    workflow = yaml.load(
        (ROOT / ".github/workflows/tui-png.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert "scripts/tui_ci_status.py" in workflow["on"]["push"]["paths"]
    status_step = next(
        step
        for step in workflow["jobs"]["png-visual"]["steps"]
        if step.get("name") == "Record same-run SVG, PNG and Pilot CI statuses"
    )
    assert "python3 -m scripts.tui_ci_status" in status_step["run"]
    job = workflow["jobs"]["png-visual"]
    steps = job["steps"]
    by_id = {step["id"]: (index, step) for index, step in enumerate(steps) if "id" in step}
    for key in (
        "install_deps",
        "svg_regression",
        "tooling",
        "terminal_smoke",
        "png_regression",
        "pilot_regression",
    ):
        assert key in by_id
    assert by_id["svg_regression"][0] < by_id["tooling"][0]
    assert 0 < int(by_id["tooling"][1]["timeout-minutes"]) < int(job["timeout-minutes"])
    for smoke in ("terminal_smoke", "svg_preview_smoke"):
        assert "steps.tooling.outcome == 'success'" in by_id[smoke][1]["if"]
    assert by_id["png_regression"][0] < by_id["pilot_regression"][0]
    assert "tests/visual" in by_id["svg_regression"][1]["run"]
    assert "svg-tests.xml" in by_id["svg_regression"][1]["run"]
    assert "@tests/visual/pilot_cases.txt" in by_id["pilot_regression"][1]["run"]
    assert "pilot-tests.xml" in by_id["pilot_regression"][1]["run"]
    assert "--require-no-skips" in by_id["pilot_regression"][1]["run"]
    capture = by_id["pilot_regression"][1]["run"]
    assert capture.index("scripts/capture_tui_journey.py --png-preview") < capture.index(
        "python3 -m scripts.tui_flow_catalog"
    )
    summary = next(
        index
        for index, step in enumerate(steps)
        if step.get("name", "").startswith("Record same-run")
    )
    renderer = next(
        index
        for index, step in enumerate(steps)
        if step.get("name", "").startswith("Render standalone")
    )
    assert by_id["pilot_regression"][0] < summary < renderer
    artifacts = next(step for step in steps if step.get("name", "").startswith("Upload HTML"))
    assert "ci-status.json" in artifacts["with"]["path"]
    assert "svg-tests.xml" in artifacts["with"]["path"]
    assert "pilot-tests.xml" in artifacts["with"]["path"]
    deploy = next(step for step in steps if step.get("id") == "prepare_pages")
    assert "refs/heads/main" in deploy["if"]
    assert "pull_request" in deploy["if"]
