"""The workflow viewer reuses reviewed snapshots without claiming live execution."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from textual.widgets import Button

from scripts.render_tui_png_report import render_report
from scripts.tui_flow_catalog import (
    CATALOG,
    ROOT,
    check_visual_coverage,
    load_catalog,
    png_inventory,
    svg_inventory,
)
from tests.support.tui_fixtures import AcprofTui


@pytest.mark.tui
async def test_flow_catalog_matches_visible_tui_pages(tmp_path: Path) -> None:
    catalog = load_catalog()
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        visible_pages = [
            (button.name, button.label.plain)
            for button in app.query("#page-navigation .nav-page").results(Button)
            if button.display
        ]
        assert [(page["id"], page.get("tab_label", page["title"])) for page in catalog["pages"]] == visible_pages


@pytest.mark.parametrize("label", ["⚙", "", 7])
def test_catalog_checks_icon_tab_label_separately_from_page_title(tmp_path, label):
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    settings = next(page for page in data["pages"] if page["id"] == "settings-tab")
    settings["tab_label"] = label
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    if label == "⚙":
        loaded = load_catalog(source=source)
        assert loaded["pages"][-1]["title"] == "应用设置"
        assert loaded["pages"][-1]["tab_label"] == "⚙"
    else:
        with pytest.raises(ValueError, match="page tab label must be nonempty text"):
            load_catalog(source=source)


def test_flow_catalog_references_existing_public_fixtures_and_pages() -> None:
    catalog = load_catalog()
    steps = catalog["checkpoints"]
    by_id = {item["id"]: item for item in steps}
    assert by_id.keys() >= {
        "configure", "launch-review", "prepare", "task-selection", "model-review", "download",
        "validation", "measurement", "finish", "results", "profiler", "recover", "preflight-error",
        "docker-repair", "stop-confirm", "cleanup-incomplete", "images", "settings", "version-management",
    }
    assert set(by_id["validation"]["pilot"]) >= {"validation-running", "validation-passed", "validation-failed"}
    assert set(by_id["finish"]["pilot"]) >= {"finished", "finish-partial", "finish-failed"}
    for checkpoint in ("launch-review", "recover"):
        assert "launch-review" in by_id[checkpoint]["pilot"]
    assert "tests/visual/__snapshots__/test_settings/test_settings_page.svg" in by_id["settings"]["svg"]
    assert {name for item in steps for name in item["png"]} == set(png_inventory())
    assert {name for item in steps for name in item["svg"]} == set(svg_inventory())
    assert all(item["tests"] for item in steps)


def test_coverage_gate_requires_the_missing_branch_even_with_a_success_screenshot():
    catalog = {"checkpoints": [
        {"id": "finish", "png": [], "svg": [], "pilot": ["finished", "finish-failed"]},
        {"id": "configure", "png": ["form"], "svg": []},
    ]}
    steps = [{"id": "finished"}]
    manifest = {"scenarios": [{"steps": steps}]}
    with pytest.raises(ValueError, match="finish: finish-failed"):
        check_visual_coverage(catalog, manifest)
    steps.append({"id": "finish-failed"})
    check_visual_coverage(catalog, manifest)
    catalog["checkpoints"][0].pop("pilot")
    with pytest.raises(ValueError, match="finish: 未登记视觉证据"):
        check_visual_coverage(catalog, manifest)


@pytest.mark.parametrize(
    ("field", "unsafe"),
    [
        ("svg", "../../.env.local"),
        ("svg", "tests/visual/__snapshots__/../../private.svg"),
        ("tests", "docs/GOAL.md"),
        ("tests", "tests/tui/../visual/test_snapshots.py"),
        ("png", "nonexistent"),
        ("pilot", "unknown-step"),
        ("pilot", "finish-failed"),
    ],
)
def test_catalog_rejects_missing_or_unsafe_sources(tmp_path: Path, field: str, unsafe: str) -> None:
    data = deepcopy(json.loads(CATALOG.read_text(encoding="utf-8")))
    data["checkpoints"][0][field] = [unsafe]
    altered = tmp_path / "unsafe.json"
    altered.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid TUI flow catalog"):
        load_catalog(source=altered, root=ROOT)


def test_report_includes_workflow_coverage_all_svgs_and_png_comparison(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text(
        json.dumps([{"scene": "form", "status": "passed", "changed_pixels": 0}]),
        encoding="utf-8",
    )
    output = render_report(actual=tmp_path)
    page = output.read_text(encoding="utf-8")
    assert page.count("data-report-view=") == 3
    assert 'data-view-panel="workflow" hidden' in page
    assert 'data-view-panel="visual">' in page
    assert 'data-view-panel="coverage" hidden' in page
    for checkpoint in (
        "configure",
        "task-selection",
        "download",
        "measurement",
        "finish",
        "results",
        "profiler",
        "settings",
    ):
        assert f'id="flow-{checkpoint}"' in page
    assert f"SVG 基准图库（{len(svg_inventory())} 张）" in page
    assert page.count('alt="SVG 历史基准"') == len(svg_inventory())
    assert "data:image/svg+xml;base64," in page
    # Optional confirmations belong before runtime validation in the flow view.
    assert page.index('id="flow-task-selection"') < page.index('id="flow-download"')
    assert page.index('id="flow-download"') < page.index('id="flow-validation"')
    assert "差异像素：0" in page
    assert "已匹配 (1)" in page


def test_missing_png_comparison_does_not_infer_success(tmp_path: Path) -> None:
    page = render_report(actual=tmp_path).read_text(encoding="utf-8")
    assert "未生成 report.json，无法判断通过或失败" in page
    assert "not-compared" in page
    assert "测试文件索引不代表本次执行通过" in page
    assert "SVG CI" in page
    assert "data:image/svg+xml;base64," in page


def test_report_shows_unavailable_svg_without_embedding_an_unreviewed_copy(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.tui_flow_report.svg_inventory", lambda **kwargs: [])
    report = render_report(actual=tmp_path).read_text(encoding="utf-8")
    assert "缺少已审阅的 SVG 基准" in report
    assert "data:image/svg+xml;base64," not in report
    assert "缺少本次 Pilot 测试 JUnit 结果" in report


def test_flow_report_never_collects_generated_run_data(tmp_path: Path) -> None:
    # The report index must ignore files alongside a capture output that are
    # not declared as fixed public visual scenes.
    (tmp_path / "private-run.log").write_text(
        "DO_NOT_EXPOSE_AN_ACTUAL_EXPERIMENT", encoding="utf-8"
    )
    page = render_report(actual=tmp_path).read_text(encoding="utf-8")
    assert "DO_NOT_EXPOSE_AN_ACTUAL_EXPERIMENT" not in page
