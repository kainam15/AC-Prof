"""The workflow viewer reuses reviewed snapshots without claiming live execution."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.render_tui_png_report import render_report
from scripts.tui_flow_catalog import CATALOG, ROOT, load_catalog, png_inventory, svg_inventory


def test_flow_catalog_references_existing_public_fixtures_and_pages() -> None:
    catalog = load_catalog()
    steps = catalog["checkpoints"]
    assert len(catalog["pages"]) == 5
    assert {item["id"] for item in steps} >= {
        "configure",
        "prepare",
        "download",
        "measurement",
        "finish",
        "results",
        "recover",
    }
    assert {name for item in steps for name in item["png"]} == set(png_inventory())
    assert {name for item in steps for name in item["svg"]} == set(svg_inventory())
    assert all(item["tests"] for item in steps)


@pytest.mark.parametrize(
    ("field", "unsafe"),
    [
        ("svg", "../../.env.local"),
        ("svg", "tests/visual/__snapshots__/../../private.svg"),
        ("tests", "docs/GOAL.md"),
        ("tests", "tests/tui/../visual/test_snapshots.py"),
        ("png", "nonexistent"),
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
    assert 'data-view-panel="workflow"' in page
    assert 'data-view-panel="visual" hidden' in page
    assert 'data-view-panel="coverage" hidden' in page
    for checkpoint in (
        "configure",
        "task-selection",
        "download",
        "measurement",
        "finish",
        "results",
        "settings",
    ):
        assert f'id="flow-{checkpoint}"' in page
    assert "SVG 基准图库（21 张）" in page
    assert page.count('alt="SVG 历史基准"') == len(svg_inventory())
    assert "data:image/svg+xml;base64," in page
    # Optional confirmations belong before runtime validation in the flow view.
    assert page.index('id="flow-task-selection"') < page.index('id="flow-download"')
    assert page.index('id="flow-download"') < page.index('id="flow-validation"')
    assert "0 changed pixels" in page
    assert "<strong>1</strong><span>Matched</span>" in page


def test_missing_png_comparison_does_not_infer_success(tmp_path: Path) -> None:
    page = render_report(actual=tmp_path).read_text(encoding="utf-8")
    assert "Comparison not available." in page
    assert "not-compared" in page
    assert "测试文件索引不代表本次执行通过" in page
    assert "SVG CI" in page
    assert "data:image/svg+xml;base64," in page


def test_flow_report_never_collects_generated_run_data(tmp_path: Path) -> None:
    # The report index must ignore files alongside a capture output that are
    # not declared as fixed public visual scenes.
    (tmp_path / "private-run.log").write_text(
        "DO_NOT_EXPOSE_AN_ACTUAL_EXPERIMENT", encoding="utf-8"
    )
    page = render_report(actual=tmp_path).read_text(encoding="utf-8")
    assert "DO_NOT_EXPOSE_AN_ACTUAL_EXPERIMENT" not in page
