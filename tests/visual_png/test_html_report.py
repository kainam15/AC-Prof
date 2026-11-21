"""Offline report rendering tests; no VHS, Chrome or terminal emulator required."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.render_tui_png_report import render_report


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 10), "#242833").save(path)


def _results(actual: Path, rows: list[dict[str, object]]) -> None:
    actual.mkdir(parents=True, exist_ok=True)
    (actual / "report.json").write_text(json.dumps(rows), encoding="utf-8")


def test_html_renders_pass_and_failure_with_embedded_pngs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This test checks portable/offline HTML, not the main-branch review flow.
    monkeypatch.delenv("GITHUB_REF", raising=False)
    actual, baselines = tmp_path / "actual", tmp_path / "baselines"
    for name in ("form", "error-modal"):
        _png(actual / f"{name}.png")
        _png(baselines / f"{name}.png")
    _png(actual / "diff" / "error-modal.png")
    _results(
        actual,
        [
            {
                "scene": "form",
                "status": "passed",
                "changed_pixels": 0,
                "allowed_pixels": 120,
                "width": 16,
                "height": 10,
            },
            {
                "scene": "error-modal",
                "status": "visual-regression",
                "changed_pixels": 900,
                "allowed_pixels": 120,
                "details": "<script>alert('x')</script>",
            },
        ],
    )
    originals = {name: (baselines / f"{name}.png").read_bytes() for name in ("form", "error-modal")}

    report_path = render_report(actual=actual, baselines=baselines)
    document = report_path.read_text(encoding="utf-8")

    assert report_path == actual / "report.html"
    assert 'aria-pressed="true">全部 (2)</button>' in document
    assert 'aria-pressed="false">变更或不可用 (1)</button>' in document
    assert 'aria-pressed="false">已匹配 (1)</button>' in document
    assert document.index('data-state="failed" open>') < document.index('data-state="passed">')
    assert 'data-report-view="visual" aria-pressed="true"' in document
    assert 'alt="历史基准"' in document
    assert 'alt="当前截图"' in document
    # Original PNG comparisons remain intact; flow cards also embed preview images.
    assert document.count("data:image/png;base64,") >= 5
    assert 'data-view-panel="workflow"' in document
    assert "&lt;script&gt;alert" in document
    assert "<script>alert" not in document
    assert document.count("<figcaption>差异图</figcaption>") == 1
    assert document.count('alt="差异图"') == 1
    assert "http://" not in document
    assert "https://" not in document
    assert originals == {name: (baselines / f"{name}.png").read_bytes() for name in originals}


def test_missing_comparison_produces_explicit_unavailable_report(tmp_path: Path) -> None:
    actual = tmp_path / "not-run"
    output = tmp_path / "export" / "report.html"
    document = render_report(actual=actual, output=output).read_text(encoding="utf-8")
    assert "未生成 report.json，无法判断通过或失败" in document
    assert 'data-state="passed"' not in document
    assert "全部 (0)" not in document
    assert output.is_file()


def test_missing_evidence_has_placeholders(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    _results(actual, [{"scene": "form", "status": "missing-capture", "details": "not found"}])
    document = render_report(actual=actual, baselines=tmp_path / "baseline").read_text()
    assert "历史基准不可用" in document
    assert "当前截图不可用" in document
    assert "<figcaption>差异图</figcaption>" not in document
    assert "missing-capture" in document
    assert "未比较" in document


def test_empty_comparison_is_not_a_success(tmp_path: Path) -> None:
    _results(tmp_path, [])
    document = render_report(actual=tmp_path).read_text(encoding="utf-8")
    assert "report.json 中没有场景记录" in document
    assert 'data-state="passed"' not in document
    assert "全部 (0)" not in document


def test_completed_comparison_identifies_missing_diff(tmp_path: Path) -> None:
    _results(tmp_path, [{"scene": "form", "status": "visual-regression"}])
    document = render_report(actual=tmp_path).read_text(encoding="utf-8")
    assert "差异图不可用" in document
    assert "差异像素：—" in document


@pytest.mark.parametrize(
    "payload",
    [
        {"not": "a list"},
        [{"scene": "../../outside", "status": "passed"}],
        [{"scene": "form", "status": None}],
    ],
)
def test_invalid_report_is_rejected(tmp_path: Path, payload: object) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "report.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="report.json|Invalid scene"):
        render_report(actual=actual)
