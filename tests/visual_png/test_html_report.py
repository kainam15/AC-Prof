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


def test_html_renders_pass_and_failure_with_embedded_pngs(tmp_path: Path) -> None:
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
    assert "<strong>2</strong>" in document
    assert "<strong>1</strong><span>Matched</span>" in document
    assert "<strong>1</strong><span>Changed / unavailable</span>" in document
    assert 'data-state="passed"' in document
    assert 'data-state="failed"' in document
    assert 'data-filter="failed"' in document
    assert "Historical baseline" in document
    assert "Current capture" in document
    assert "Highlighted differences" in document
    assert document.count("data:image/png;base64,") == 5
    assert "&lt;script&gt;alert" in document
    assert "<script>alert" not in document
    assert '<div class="placeholder">No significant differences</div>' in document
    assert "http://" not in document
    assert "https://" not in document
    assert originals == {name: (baselines / f"{name}.png").read_bytes() for name in originals}


def test_missing_comparison_produces_explicit_unavailable_report(tmp_path: Path) -> None:
    actual = tmp_path / "not-run"
    output = tmp_path / "export" / "report.html"
    document = render_report(actual=actual, output=output).read_text(encoding="utf-8")
    assert "Comparison not available." in document
    assert "no pass/fail result" in document
    assert 'data-state="passed"' not in document
    assert output.is_file()


def test_missing_evidence_has_placeholders(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    _results(actual, [{"scene": "form", "status": "missing-capture", "details": "not found"}])
    document = render_report(actual=actual, baselines=tmp_path / "baseline").read_text()
    assert "Baseline not available" in document
    assert "Capture not available" in document
    assert "Difference image not available" in document
    assert "missing-capture" in document
    assert "Not compared" in document


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
