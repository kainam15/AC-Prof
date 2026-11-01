"""SVG Golden previews are read-only diagnostics, never additional PNG baselines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.render_tui_png_report import BASELINES, ROOT, render_report
from scripts.tui_svg_preview_png import (
    FONT_FAMILY,
    _prepare_svg,
    _preview_name,
    load_preview_index,
)

SAMPLE = "tests/visual/__snapshots__/test_snapshots/test_fixed_scenes[zh-size0-form].svg"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(actual: Path, *, source: str = SAMPLE) -> tuple[Path, Path]:
    folder = actual / "svg-preview"
    target = folder / _preview_name(source)
    target.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), "white").save(target)
    manifest = folder / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "svg-cjk-png-preview",
                "entries": [
                    {
                        "source": source,
                        "source_sha256": _digest(ROOT / source),
                        "preview": _preview_name(source),
                        "preview_sha256": _digest(target),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, target


def test_preview_only_changes_derived_svg_content():
    source = ROOT / SAMPLE
    before = _digest(source)
    preview_svg, width, height = _prepare_svg(source.read_text(encoding="utf-8"))
    assert width > 500
    assert height > 300
    assert f'font-family: "{FONT_FAMILY}", monospace;' in preview_svg
    assert "font-variant-east-asian: full-width;" not in preview_svg
    assert 'lengthAdjust="spacingAndGlyphs"' in preview_svg
    assert "cdnjs.cloudflare.com" not in preview_svg
    assert _digest(source) == before


def test_html_prefers_png_preview_without_changing_golden_counts(tmp_path):
    _write_manifest(tmp_path)
    report = render_report(actual=tmp_path, baselines=BASELINES).read_text(encoding="utf-8")
    assert "SVG 基准的 CJK PNG 预览（非 Golden PNG）" in report
    assert "查看原始 SVG Golden（诊断）" in report
    assert "data:image/png;base64," in report
    assert "data:image/svg+xml;base64," in report
    assert "Comparison not available." in report
    assert "SVG 基准图库（21 张）" in report


def test_absent_previews_keep_original_svg_behavior(tmp_path):
    assert load_preview_index(root=ROOT, output=tmp_path / "svg-preview") == {}
    report = render_report(actual=tmp_path).read_text(encoding="utf-8")
    assert "已审阅 SVG 基准（非本次截图）" in report
    assert "SVG 基准的 CJK PNG 预览" not in report


def test_stale_preview_is_rejected(tmp_path):
    manifest, _ = _write_manifest(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["entries"][0]["source_sha256"] = "0" * 64
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid/stale SVG preview"):
        render_report(actual=tmp_path)


def test_preview_paths_must_be_derived_from_tracked_sources(tmp_path):
    manifest, _ = _write_manifest(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["entries"][0]["preview"] = "../../untrusted.png"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe SVG preview path"):
        render_report(actual=tmp_path)


def test_untrusted_preview_source_is_rejected():
    with pytest.raises(ValueError, match="unsafe SVG preview source"):
        _preview_name("../../private.svg")


def test_symlinked_preview_does_not_embed_external_files(tmp_path):
    _, target = _write_manifest(tmp_path)
    outside = tmp_path / "unrelated.png"
    target.rename(outside)
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="invalid/stale SVG preview"):
        load_preview_index(root=ROOT, output=tmp_path / "svg-preview")
