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


def test_chrome_capture_accepts_png_before_browser_exits(tmp_path):
    """Real process group: a usable screenshot beats a stuck Chrome shutdown."""
    import sys
    import time

    from scripts.tui_svg_preview_png import _chrome_capture

    reference = tmp_path / "valid.png"
    Image.effect_noise((120, 100), 100).convert("RGB").save(reference)
    assert reference.stat().st_size >= 1024
    fake_chrome = tmp_path / "fake-chrome"
    fake_chrome.write_text(
        f"#!{sys.executable}\n"
        "import shutil, sys, time\n"
        "destination = next(arg.split('=', 1)[1] for arg in sys.argv "
        "if arg.startswith('--screenshot='))\n"
        f"shutil.copyfile({str(reference)!r}, destination)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    fake_chrome.chmod(0o755)
    html = tmp_path / "scene.html"
    html.write_text("<html></html>", encoding="utf-8")
    start = time.monotonic()
    _chrome_capture(str(fake_chrome), html, tmp_path / "scene.png", width=120, height=100)
    assert time.monotonic() - start < 5
    assert (tmp_path / "scene.png").is_file()


def test_chrome_capture_timeout_reports_stderr_and_reaps(tmp_path, monkeypatch):
    import sys

    from scripts import tui_svg_preview_png as preview

    monkeypatch.setattr(preview, "CHROME_CAPTURE_TIMEOUT_SECONDS", 0.4)
    fake_chrome = tmp_path / "fake-chrome-hang"
    fake_chrome.write_text(
        f"#!{sys.executable}\n"
        "import sys, time\n"
        "sys.stderr.write('CHROME_TEST_HANG\\n')\n"
        "sys.stderr.flush()\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    fake_chrome.chmod(0o755)
    html = tmp_path / "scene.html"
    html.write_text("<html></html>", encoding="utf-8")
    with pytest.raises(RuntimeError, match="CHROME_TEST_HANG"):
        preview._chrome_capture(
            str(fake_chrome), html, tmp_path / "scene.png", width=120, height=100
        )
    assert (tmp_path / "scene.stderr.log").is_file()


def test_failed_capture_preserves_only_fixture_evidence(tmp_path, monkeypatch):
    from scripts import tui_svg_preview_png as preview

    monkeypatch.setattr(preview, "FAILURES_OUTPUT", tmp_path / "evidence")
    html = tmp_path / "scene-0.html"
    html.write_text("<html>test-only fixture</html>", encoding="utf-8")
    html.with_suffix(".stderr.log").write_text("browser stderr", encoding="utf-8")
    evidence = preview._retain_failure(html, tmp_path / "missing.png", index=0, scene=SAMPLE)
    assert (evidence / "scene-0.html").read_text(encoding="utf-8").startswith("<html>")
    assert (evidence / "scene-0.stderr.log").read_text(encoding="utf-8") == "browser stderr"
    assert (evidence / "source.txt").read_text(encoding="utf-8").strip() == SAMPLE
    assert not (evidence / "missing.png").exists()


def test_svg_preview_ci_runs_one_actual_chrome_smoke_before_batch():
    workflow = (ROOT / ".github/workflows/tui-png.yml").read_text(encoding="utf-8")
    smoke = workflow.index("Smoke-test one reviewed CJK SVG preview")
    full = workflow.index("Render CJK previews of reviewed SVG baselines")
    assert smoke < full
    assert "scripts/tui_svg_preview_png.py --smoke" in workflow
    assert "svg-preview-failures/**/*" in workflow
