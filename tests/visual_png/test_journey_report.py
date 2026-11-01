"""Synthetic Pilot evidence is strict, opt-in, and distinct from PNG goldens."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from rich.console import Console

from scripts.render_tui_png_report import render_report
from scripts.tui_journey_contract import CHECKPOINTS, SCENARIOS, load_manifest
from scripts.tui_pilot_font import use_cjk_svg_font

STEPS = (
    "configuration",
    "launch-review",
    "download-review",
    "download-progress",
    "measurement",
    "finished",
    "result-overview",
)


def make_synthetic_manifest(actual: Path) -> Path:
    folder = actual / "journey"
    folder.mkdir(parents=True)
    scenarios = []
    for name, lang, size in SCENARIOS:
        steps = []
        path = folder / name
        path.mkdir()
        for step in STEPS:
            (path / f"{step}.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg"><text>synthetic</text></svg>',
                encoding="utf-8",
            )
            page = (
                "run-tab"
                if step in {"configuration", "launch-review"}
                else "results-tab"
                if step == "result-overview"
                else "monitor-tab"
            )
            screen = (
                "RecoveryScreen"
                if step == "launch-review"
                else "PreparationScreen"
                if step == "download-review"
                else "CjkScreen"
            )
            steps.append(
                {
                    "id": step,
                    "checkpoint": CHECKPOINTS[step],
                    "image": f"{name}/{step}.svg",
                    "action": "synthetic",
                    "page": page,
                    "screen": screen,
                    "focus": "start-run",
                    "buttons": [],
                    "regions": {},
                    "terminal": list(size),
                }
            )
        scenarios.append({"id": name, "language": lang, "terminal": list(size), "steps": steps})
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fixture": "synthetic-pilot",
                "collection_executed": False,
                "scenarios": scenarios,
            }
        ),
        encoding="utf-8",
    )
    return folder


def test_pilot_journey_is_visible_separately_from_golden_comparisons(tmp_path):
    folder = make_synthetic_manifest(tmp_path)
    assert load_manifest(folder)
    page = render_report(actual=tmp_path).read_text(encoding="utf-8")
    assert 'id="pilot-step-zh-wide-download-review"' in page
    assert 'id="pilot-step-zh-wide-finished"' in page
    assert 'id="pilot-step-zh-wide-result-overview"' in page
    assert page.count("data:image/svg+xml;base64,") >= 21
    assert "连续 Pilot 图像由测试夹具生成" in page
    assert "Comparison not available." in page  # No PNG report.json is present.


def test_absent_journey_does_not_claim_a_successful_pilot_run(tmp_path):
    page = render_report(actual=tmp_path).read_text(encoding="utf-8")
    assert "本次没有连续走查证据" in page
    assert 'id="pilot-step-zh-wide-finished"' not in page


@pytest.mark.parametrize(
    "bad", ["../../outside.svg", "/tmp/user-report.svg", "zh-wide/another.svg"]
)
def test_untrusted_journey_image_references_are_rejected(tmp_path, bad):
    folder = make_synthetic_manifest(tmp_path)
    document = json.loads((folder / "manifest.json").read_text())
    document["scenarios"][0]["steps"][0]["image"] = bad
    (folder / "manifest.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unsafe synthetic Pilot path"):
        render_report(actual=tmp_path)


def test_symlinked_journey_directory_cannot_escape_fixture_root(tmp_path):
    folder = make_synthetic_manifest(tmp_path)
    outside = tmp_path / "external-artifacts"
    (folder / "zh-wide").rename(outside)
    (folder / "zh-wide").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="invalid/missing synthetic Pilot image"):
        load_manifest(folder)


def test_manifest_cannot_claim_real_collection_execution(tmp_path):
    folder = make_synthetic_manifest(tmp_path)
    document = json.loads((folder / "manifest.json").read_text())
    document["collection_executed"] = True
    (folder / "manifest.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="invalid synthetic Pilot manifest identity"):
        render_report(actual=tmp_path)


def test_rich_svg_font_override_is_cjk_and_offline():
    console = Console(record=True, width=40)
    console.print("中文 CJK glyphs")
    patched = use_cjk_svg_font(console.export_svg())
    assert 'font-family: "Noto Sans Mono CJK SC", monospace;' in patched
    assert 'font-family: "Fira Code";' not in patched
    assert "cdnjs.cloudflare.com" not in patched
    assert "中文" in patched


def test_pilot_png_previews_take_priority_over_svg(tmp_path):
    folder = make_synthetic_manifest(tmp_path)
    preview = folder / "zh-wide/configuration.png"
    Image.new("RGB", (100, 50), "white").save(preview)
    doc = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    doc["scenarios"][0]["steps"][0]["preview"] = "zh-wide/configuration.png"
    (folder / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
    html = render_report(actual=tmp_path).read_text(encoding="utf-8")
    assert "合成 Pilot PNG（Noto CJK）" in html
    assert "data:image/png;base64," in html
    assert 'id="pilot-step-zh-wide-configuration"' in html


def test_untrusted_pilot_png_reference_is_rejected(tmp_path):
    folder = make_synthetic_manifest(tmp_path)
    doc = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    doc["scenarios"][0]["steps"][0]["preview"] = "../../private.png"
    (folder / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe synthetic Pilot preview path"):
        render_report(actual=tmp_path)


def test_pilot_png_renderer_uses_exact_dimensions_for_resized_steps(tmp_path):
    from unittest.mock import call, patch

    from scripts.tui_pilot_png import capture_pilot_previews

    steps = [
        {"id": "configuration", "terminal": [120, 36]},
        {"id": "results-small", "terminal": [80, 24]},
    ]
    scenarios = [{"id": "zh-wide", "terminal": [120, 36], "steps": steps}]
    with (
        patch("scripts.tui_pilot_png.check_capture_prerequisites"),
        patch("scripts.tui_pilot_png._capture_size") as capture_size,
    ):
        capture_pilot_previews(tmp_path, scenarios)
    assert capture_size.call_args_list == [
        call(tmp_path, "zh-wide", (120, 36), [steps[0]]),
        call(tmp_path, "zh-wide", (80, 24), [steps[1]]),
    ]
