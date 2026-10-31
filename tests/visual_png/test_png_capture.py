"""Opt-in PNG visual smoke checks: artifact existence, dimensions, and distinct scenes.

Run with ACPROF_PNG_VISUAL=1; ordinary pytest does not require VHS.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from scripts.capture_tui_png import (
    FONT_FAMILY,
    ROOT,
    build_tape,
    capture,
)
from scripts.compare_tui_png import compare_all
from tests.visual_png.scenarios import SCENARIOS

pytestmark = pytest.mark.visual_png


def test_tapes_fix_terminal_cells_font_and_fixture_boundary(tmp_path: Path) -> None:
    for name in SCENARIOS:
        tape = build_tape(
            name, tmp_path / f"{name}.png", tmp_path / "fixture", python=Path(sys.executable)
        )
        assert f"Set Columns {SCENARIOS[name].columns}\n" in tape
        assert f"Set Rows {SCENARIOS[name].rows}\n" in tape
        assert f'Set FontFamily "{FONT_FAMILY}"' in tape
        assert "HF_HUB_OFFLINE" in tape
        assert "Sleep 4s" in tape
        assert "Output " in tape
        assert "Set Framerate 1" in tape
        assert "frames/" in tape
        assert "acprof run" not in tape
        assert "docker " not in tape


def test_stale_png_does_not_mask_missing_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    stale = output / "form.png"
    stale.write_bytes(b"old screenshot")
    monkeypatch.setattr("scripts.capture_tui_png.check_capture_prerequisites", lambda: None)
    monkeypatch.setattr(
        "scripts.capture_tui_png.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="", stderr=""),
    )
    with pytest.raises(RuntimeError, match="did not render PNG text frames"):
        capture(("form",), output, python=Path(sys.executable))
    assert not stale.exists()


def test_fixture_pins_environment_only_during_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from acprof import platform
    from acprof.platform import Environment
    from tests.visual_png import fixture_app

    # Simulate a virtualized CI runner without depending on the actual host.
    monkeypatch.setattr(platform, "detect_environment", lambda: Environment(environment="vm"))
    observed: list[str] = []

    class FakeVisualFixture:
        def __init__(self, scenario: str, *, settings_path: Path) -> None:
            assert scenario == "form"
            assert settings_path == tmp_path / "settings.json"

        def run(self) -> int:
            observed.append(platform.detect_environment().label)
            return 0

    monkeypatch.setattr(fixture_app, "VisualFixtureApp", FakeVisualFixture)
    monkeypatch.setattr(
        sys, "argv", ["fixture_app.py", "--scenario", "form", "--fixture-dir", str(tmp_path)]
    )
    assert fixture_app.main() == 0
    assert observed == ["Native Linux / FULL"]
    assert platform.detect_environment().label == "vm / PARTIAL"


@pytest.mark.skipif(
    os.environ.get("ACPROF_PNG_VISUAL") != "1", reason="opt-in VHS screenshot suite"
)
def test_real_tui_png_scenes() -> None:
    output = ROOT / "internal-testing" / "tui-png"
    paths = capture(tuple(SCENARIOS), output, python=Path(sys.executable))
    grid_dimensions: dict[tuple[int, int], tuple[int, int]] = {}
    samples: set[bytes] = set()
    for path in paths:
        scene = SCENARIOS[path.stem]
        assert path.is_file()
        assert path.stat().st_size > 4096
        with Image.open(path) as image:
            assert image.format == "PNG"
            image.load()
            assert scene.columns * 7 <= image.width <= scene.columns * 18
            assert scene.rows * 12 <= image.height <= scene.rows * 28
            grid = (scene.columns, scene.rows)
            assert grid_dimensions.setdefault(grid, image.size) == image.size
            samples.add(image.convert("RGB").tobytes())
    assert len(samples) == len(SCENARIOS)
    comparisons = compare_all(tuple(SCENARIOS), actual=output)
    failures = [result for result in comparisons if result.status != "passed"]
    assert not failures, f"PNG regressions; inspect {output / 'diff'} and report.json: {failures}"
