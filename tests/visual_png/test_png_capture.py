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
    COLUMNS,
    FONT_FAMILY,
    ROOT,
    ROWS,
    build_tape,
    capture,
)
from tests.visual_png.scenarios import SCENARIOS

pytestmark = pytest.mark.visual_png


def test_tapes_fix_terminal_cells_font_and_fixture_boundary(tmp_path: Path) -> None:
    for name in SCENARIOS:
        tape = build_tape(name, tmp_path / f"{name}.png", tmp_path / "fixture", python=Path(sys.executable))
        assert f"Set Columns {COLUMNS}\n" in tape
        assert f"Set Rows {ROWS}\n" in tape
        assert f'Set FontFamily "{FONT_FAMILY}"' in tape
        assert "HF_HUB_OFFLINE" in tape
        assert "Wait+Screen /" in tape or "Sleep 4s" in tape
        assert "Screenshot " in tape
        assert "acprof run" not in tape
        assert "docker " not in tape


def test_stale_png_does_not_mask_missing_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    stale = output / "form.png"
    stale.write_bytes(b"old screenshot")
    monkeypatch.setattr("scripts.capture_tui_png.check_capture_prerequisites", lambda: None)
    monkeypatch.setattr(
        "scripts.capture_tui_png.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )
    with pytest.raises(RuntimeError, match="did not produce a usable PNG"):
        capture(("form",), output, python=Path(sys.executable))
    assert not stale.exists()


@pytest.mark.skipif(os.environ.get("ACPROF_PNG_VISUAL") != "1", reason="opt-in VHS screenshot suite")
def test_real_tui_png_scenes() -> None:
    output = ROOT / "internal-testing" / "tui-png"
    paths = capture(tuple(SCENARIOS), output, python=Path(sys.executable))
    dimensions: set[tuple[int, int]] = set()
    samples: set[bytes] = set()
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 4096
        with Image.open(path) as image:
            assert image.format == "PNG"
            image.load()
            assert COLUMNS * 7 <= image.width <= COLUMNS * 18
            assert ROWS * 12 <= image.height <= ROWS * 28
            dimensions.add(image.size)
            samples.add(image.convert("RGB").tobytes())
    # Same fixed font/cell grid across scenes; distinct rendered UI content.
    assert len(dimensions) == 1
    assert len(samples) == len(SCENARIOS)
