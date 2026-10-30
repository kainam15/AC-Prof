"""Image regression decision tests that require no VHS, Chrome or snapshots."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.compare_tui_png import compare_all, compare_scene


def _image(path: Path, *, width: int = 100, height: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height), "#20222a")
    image.save(path)


def test_identical_snapshot_passes_and_removes_stale_diff(tmp_path: Path) -> None:
    expected, actual, diff = (tmp_path / name for name in ("golden", "actual", "diff"))
    _image(expected / "form.png")
    _image(actual / "form.png")
    diff.mkdir()
    (diff / "form.png").write_bytes(b"stale")
    report = compare_all(("form",), baselines=expected, actual=actual, differences=diff)
    assert report[0].status == "passed"
    assert not (diff / "form.png").exists()
    assert json.loads((actual / "report.json").read_text())[0]["status"] == "passed"


def test_large_local_change_fails_and_saves_evidence(tmp_path: Path) -> None:
    expected, actual, diff = (tmp_path / name for name in ("golden", "actual", "diff"))
    _image(expected / "error-modal.png")
    _image(actual / "error-modal.png")
    with Image.open(actual / "error-modal.png") as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 44, 33), fill="#e4e4e4")
        image.save(actual / "error-modal.png")
    result = compare_scene("error-modal", baselines=expected, actual=actual, differences=diff)
    assert result.status == "visual-regression"
    assert result.changed_pixels > result.allowed_pixels
    assert (diff / "error-modal.png").is_file()
    assert (diff / "expected" / "error-modal.png").is_file()


def test_minor_color_difference_is_ignored(tmp_path: Path) -> None:
    expected, actual, diff = (tmp_path / name for name in ("golden", "actual", "diff"))
    _image(expected / "form.png")
    _image(actual / "form.png")
    with Image.open(actual / "form.png") as image:
        image.putpixel((12, 12), (47, 49, 54))
        image.save(actual / "form.png")
    assert (
        compare_scene("form", baselines=expected, actual=actual, differences=diff).status
        == "passed"
    )


def test_changed_dimensions_fail(tmp_path: Path) -> None:
    expected, actual, diff = (tmp_path / name for name in ("golden", "actual", "diff"))
    _image(expected / "form.png")
    _image(actual / "form.png", width=110)
    result = compare_scene("form", baselines=expected, actual=actual, differences=diff)
    assert result.status == "size-mismatch"
    assert (diff / "form.png").exists()


def test_missing_baseline_and_capture_fail_closed(tmp_path: Path) -> None:
    expected, actual, diff = (tmp_path / name for name in ("golden", "actual", "diff"))
    _image(actual / "form.png")
    assert (
        compare_scene("form", baselines=expected, actual=actual, differences=diff).status
        == "missing-baseline"
    )
    _image(expected / "form.png")
    (actual / "form.png").unlink()
    assert (
        compare_scene("form", baselines=expected, actual=actual, differences=diff).status
        == "missing-capture"
    )
