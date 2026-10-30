"""Compare VHS screenshots with reviewed Git-tracked golden PNGs.

This tool NEVER creates or changes baselines. Run it after capture_tui_png.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageChops, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.visual_png.scenarios import SCENARIOS  # noqa: E402

BASELINES = ROOT / "tests" / "visual_png" / "__snapshots__"
ACTUAL = ROOT / "internal-testing" / "tui-png"
# Ignore small color/antialiasing variations, not changes in glyph position.
COLOR_THRESHOLD = 32
MAX_DIFFERENT_PIXELS = 120


@dataclass(frozen=True)
class ImageResult:
    scene: str
    status: str
    changed_pixels: int = 0
    allowed_pixels: int = MAX_DIFFERENT_PIXELS
    width: int = 0
    height: int = 0
    details: str = ""


def _diff(expected: Image.Image, actual: Image.Image, destination: Path) -> int:
    """Save a red-highlighted diff; return number of significantly changed pixels."""
    left = expected.convert("RGB")
    right = actual.convert("RGB")
    if left.size != right.size:
        width = max(left.width, right.width)
        height = max(left.height, right.height)
        canvas = Image.new("RGB", (width * 3, height), "#22252b")
        canvas.paste(left, (0, 0))
        canvas.paste(right, (width, 0))
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination)
        return width * height
    delta = ImageChops.difference(left, right)
    red, green, blue = delta.split()
    max_delta = ImageChops.lighter(red, ImageChops.lighter(green, blue))
    mask = max_delta.point(lambda value: 255 if value >= COLOR_THRESHOLD else 0)
    count = mask.histogram()[255]
    overlay = Image.blend(left, right, 0.5)
    overlay.paste(Image.new("RGB", left.size, "#ff3950"), mask=mask)
    destination.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(destination)
    return count


def compare_scene(scene: str, *, baselines: Path, actual: Path, differences: Path) -> ImageResult:
    """Fail closed on missing/corrupt baselines; never overwrite expected images."""
    expected_path = baselines / f"{scene}.png"
    actual_path = actual / f"{scene}.png"
    if not expected_path.is_file():
        return ImageResult(scene, "missing-baseline", details=str(expected_path))
    if not actual_path.is_file():
        return ImageResult(scene, "missing-capture", details=str(actual_path))
    try:
        with Image.open(expected_path) as golden, Image.open(actual_path) as current:
            golden.load()
            current.load()
            if golden.format != "PNG" or current.format != "PNG":
                return ImageResult(scene, "invalid-format", details="Both images must be PNG")
            width, height = current.size
            destination = differences / f"{scene}.png"
            changed = _diff(golden, current, destination)
            if golden.size != current.size:
                return ImageResult(
                    scene,
                    "size-mismatch",
                    changed_pixels=changed,
                    width=width,
                    height=height,
                    details=f"expected {golden.size}, actual {current.size}",
                )
            status = "passed" if changed <= MAX_DIFFERENT_PIXELS else "visual-regression"
            if status == "passed":
                destination.unlink(missing_ok=True)
            else:
                expected_copy = differences / "expected" / f"{scene}.png"
                expected_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(expected_path, expected_copy)
            return ImageResult(scene, status, changed_pixels=changed, width=width, height=height)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        return ImageResult(scene, "invalid-image", details=str(exc))


def compare_all(
    scenes: tuple[str, ...],
    *,
    baselines: Path = BASELINES,
    actual: Path = ACTUAL,
    differences: Path | None = None,
) -> list[ImageResult]:
    """Write a machine-readable summary even if one or more scenes fail."""
    diff_dir = differences or actual / "diff"
    diff_dir.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        (diff_dir / f"{scene}.png").unlink(missing_ok=True)
        (diff_dir / "expected" / f"{scene}.png").unlink(missing_ok=True)
    results = [
        compare_scene(scene, baselines=baselines, actual=actual, differences=diff_dir)
        for scene in scenes
    ]
    (actual / "report.json").write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, default=ACTUAL)
    parser.add_argument("--baselines", type=Path, default=BASELINES)
    parser.add_argument("--diff", type=Path, default=None)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default=None)
    args = parser.parse_args()
    scenes = (args.scenario,) if args.scenario else tuple(SCENARIOS)
    results = compare_all(
        scenes, baselines=args.baselines, actual=args.actual, differences=args.diff
    )
    for result in results:
        print(
            f"{result.scene}: {result.status} ({result.changed_pixels} changed pixels; "
            f"limit {result.allowed_pixels}) {result.details}",
            flush=True,
        )
    return int(any(result.status != "passed" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
