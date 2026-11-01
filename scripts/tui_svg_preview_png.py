"""Render read-only CJK PNG previews of committed Rich SVG baselines.

Previews are *not* PNG Goldens, and never enter pixel comparison. Chrome runs
only against a temporary offline HTML rendering of each fixed SVG snapshot.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tui_flow_catalog import SVG_DIR as SVG_ROOT, svg_inventory  # noqa: E402

DEFAULT_OUTPUT = ROOT / "internal-testing/shared/tui-png/svg-preview"
FONT_FAMILY = "Noto Sans Mono CJK SC"
FONT_FACE = re.compile(r"(?m)^    @font-face \{\n.*?^    \}\n", re.DOTALL)
TEXT_LENGTH = re.compile(r'(<text\b[^>]*\btextLength="[^"]+"[^>]*)(?=>)')
VIEW_BOX = re.compile(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"')


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_svg(source: str) -> tuple[str, int, int]:
    """Repair Rich glyph spacing for the preview without changing the SVG Golden."""
    svg = ET.fromstring(source)
    if svg.tag != "{http://www.w3.org/2000/svg}svg":
        raise ValueError("not a Rich SVG")
    view = VIEW_BOX.search(source[:500])
    if view is None:
        raise ValueError("unsupported Rich SVG viewBox")
    width, height = (math.ceil(float(x)) for x in view.groups())
    if not (100 <= width <= 4096 and 100 <= height <= 4096):
        raise ValueError("unsupported SVG dimensions")
    patched, faces = FONT_FACE.subn("", source)
    if faces != 2 or patched.count("font-family: Fira Code, monospace;") != 1:
        raise ValueError("unexpected Rich SVG font template")
    patched = patched.replace(
        "font-family: Fira Code, monospace;",
        f'font-family: "{FONT_FAMILY}", monospace;',
    ).replace(
        "font-variant-east-asian: full-width;",
        "font-variant-ligatures: none;",
    )
    patched, adjusted = TEXT_LENGTH.subn(r'\1 lengthAdjust="spacingAndGlyphs"', patched)
    if adjusted < 1:
        raise ValueError("Rich SVG lacks positioned text segments")
    patched = patched.replace(
        view.group(0),
        f'{view.group(0)} width="{width}" height="{height}"',
        1,
    )
    return patched, width, height


def _preview_name(source: str) -> str:
    path = Path(source)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.parts[: len(SVG_ROOT.parts)] != SVG_ROOT.parts
        or path.suffix != ".svg"
    ):
        raise ValueError(f"unsafe SVG preview source: {source!r}")
    return (path.relative_to(SVG_ROOT)).with_suffix(".png").as_posix()


def load_preview_index(*, root: Path, output: Path) -> dict[str, Path]:
    """Validate optional generated previews against exact checked-in SVG bytes."""
    manifest = output / "manifest.json"
    if not manifest.is_file():
        return {}
    if manifest.is_symlink() or manifest.stat().st_size > 100_000:
        raise ValueError("unsafe SVG preview manifest")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "format", "entries"}
        or document["schema_version"] != 1
        or document["format"] != "svg-cjk-png-preview"
        or not isinstance(document["entries"], list)
        or len(document["entries"]) > 256
    ):
        raise ValueError("invalid SVG preview manifest")
    index: dict[str, Path] = {}
    allowed = set(svg_inventory(root=root))
    for row in document["entries"]:
        if not isinstance(row, dict) or set(row) != {
            "source",
            "source_sha256",
            "preview",
            "preview_sha256",
        }:
            raise ValueError("invalid SVG preview entry")
        name = row["source"]
        if not isinstance(name, str) or name not in allowed or name in index:
            raise ValueError("duplicate/invalid SVG preview source")
        expected = _preview_name(name)
        if row["preview"] != expected:
            raise ValueError("unsafe SVG preview path")
        source = root / name
        preview = output / expected
        if (
            not source.is_file()
            or not source.resolve().is_relative_to((root / SVG_ROOT).resolve())
            or source.is_symlink()
            or not preview.is_file()
            or not preview.resolve().is_relative_to(output.resolve())
            or preview.is_symlink()
            or preview.stat().st_size > 2_000_000
            or _hash(source) != row["source_sha256"]
            or _hash(preview) != row["preview_sha256"]
            or not preview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        ):
            raise ValueError(f"invalid/stale SVG preview: {name}")
        index[name] = preview
    return index


def generate(*, root: Path = ROOT, output: Path = DEFAULT_OUTPUT) -> int:
    """Build an isolated, deterministic preview set from committed SVGs."""
    from PIL import Image

    if output.absolute() != (root / "internal-testing/shared/tui-png/svg-preview").absolute():
        raise ValueError("SVG previews must use their dedicated output directory")

    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        raise RuntimeError("Chrome/Chromium required for SVG PNG previews")
    family = subprocess.run(
        ["fc-match", "-f", "%{family}", FONT_FAMILY],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout
    if FONT_FAMILY not in family:
        raise RuntimeError(f"Missing required CJK font: {FONT_FAMILY} ({family})")
    sources = [root / name for name in svg_inventory(root=root)]
    if not sources:
        raise RuntimeError("No committed SVG baselines found")
    # Remove stale preview evidence before the new capture. Manifest is last.
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise ValueError("unsafe SVG preview output directory")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    entries = []
    with tempfile.TemporaryDirectory(prefix="acprof-svg-cjk-preview-") as temp:
        tmp = Path(temp)
        for i, source in enumerate(sources):
            if source.is_symlink() or not source.resolve().is_relative_to(
                (root / SVG_ROOT).resolve()
            ):
                raise ValueError("symlinked SVG baseline")
            relative = source.relative_to(root).as_posix()
            target_name = _preview_name(relative)
            target = output / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            svg, width, height = _prepare_svg(source.read_text(encoding="utf-8"))
            html = tmp / f"scene-{i}.html"
            html.write_text(
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<style>html,body{margin:0;padding:0;overflow:hidden;"
                "background:#fff}</style></head><body>" + svg + "</body></html>",
                encoding="utf-8",
            )
            png = tmp / f"scene-{i}.png"
            result = subprocess.run(
                [
                    chrome,
                    "--headless=new",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--hide-scrollbars",
                    "--force-device-scale-factor=1",
                    f"--user-data-dir={tmp / f'profile-{i}'}",
                    f"--screenshot={png}",
                    f"--window-size={width},{height + 100}",
                    html.as_uri(),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode or not png.is_file():
                raise RuntimeError(
                    f"SVG preview Chrome failed for {relative}: "
                    + (result.stdout + result.stderr)[-1800:]
                )
            with Image.open(png) as picture:
                if picture.width < width or picture.height < height:
                    raise RuntimeError(f"truncated SVG preview: {relative}")
                picture.crop((0, 0, width, height)).convert("RGB").save(target)
            if target.stat().st_size > 2_000_000:
                raise RuntimeError(f"SVG preview too large: {relative}")
            entries.append(
                {
                    "source": relative,
                    "source_sha256": _hash(source),
                    "preview": target_name,
                    "preview_sha256": _hash(target),
                }
            )
    (output / "manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "format": "svg-cjk-png-preview", "entries": entries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(entries)


def main() -> None:
    print(f"Generated {generate()} Noto CJK SVG previews")


if __name__ == "__main__":
    main()
