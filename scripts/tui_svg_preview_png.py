"""Render read-only CJK PNG previews of committed Rich SVG baselines.

Previews are *not* PNG Goldens, and never enter pixel comparison. Chrome runs
only against a temporary offline HTML rendering of each fixed SVG snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.smoke_tui_png_environment import _stop_browser_group  # noqa: E402
from scripts.tui_flow_catalog import SVG_DIR as SVG_ROOT, svg_inventory  # noqa: E402

DEFAULT_OUTPUT = ROOT / "internal-testing/shared/tui-png/svg-preview"
FAILURES_OUTPUT = ROOT / "internal-testing/shared/tui-png/svg-preview-failures"
SMOKE_SCENE = "tests/visual/__snapshots__/test_snapshots/test_fixed_scenes[zh-size0-form].svg"
CHROME_CAPTURE_TIMEOUT_SECONDS = 30
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


def _chrome_capture(chrome: str, html: Path, png: Path, *, width: int, height: int) -> None:
    """Wait for an intact PNG, not Chrome's potentially lingering child processes."""
    from PIL import Image

    argv = [
        chrome,
        "--headless",
        "--ozone-platform=headless",
        # do not let offline preview capture wait on Chrome update/sync traffic
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-default-apps",
        "--disable-sync",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-first-run",
        "--no-default-browser-check",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        f"--user-data-dir={html.parent / (html.stem + '-profile')}",
        f"--screenshot={png}",
        f"--window-size={width},{height + 100}",
        html.as_uri(),
    ]
    stdout_path = html.with_suffix(".stdout.log")
    stderr_path = html.with_suffix(".stderr.log")
    outcome = "Chrome screenshot not produced"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
        except OSError as exc:
            raise RuntimeError(f"Cannot launch Chrome screenshot: {exc}") from exc
        try:
            deadline = time.monotonic() + CHROME_CAPTURE_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if png.is_file() and png.stat().st_size >= 1024:
                    try:
                        with Image.open(png) as picture:
                            picture.verify()
                        return
                    except (OSError, ValueError):
                        # Chrome may still be writing the PNG.
                        pass
                code = process.poll()
                if code is not None:
                    outcome = f"Chrome exited {code} without a valid screenshot"
                    break
                time.sleep(0.1)
            else:
                outcome = (
                    f"Chrome did not produce a valid screenshot within "
                    f"{CHROME_CAPTURE_TIMEOUT_SECONDS}s"
                )
        finally:
            # Chrome's wrapper/zygotes can outlive --screenshot; always reap the
            # whole process group, as in the existing VHS browser smoke check.
            _stop_browser_group(process)
    stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    stdout_tail = stdout_path.read_text(encoding="utf-8", errors="replace")[-1000:]
    raise RuntimeError(
        f"{outcome}: {html.name}; PNG exists={png.is_file()}, "
        f"bytes={png.stat().st_size if png.is_file() else 0}.\n"
        f"Chrome stderr:\n{stderr_tail}\nChrome stdout:\n{stdout_tail}"
    )


def _retain_failure(html: Path, png: Path, *, index: int, scene: str) -> Path:
    """Keep only fixture-derived evidence in the ignored CI artifact directory."""
    if FAILURES_OUTPUT.is_symlink():
        raise ValueError("unsafe SVG preview failure directory")
    FAILURES_OUTPUT.mkdir(parents=True, exist_ok=True)
    target = FAILURES_OUTPUT / f"scene-{index:02d}"
    if target.exists():
        if target.is_symlink() or not target.is_dir():
            raise ValueError("unsafe SVG preview failure evidence")
        shutil.rmtree(target)
    target.mkdir()
    for item in (html, html.with_suffix(".stdout.log"), html.with_suffix(".stderr.log"), png):
        if item.is_file():
            shutil.copy2(item, target / item.name)
    (target / "source.txt").write_text(scene + "\n", encoding="utf-8")
    return target


def _svg_renderer() -> str:
    """Prefer Chrome's isolated headless shell for Rich SVG screenshot previews."""
    for binary in ("chrome-headless-shell", "google-chrome", "chromium"):
        if executable := shutil.which(binary):
            return executable
    raise RuntimeError("Chrome Headless Shell/Chrome/Chromium required for SVG PNG previews")


def generate(*, root: Path = ROOT, output: Path = DEFAULT_OUTPUT, smoke: bool = False) -> int:
    """Build committed SVG previews, or smoke-test one real Chinese Golden."""
    from PIL import Image

    if output.absolute() != (root / "internal-testing/shared/tui-png/svg-preview").absolute():
        raise ValueError("SVG previews must use their dedicated output directory")

    chrome = _svg_renderer()
    family = subprocess.run(
        ["fc-match", "-f", "%{family}", FONT_FAMILY],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout
    if FONT_FAMILY not in family:
        raise RuntimeError(f"Missing required CJK font: {FONT_FAMILY} ({family})")
    catalog = svg_inventory(root=root)
    if not catalog or (smoke and SMOKE_SCENE not in catalog):
        raise RuntimeError("Required committed SVG baselines missing")
    sources = [root / name for name in ([SMOKE_SCENE] if smoke else catalog)]
    # Failed runs must not leave an older preview manifest that looks complete.
    # The smoke path never touches the full set of previews.
    if not smoke:
        if output.exists():
            if output.is_symlink() or not output.is_dir():
                raise ValueError("unsafe SVG preview output directory")
            shutil.rmtree(output)
        output.mkdir(parents=True)
    if FAILURES_OUTPUT.exists():
        if FAILURES_OUTPUT.is_symlink() or not FAILURES_OUTPUT.is_dir():
            raise ValueError("unsafe SVG preview failure directory")
        shutil.rmtree(FAILURES_OUTPUT)
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
            target = (tmp / "smoke-preview.png") if smoke else (output / target_name)
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
            try:
                _chrome_capture(chrome, html, png, width=width, height=height)
            except RuntimeError as exc:
                evidence = _retain_failure(html, png, index=i, scene=relative)
                raise RuntimeError(
                    f"SVG preview failed for {relative}; diagnostic files: {evidence}. {exc}"
                ) from exc
            with Image.open(png) as picture:
                if picture.width < width or picture.height < height:
                    raise RuntimeError(f"truncated SVG preview: {relative}")
                picture.crop((0, 0, width, height)).convert("RGB").save(target)
            if target.stat().st_size > 2_000_000:
                raise RuntimeError(f"SVG preview too large: {relative}")
            if smoke:
                return 1
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Render one committed CJK SVG")
    args = parser.parse_args()
    count = generate(smoke=args.smoke)
    print(f"{'Smoke-verified' if args.smoke else 'Generated'} {count} Noto CJK SVG previews")


if __name__ == "__main__":
    main()
