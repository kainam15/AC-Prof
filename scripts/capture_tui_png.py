"""Optional, desktop-free VHS screenshots of real AC-Prof TUI fixture scenes."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.visual_png.scenarios import SCENARIOS  # noqa: E402

COLUMNS = 120
ROWS = 36
FONT_FAMILY = "Noto Sans Mono CJK SC"
FONT_SIZE = 16
CAPTURE_TIMEOUT_SECONDS = 90


def _quoted(value: str | Path) -> str:
    """VHS accepts quoted strings with the same escapes as JSON strings."""
    return json.dumps(str(value), ensure_ascii=False)


def build_tape(scenario: str, screenshot: Path, fixture_home: Path, *, python: Path) -> str:
    """Produce an inspectable, fixed-cell-size tape with an isolated fixture home."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown PNG scenario: {scenario}")
    command = shlex.join((
        str(python), "-m", "tests.visual_png.fixture_app",
        "--scenario", scenario, "--fixture-dir", str(fixture_home),
    ))
    environment = {
        "HOME": fixture_home,
        "XDG_CONFIG_HOME": fixture_home / "config",
        "XDG_CACHE_HOME": fixture_home / "cache",
        "HF_HOME": fixture_home / "hf",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "PYTHONHASHSEED": "0",
    }
    lines = [
        'Require bash',
        'Set Shell "bash"',
        f"Set Columns {COLUMNS}",
        f"Set Rows {ROWS}",
        f"Set FontFamily {_quoted(FONT_FAMILY)}",
        f"Set FontSize {FONT_SIZE}",
        "Set Padding 0",
        "Set Margin 0",
        "Set TypingSpeed 1ms",
        *(f"Env {key} {_quoted(value)}" for key, value in environment.items()),
        "Hide",
        f"Type {_quoted(command)}",
        "Enter",
        "Show",
        # VHS screen-regex detection can miss a rapid full-frame redraw after
        # synthetic preflight completion. Give that single scene a fixed settle.
        (f"Wait+Screen /{SCENARIOS[scenario].ready_pattern}/"
         if SCENARIOS[scenario].ready_pattern else "Sleep 4s"),
        "Sleep 600ms",
        f"Screenshot {_quoted(screenshot)}",
        "",
    ]
    return "\n".join(lines)


def check_capture_prerequisites() -> None:
    required = ("vhs", "ttyd", "ffmpeg", "fc-match")
    missing = [program for program in required if shutil.which(program) is None]
    if not any(shutil.which(program) for program in ("google-chrome", "chromium", "chromium-browser")):
        missing.append("Chrome/Chromium")
    if missing:
        raise RuntimeError(
            f"Optional PNG testing tools missing: {', '.join(missing)}. "
            "See docs/development/png-visual-validation.md; these are not AC-Prof runtime dependencies."
        )
    family = subprocess.run(
        ["fc-match", "-f", "%{family}", FONT_FAMILY],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout
    if FONT_FAMILY not in family:
        raise RuntimeError(f"Required CJK monospace font is unavailable: {FONT_FAMILY} (resolved: {family!r})")


def capture(scenarios: tuple[str, ...], output_dir: Path, *, python: Path) -> list[Path]:
    """Capture scenes sequentially; each uses an independent temporary home and tape."""
    check_capture_prerequisites()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshots: list[Path] = []
    for scenario in scenarios:
        screenshot = output_dir / f"{scenario}.png"
        # A stale image must never make a failed VHS invocation look successful.
        screenshot.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix="acprof-png-vhs-") as temp:
            temp_dir = Path(temp)
            fixture_home = temp_dir / "home"
            fixture_home.mkdir()
            tape = temp_dir / "capture.tape"
            tape.write_text(build_tape(scenario, screenshot, fixture_home, python=python), encoding="utf-8")
            try:
                result = subprocess.run(
                    ["vhs", str(tape)], cwd=ROOT, capture_output=True, text=True,
                    timeout=CAPTURE_TIMEOUT_SECONDS, check=False,
                    env={**os.environ, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC"},
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"VHS timed out for {scenario}; tape: {tape.read_text()}") from exc
            if result.returncode:
                raise RuntimeError(
                    f"VHS failed for {scenario} (exit {result.returncode}).\n"
                    f"{(result.stdout + result.stderr)[-3500:]}\n"
                    f"Tape:\n{tape.read_text()}"
                )
        if not screenshot.is_file() or screenshot.stat().st_size < 1024:
            raise RuntimeError(f"VHS did not produce a usable PNG for {scenario}: {screenshot}")
        screenshots.append(screenshot)
        print(f"{scenario}: {screenshot}", flush=True)
    return screenshots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selected = parser.add_mutually_exclusive_group(required=True)
    selected.add_argument("--scenario", choices=tuple(SCENARIOS))
    selected.add_argument("--all", action="store_true", help="Capture every fixed fixture")
    parser.add_argument("--output", type=Path, default=ROOT / "internal-testing" / "tui-png")
    args = parser.parse_args()
    scenarios = tuple(SCENARIOS) if args.all else (args.scenario,)
    try:
        capture(scenarios, args.output, python=Path(sys.executable))
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
