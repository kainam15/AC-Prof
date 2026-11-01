"""Render synthetic Pilot ANSI states with the existing VHS/xterm.js test renderer.

This is an opt-in developer/CI step: it never launches the AC-Prof collector.
SVG remains read-only diagnostics, while PNG previews are stable across viewers.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

from scripts.capture_tui_png import FONT_FAMILY, FONT_SIZE, check_capture_prerequisites
from scripts.tui_journey_contract import CHECKPOINTS, SCENARIOS

# Each state stays on screen for three seconds; VHS captures it at second two.
_REPLAY_SCRIPT = """import json
import sys
import time
from pathlib import Path
for filename in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")):
    sys.stdout.buffer.write(b"\\x1b[2J\\x1b[H" + Path(filename).read_bytes())
    sys.stdout.buffer.flush()
    time.sleep(3)
time.sleep(12)
"""


def _quoted(value: Path | str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _capture_size(folder: Path, name: str, size: tuple[int, int], steps: list[dict]) -> None:
    """Replay one fixed terminal geometry without changing the mocked UI state."""
    for step in steps:
        if step["id"] not in CHECKPOINTS:
            raise ValueError("unknown Pilot PNG step")
        ansi = folder / name / (step["id"] + ".ansi")
        if not ansi.is_file() or ansi.is_symlink():
            raise RuntimeError(f"Missing synthetic Pilot terminal state: {ansi}")
        # A stale preview must never turn a failed VHS screenshot into success.
        (folder / name / (step["id"] + ".png")).unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="acprof-pilot-vhs-") as temp:
        temporary = Path(temp)
        replay = temporary / "replay.py"
        replay.write_text(_REPLAY_SCRIPT, encoding="utf-8")
        selection = temporary / "states.json"
        selection.write_text(
            json.dumps([str(folder / name / (step["id"] + ".ansi")) for step in steps]),
            encoding="utf-8",
        )
        command = shlex.join((sys.executable, "-u", str(replay), str(selection)))
        cols, rows = size
        lines = [
            f"Output {_quoted(str(temporary / 'frames') + '/')}",
            "Require bash",
            'Set Shell "bash"',
            f"Set Columns {cols}",
            f"Set Rows {rows}",
            f"Set FontFamily {_quoted(FONT_FAMILY)}",
            f"Set FontSize {FONT_SIZE}",
            "Set Framerate 1",
            "Set Padding 0",
            "Set Margin 0",
            "Set TypingSpeed 1ms",
            "Hide",
            f"Type {_quoted(command)}",
            "Enter",
            "Show",
        ]
        for index, step in enumerate(steps):
            lines.append("Sleep 2s" if index == 0 else "Sleep 3s")
            lines.append(f"Screenshot {_quoted(folder / name / (step['id'] + '.png'))}")
        lines.append("Sleep 600ms")
        tape = temporary / "capture.tape"
        tape.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = subprocess.run(
            ["vhs", str(tape)],
            cwd=folder,
            env={**os.environ, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC"},
            text=True,
            capture_output=True,
            timeout=150,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"VHS Pilot PNG failed ({name}/{cols}x{rows}, exit {result.returncode}): "
                + (result.stdout + result.stderr)[-2500:]
            )
    for step in steps:
        png = folder / name / (step["id"] + ".png")
        if not png.is_file() or png.is_symlink() or png.stat().st_size > 2_000_000:
            raise RuntimeError(f"VHS omitted/invalid Pilot preview: {png}")
        with Image.open(png) as image:
            image.verify()
        step["preview"] = f"{name}/{step['id']}.png"


def capture_pilot_previews(folder: Path, scenarios: list[dict]) -> None:
    """Fill preview references only after the exact terminal-size PNG is verified."""
    check_capture_prerequisites()
    expected = {name: terminal for name, _, terminal in SCENARIOS}
    for scenario in scenarios:
        name = scenario["id"]
        if name not in expected or scenario["terminal"] != list(expected[name]):
            raise ValueError("unknown Pilot PNG scenario")
        groups: dict[tuple[int, int], list[dict]] = {}
        for step in scenario["steps"]:
            size = tuple(step["terminal"])
            if size not in (expected[name], (80, 24)):
                raise ValueError("unsupported Pilot terminal resize")
            groups.setdefault(size, []).append(step)
        for size, steps in groups.items():
            _capture_size(folder, name, size, steps)
        print(
            f"{name}: {sum(len(steps) for steps in groups.values())} Noto PNG previews", flush=True
        )
