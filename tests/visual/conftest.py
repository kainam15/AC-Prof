"""Deterministic SVG snapshot fixtures without pinned third-party pytest internals."""
from __future__ import annotations

import difflib
import inspect
import os
import re
from pathlib import Path

import pytest
from textual.app import App


def _normalize_svg(svg: str) -> str:
    # Rich generates a unique terminal identifier on every screenshot.
    return re.sub(r"\bterminal-\d+-([\w-]+)", r"terminal-\1",
                  "\n".join(line.rstrip() for line in svg.splitlines()) + "\n")


@pytest.fixture
def snap_compare(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Compare headless Textual output against reviewed, checked-in SVG evidence."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("NO_COLOR", raising=False)

    def compare(app: App, *, terminal_size: tuple[int, int], run_before=None) -> bool:
        async def capture(pilot) -> None:
            if run_before is not None:
                outcome = run_before(pilot)
                if inspect.isawaitable(outcome):
                    await outcome
            await pilot.pause()
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()
            await pilot.wait_for_scheduled_animations()
            app.exit(app.export_screenshot(title=app.title, simplify=True))
        actual = _normalize_svg(app.run(headless=True, auto_pilot=capture, size=terminal_size))
        node = request.node
        baseline = (Path(str(node.path)).parent / "__snapshots__" /
                    Path(node.path).stem / f"{node.name}.svg")
        if not baseline.is_file():
            pytest.fail(f"No reviewed SVG baseline: {baseline}")
        expected = baseline.read_text(encoding="utf-8")
        if actual != expected:
            output = os.environ.get("ACPROF_SNAPSHOT_OUTPUT_DIR")
            folder = Path(output) if output else tmp_path
            folder.mkdir(parents=True, exist_ok=True)
            actual_path = folder / (f"{node.name}.svg" if output else "actual.svg")
            actual_path.write_text(actual, encoding="utf-8")
            diff = "".join(difflib.unified_diff(expected.splitlines(keepends=True),
                                                actual.splitlines(keepends=True),
                                                fromfile=str(baseline), tofile=str(actual_path), n=2))
            pytest.fail(f"SVG snapshot changed (review output before updating): {actual_path}\n{diff[:3000]}")
        return True
    return compare
