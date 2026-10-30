"""Run the real Textual app with test-only fixed state under a VHS terminal.

Do not use in a production run. No real preflight, downloads, subprocesses,
hardware queries, model resolution, or profiling are allowed in this fixture.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from acprof.tui.app import AcprofTui
from tests.visual_png.scenarios import SCENARIOS


def forbidden_operation(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("PNG fixture must not execute a real operation")


class VisualFixtureApp(AcprofTui):
    def __init__(self, scenario: str, *, settings_path: Path):
        self.scenario = SCENARIOS[scenario]
        super().__init__(self.scenario.config(), settings_path=settings_path)
        self.configuration.preferences = replace(
            self.configuration.preferences,
            language="zh",
            theme="acprof-graphite",
            show_command_bar=True,
        )
        # No work is launched by a screenshot, even if lifecycle code changes.
        self.preflight.quick_check = lambda: None  # type: ignore[method-assign]
        self.preflight.execute_quick_check = forbidden_operation  # type: ignore[method-assign]
        self.process.execute_command = forbidden_operation  # type: ignore[method-assign]
        self.results.restore_recent_result = lambda: None  # type: ignore[method-assign]
        self.images.refresh_images = forbidden_operation  # type: ignore[method-assign]

    def on_mount(self) -> None:
        super().on_mount()
        self.call_after_refresh(self.scenario.mount, self)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=tuple(SCENARIOS))
    parser.add_argument("--fixture-dir", type=Path, required=True)
    args = parser.parse_args()
    fixture_dir = args.fixture_dir.resolve()
    fixture_dir.mkdir(parents=True, exist_ok=True)
    with patch("acprof.tui.app.PROJECT_DIR", fixture_dir / "workspace"), patch(
        "acprof.tui.app.PYTHON_EXECUTABLE", Path(sys.executable)
    ):
        app = VisualFixtureApp(args.scenario, settings_path=fixture_dir / "settings.json")
        return app.run() or 0


if __name__ == "__main__":
    raise SystemExit(main())
