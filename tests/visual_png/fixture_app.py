"""Run the real Textual app with test-only fixed state under a VHS terminal.

Do not use in a production run. No real preflight, downloads, subprocesses,
hardware queries, model resolution, or profiling are allowed in this fixture.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import NoReturn
from unittest.mock import Mock, patch

from textual.pilot import Pilot
from textual.widgets import Button, TabbedContent, TabPane

from acprof.platform import Environment
from acprof.tui.app import AcprofTui
from acprof.tui.run.process import ProcessLifecycle, StopResult
from tests.support.tui_fixtures import SNAPSHOT_TITLE
from tests.support.tui_waits import wait_for_condition
from tests.visual_png.scenarios import SCENARIOS

GUARDED_OPERATIONS = (
    "subprocess.Popen",
    "os.kill",
    "os.killpg",
    "socket.socket.connect",
    "socket.socket.connect_ex",
    "socket.getaddrinfo",
    "acprof.tui.run.diagnostics.quick_preflight",
    "acprof.tui.run.preflight_controller.quick_preflight",
    "acprof.tui.run.preflight_controller.scan_environment",
    "acprof.tui.settings.environment.quick_preflight",
)


def forbidden_operation(operation: str, *_args: object, **_kwargs: object) -> NoReturn:
    raise RuntimeError(f"PNG fixture must not execute a real operation: {operation}")


@dataclass
class FixtureIsolation:
    """Remember violations even when application error handling catches them."""

    report_path: Path | None = None
    blocked_calls: list[str] = field(default_factory=list)
    state: str = "active"
    closed: bool = False
    _record_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def write_record(self) -> None:
        if self.report_path is None:
            return
        with self._record_lock:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            pending = self.report_path.with_name(self.report_path.name + ".tmp")
            pending.write_text(json.dumps({
                "state": self.state, "closed": self.closed,
                "blocked_calls": self.blocked_calls,
            }, indent=2) + "\n", encoding="utf-8")
            pending.replace(self.report_path)

    def forbid(self, operation: str, *_args: object, **_kwargs: object) -> NoReturn:
        self.blocked_calls.append(operation)
        self.state = "blocked"
        self.write_record()
        raise RuntimeError(f"PNG fixture must not execute a real operation: {operation}")


@contextmanager
def isolate_fixture(*, report_path: Path | None = None) -> Iterator[FixtureIsolation]:
    """Guard construction, rendering and cleanup; always restore global patches."""
    isolation = FixtureIsolation(report_path=report_path)
    with ExitStack() as patches:
        for target in GUARDED_OPERATIONS:
            patches.enter_context(patch(target, partial(isolation.forbid, target)))
        # Instance entry points bind this recorder while the app is constructed.
        patches.enter_context(patch(__name__ + ".forbidden_operation", isolation.forbid))
        isolation.write_record()
        try:
            yield isolation
        except BaseException:
            if not isolation.blocked_calls:
                isolation.state = "failed"
            raise
        else:
            isolation.closed = not isolation.blocked_calls
            if isolation.closed:
                isolation.state = "closed"
        finally:
            isolation.write_record()
            if isolation.blocked_calls:
                raise AssertionError(f"PNG fixture blocked real operations: {isolation.blocked_calls}")


def stop_fixture_process(
    lifecycle: ProcessLifecycle, *, closing: bool = False, expected_process: object = None,
) -> StopResult:
    """Release only a synthetic process; never call production signal cleanup."""
    process = lifecycle.process
    if expected_process is not None and process is not expected_process:
        return StopResult(None, None)
    if process is None:
        return StopResult(None, None)
    if not isinstance(process, Mock) or process.pid is not None:
        forbidden_operation("lifecycle.stop(non-fixture)")
    process.returncode = 0
    process.poll.return_value = 0
    lifecycle.process = None
    return StopResult(None, 0)


class VisualFixtureApp(AcprofTui):
    TITLE = SNAPSHOT_TITLE

    def __init__(self, scenario: str, *, settings_path: Path):
        self.scenario = SCENARIOS[scenario]
        self.scene_ready = asyncio.Event()
        super().__init__(self.scenario.config(), settings_path=settings_path)
        # Snapshot identity remains stable across host platforms.
        self.title = self.TITLE
        self.configuration.preferences = replace(
            self.configuration.preferences,
            language=self.scenario.language,
            theme="acprof-graphite",
            show_command_bar=True,
        )
        # Only fixture startup is a no-op; unexpected real work remains an error.
        self.preflight.quick_check = lambda: None  # type: ignore[method-assign]
        self.results.restore_recent_result = lambda: None  # type: ignore[method-assign]
        for owner, name, operation in (
            (self.preflight, "execute_quick_check", "preflight.execute_quick_check"),
            (self.process, "execute_command", "process.execute_command"),
            (self.images, "refresh_images", "images.refresh_images"),
            (self.execution.lifecycle, "start", "lifecycle.start"),
            (self.updates, "start_check", "updates.start_check"),
        ):
            setattr(owner, name, partial(forbidden_operation, operation))
        self.execution.lifecycle.stop = partial(  # type: ignore[method-assign]
            stop_fixture_process, self.execution.lifecycle,
        )

    def on_mount(self) -> None:
        # Textual dispatches the production mount handler separately. A worker
        # can await the resulting focus and navigation messages off the UI pump.
        self.scene_worker = self.run_worker(self.prepare_scene(), name="prepare PNG scene")

    async def wait_for_scene(self, pilot: Pilot) -> None:
        """Wait for navigation and mounting before checking the rendered scene."""
        await self.scene_worker.wait()
        await wait_for_condition(pilot, self.scene_ready.is_set,
                                 description="mounted PNG scene refreshed", timeout=5)

    async def prepare_scene(self) -> None:
        pilot = Pilot(self)
        await pilot.pause()
        tabs = self.query_one("#main-tabs", TabbedContent)
        page = self.scenario.page
        if tabs.active != page:
            self.activate_tab(page)
        pane = tabs.query_one(f"#{page}", TabPane)
        button = self.query_one(f"#nav-{page}", Button)
        await wait_for_condition(
            pilot,
            lambda: tabs.active == page and pane.display and pane.region.area > 0 and button.has_class("selected-page"),
            description=f"PNG scene page {page} visible and selected", timeout=8,
        )
        self.scenario.mount(self)
        self.call_after_refresh(self.scene_ready.set)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=tuple(SCENARIOS))
    parser.add_argument("--fixture-dir", type=Path, required=True)
    args = parser.parse_args()
    fixture_dir = args.fixture_dir.resolve()
    fixture_dir.mkdir(parents=True, exist_ok=True)
    guard_report = os.environ.get("ACPROF_PNG_GUARD_REPORT")
    # Golden PNGs must not depend on the runner's native/VM/WSL identity.
    with (
        isolate_fixture(report_path=Path(guard_report) if guard_report else None),
        patch.dict(os.environ),
        patch("acprof.tui.app.PROJECT_DIR", fixture_dir / "workspace"),
        patch("acprof.tui.app.PYTHON_EXECUTABLE", Path(sys.executable)),
        patch(
            "acprof.platform.detect_environment",
            return_value=Environment(environment="native_linux"),
        ),
    ):
        # Textual treats even an empty NO_COLOR as a monochrome request.
        # Keep the reviewed palette fixed and restore the caller's environment.
        os.environ.pop("NO_COLOR", None)
        app = VisualFixtureApp(args.scenario, settings_path=fixture_dir / "settings.json")
        return app.run() or 0


if __name__ == "__main__":
    raise SystemExit(main())
