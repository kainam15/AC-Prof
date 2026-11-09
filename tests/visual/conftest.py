"""Deterministic SVG snapshot fixtures without pinned third-party pytest internals."""
from __future__ import annotations

import difflib
import inspect
import json
import os
import re
from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.app import App

from acprof.tui.run.process import StopResult
from tests.support.tui_fixtures import SNAPSHOT_TITLE


def _normalize_svg(svg: str) -> str:
    # Rich generates a unique terminal identifier on every screenshot.
    return re.sub(r"\bterminal-\d+-([\w-]+)", r"terminal-\1",
                  "\n".join(line.rstrip() for line in svg.splitlines()) + "\n")


class SvgOperations:
    """Own permitted fixture actions; reject every unplanned external operation."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.expected_blocks: list[str] = []
        self.expected_launches: dict[object, str] = {}
        self.owned_processes: dict[object, Mock] = {}

    def deny(self, operation: str, *_args, **_kwargs):
        self.events.append({"event": "blocked", "operation": operation})
        raise AssertionError(f"SVG fixture blocked real operation: {operation}")

    def expect_blocked(self, operation: str) -> None:
        """Declare an intentional negative test without erasing its evidence."""
        self.expected_blocks.append(operation)

    def expect_launch(self, app, *, kind: str) -> None:
        assert app.process not in self.expected_launches
        self.expected_launches[app.process] = kind

    def execute_command(self, controller, command, kind, *, config) -> None:
        expected = self.expected_launches.pop(controller, None)
        session = controller.app.execution.session
        if (expected != kind or session.kind != kind or not session.token
                or tuple(command) != session.command or config is not session.config):
            self.deny("ProcessController.execute_command")
        self.events.append({"event": "fixture-launch", "kind": kind, "command": list(command)})

    def own_process(self, lifecycle, process: Mock) -> None:
        assert isinstance(process, Mock), "SVG processes must be explicit mocks"
        assert lifecycle.process is None
        assert lifecycle not in self.owned_processes
        self.owned_processes[lifecycle] = process
        lifecycle.process = process
        self.events.append({"event": "fixture-process", "pid": process.pid})

    def stop(self, lifecycle, *, closing=False, expected_process=None) -> StopResult:
        process = lifecycle.process
        if process is None:
            self.events.append({"event": "fixture-stop", "pid": None, "closing": closing})
            return StopResult(None, None)
        if self.owned_processes.get(lifecycle) is not process:
            self.deny("ProcessLifecycle.stop (unregistered process)")
        if expected_process is not None and process is not expected_process:
            self.events.append({"event": "fixture-stale-stop", "closing": closing})
            return StopResult(None, None)
        self.events.append({"event": "fixture-stop", "pid": process.pid, "closing": closing})
        process.returncode = 0
        process.poll.return_value = 0
        lifecycle.process = None
        del self.owned_processes[lifecycle]
        return StopResult(process.pid, 0)

    def assert_clean(self) -> None:
        blocked = [event["operation"] for event in self.events if event["event"] == "blocked"]
        assert blocked == self.expected_blocks, f"Unexpected SVG operations: {blocked}"
        assert not self.expected_launches, "Expected SVG launch was not exercised"
        assert not self.owned_processes, "SVG process survived application cleanup"


@pytest.fixture(autouse=True)
def svg_operations(request, monkeypatch, offline_machine_environment, offline_tui_version_check):
    """Fence construction, startup workers, rendering and unmount, including app.run()."""
    from acprof.platform import Environment
    from acprof.tui.images.controller import ImageController
    from acprof.tui.run.diagnostics import PreflightCheck
    from acprof.tui.run.environment_checks import ENVIRONMENT_CHECK_IDS
    from acprof.tui.run.preflight_controller import PreflightController
    from acprof.tui.run.process import ProcessLifecycle
    from acprof.tui.run.process_controller import ProcessController
    from acprof.tui.updates import UpdateController

    guard = SvgOperations()

    def complete_preflight(controller, config, token):
        guard.events.append({"event": "fixture-preflight"})
        checks = tuple(
            PreflightCheck(name, "ok", "fixture machine check", check_id=name, source="fixture")
            for name in ENVIRONMENT_CHECK_IDS
        )
        controller.show_quick_check([], "", token, True, checks, "")

    def skip_version_lookup(controller):
        # Version networking is not a scene input. Retain the explicit startup
        # request as evidence, while the HTTP guard catches accidental bypasses.
        guard.events.append({"event": "fixture-version-check"})
        controller.started = True

    def execute_command(controller, command, kind, *, config):
        guard.execute_command(controller, command, kind, config=config)

    def stop(lifecycle, *, closing=False, expected_process=None):
        return guard.stop(lifecycle, closing=closing, expected_process=expected_process)

    def deny(operation):
        return lambda *args, **kwargs: guard.deny(operation, *args, **kwargs)

    async def deny_async_http(*args, **kwargs):
        guard.deny("httpx.AsyncClient.send", *args, **kwargs)

    monkeypatch.setattr("acprof.platform.detect_environment", lambda: Environment("native_linux"))
    monkeypatch.setattr("acprof.tui.run.diagnostics.detect_environment", lambda: Environment("native_linux"))
    monkeypatch.setattr(PreflightController, "execute_quick_check", complete_preflight)
    monkeypatch.setattr(PreflightController, "_execute_quick_check_worker", deny("real preflight worker"))
    monkeypatch.setattr("acprof.tui.run.preflight_controller.quick_preflight", deny("quick_preflight"))
    monkeypatch.setattr("acprof.tui.run.preflight_controller.scan_environment", deny("scan_environment"))
    monkeypatch.setattr(UpdateController, "start_check", skip_version_lookup)
    monkeypatch.setattr(ProcessController, "execute_command", execute_command)
    monkeypatch.setattr(ProcessLifecycle, "start", deny("ProcessLifecycle.start"))
    monkeypatch.setattr(ProcessLifecycle, "reply", deny("ProcessLifecycle.reply"))
    monkeypatch.setattr(ProcessLifecycle, "stop", stop)
    monkeypatch.setattr(ImageController, "refresh_images", deny("ImageController.refresh_images"))
    monkeypatch.setattr("subprocess.Popen", deny("subprocess.Popen"))
    monkeypatch.setattr("os.kill", deny("os.kill"))
    monkeypatch.setattr("os.killpg", deny("os.killpg"))
    monkeypatch.setattr("httpx.Client.send", deny("httpx.Client.send"))
    monkeypatch.setattr("httpx.AsyncClient.send", deny_async_http)
    monkeypatch.setattr("socket.socket.connect", deny("socket.socket.connect"))
    monkeypatch.setattr("socket.socket.connect_ex", deny("socket.socket.connect_ex"))
    try:
        yield guard
    finally:
        output = os.environ.get("ACPROF_SNAPSHOT_OUTPUT_DIR")
        if output:
            evidence = Path(output) / "guards" / f"{request.node.name}.json"
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text(json.dumps({
                "node": request.node.nodeid,
                "blocked_count": sum(event["event"] == "blocked" for event in guard.events),
                "expected_blocks": guard.expected_blocks,
                "events": guard.events,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Worker exceptions may be presented as UI diagnostics. They still fail
        # the test here, after the full app.run()/unmount lifetime.
        guard.assert_clean()


@pytest.fixture
def snap_compare(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Compare headless Textual output against reviewed, checked-in SVG evidence."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("NO_COLOR", raising=False)

    def compare(app: App, *, terminal_size: tuple[int, int], run_before=None) -> bool:
        # Fix the display input before rendering; compare every output cell as usual.
        monkeypatch.setattr(app, "title", SNAPSHOT_TITLE)

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
        output = os.environ.get("ACPROF_SNAPSHOT_OUTPUT_DIR")
        folder = Path(output) if output else tmp_path
        actual_path = folder / (f"{node.name}.svg" if output else "actual.svg")
        # Explicit evidence requests retain matching captures too, so every
        # review image comes from this run rather than a copied Golden.
        if output or actual != expected:
            folder.mkdir(parents=True, exist_ok=True)
            actual_path.write_text(actual, encoding="utf-8")
        if actual != expected:
            diff = "".join(difflib.unified_diff(expected.splitlines(keepends=True),
                                                actual.splitlines(keepends=True),
                                                fromfile=str(baseline), tofile=str(actual_path), n=2))
            pytest.fail(f"SVG snapshot changed (review output before updating): {actual_path}\n{diff[:3000]}")
        return True
    return compare
