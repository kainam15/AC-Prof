"""Guard real operations during fixture construction, rendering and unmount."""

from __future__ import annotations

import asyncio
import json
from contextlib import ExitStack
from pkgutil import resolve_name
from unittest.mock import Mock, patch

import pytest
from textual.widgets import Button

from acprof.tui.run.process import ProcessLifecycle, StopResult
from tests.visual_png.fixture_app import VisualFixtureApp, isolate_fixture, stop_fixture_process

pytestmark = pytest.mark.visual_png


@pytest.mark.parametrize("target", [
    "subprocess.Popen", "os.kill", "os.killpg",
    "socket.socket.connect", "socket.socket.connect_ex", "socket.getaddrinfo",
    "acprof.tui.run.diagnostics.quick_preflight",
    "acprof.tui.run.preflight_controller.quick_preflight",
    "acprof.tui.run.preflight_controller.scan_environment",
    "acprof.tui.settings.environment.quick_preflight",
])
def test_guard_records_caught_violations_and_restores_operations(target: str) -> None:
    # A second outer mock ensures a broken guard test cannot perform real work.
    with patch(target) as original, ExitStack() as guards:
        isolation = guards.enter_context(isolate_fixture())
        with pytest.raises(RuntimeError, match="must not execute a real operation"):
            resolve_name(target)()
        with pytest.raises(AssertionError, match="PNG fixture blocked real operations"):
            guards.close()
        assert resolve_name(target) is original
        original.assert_not_called()
    assert isolation.blocked_calls == [target]


def test_fixture_instance_entry_points_record_violations(tmp_path) -> None:
    with ExitStack() as guards:
        isolation = guards.enter_context(isolate_fixture())
        app = VisualFixtureApp("form", settings_path=tmp_path / "settings.json")
        for operation in (
            app.preflight.execute_quick_check, app.process.execute_command,
            app.images.refresh_images, app.execution.lifecycle.start, app.updates.start_check,
        ):
            with pytest.raises(RuntimeError, match="must not execute a real operation"):
                operation()
        with pytest.raises(AssertionError, match="PNG fixture blocked real operations"):
            guards.close()
    assert isolation.blocked_calls == [
        "preflight.execute_quick_check", "process.execute_command", "images.refresh_images",
        "lifecycle.start", "updates.start_check",
    ]


def test_guard_is_active_before_app_construction(tmp_path) -> None:
    import os

    with patch("os.kill") as original:
        with patch("acprof.tui.app.AcprofTui.__init__", side_effect=lambda *a, **kw: os.kill(12345, 0)):
            with pytest.raises(AssertionError, match="PNG fixture blocked real operations"):
                with isolate_fixture() as isolation:
                    VisualFixtureApp("form", settings_path=tmp_path / "settings.json")
        original.assert_not_called()
    assert isolation.blocked_calls == ["os.kill"]


def test_fake_stop_preserves_ownership_and_clears_matching_process() -> None:
    lifecycle = ProcessLifecycle()
    process = Mock(pid=None, returncode=None)
    process.poll.return_value = None
    lifecycle.process = process
    assert stop_fixture_process(lifecycle, expected_process=object()) == StopResult(None, None)
    assert lifecycle.process is process
    assert stop_fixture_process(lifecycle, closing=True, expected_process=process) == StopResult(None, 0)
    assert lifecycle.process is None
    assert process.returncode == process.poll() == 0
    assert stop_fixture_process(lifecycle, closing=True) == StopResult(None, None)


def test_fake_stop_rejects_non_fixture_pid_without_signalling() -> None:
    lifecycle = ProcessLifecycle()
    process = Mock(pid=12345, returncode=None)
    lifecycle.process = process
    with patch("os.killpg") as original:
        with pytest.raises(AssertionError, match="PNG fixture blocked real operations"):
            with isolate_fixture() as isolation:
                stop_fixture_process(lifecycle, closing=True)
        original.assert_not_called()
    assert lifecycle.process is process
    assert isolation.blocked_calls == ["lifecycle.stop(non-fixture)"]


@pytest.mark.parametrize("scene", ["measuring", "docker-frontend"])
async def test_active_fixture_unmount_releases_only_synthetic_process(tmp_path, scene: str) -> None:
    with isolate_fixture() as isolation:
        app = VisualFixtureApp(scene, settings_path=tmp_path / "settings.json")
        async with app.run_test(size=(120, 36)):
            await asyncio.wait_for(app.scene_ready.wait(), timeout=5)
            process = app.execution.lifecycle.process
            assert isinstance(process, Mock)
            assert process.pid is None
            assert process.poll() is None
            assert not app.query_one("#stop-run", Button).disabled
        assert app.execution.lifecycle.process is None
        assert process.returncode == process.poll() == 0
    assert isolation.blocked_calls == []


def test_guard_journal_persists_a_violation_before_it_is_raised(tmp_path) -> None:
    report = tmp_path / "guard.json"
    with ExitStack() as guards:
        isolation = guards.enter_context(isolate_fixture(report_path=report))
        assert json.loads(report.read_text()) == {"state": "active", "closed": False, "blocked_calls": []}
        with pytest.raises(RuntimeError, match="intentional-journal-check"):
            isolation.forbid("intentional-journal-check")
        expected = {"state": "blocked", "closed": False, "blocked_calls": ["intentional-journal-check"]}
        assert json.loads(report.read_text()) == expected
        with pytest.raises(AssertionError, match="PNG fixture blocked real operations"):
            guards.close()
    assert json.loads(report.read_text()) == expected


def test_guard_journal_marks_success_only_after_scope_exit(tmp_path) -> None:
    report = tmp_path / "guard.json"
    with isolate_fixture(report_path=report):
        assert not json.loads(report.read_text())["closed"]
    assert json.loads(report.read_text()) == {"state": "closed", "closed": True, "blocked_calls": []}


def test_guard_journal_does_not_claim_success_after_an_exception(tmp_path) -> None:
    report = tmp_path / "guard.json"

    def fail_in_fixture() -> None:
        with isolate_fixture(report_path=report):
            raise ValueError("fixture failure")

    with pytest.raises(ValueError, match="fixture failure"):
        fail_in_fixture()
    assert json.loads(report.read_text()) == {"state": "failed", "closed": False, "blocked_calls": []}


def test_fixture_main_writes_the_requested_guard_journal(tmp_path, monkeypatch) -> None:
    import sys

    from tests.visual_png import fixture_app

    report = tmp_path / "guard.json"
    monkeypatch.setenv("ACPROF_PNG_GUARD_REPORT", str(report))
    monkeypatch.setattr(sys, "argv", [
        "fixture_app.py", "--scenario", "form", "--fixture-dir", str(tmp_path / "fixture"),
    ])
    monkeypatch.setattr(fixture_app.VisualFixtureApp, "run", lambda _app: 0)
    assert fixture_app.main() == 0
    assert json.loads(report.read_text()) == {"state": "closed", "closed": True, "blocked_calls": []}

