"""Automatic checks use real TUI timers/workers and isolated diagnostic boundaries."""
import asyncio
import subprocess
import threading
from dataclasses import replace
from unittest.mock import Mock, patch

import pytest
from textual.widgets import Button, Input, Select, Static, TabbedContent
from textual.worker import Worker, WorkerState

from acprof.experiments.configuration import RunConfig
from acprof.host.container_lifecycle import ContainerCleanupError
from acprof.host.run_state import MeasurementLock
from acprof.tui.app import AcprofTui
from acprof.tui.logging.widget import SelectableLog
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.run.progress import ProgressSnapshot
from acprof.tui.run.stages import Stage
from tests.support.tui_waits import wait_for_condition


@pytest.fixture
def diagnostics(monkeypatch):
    def ready(config, **_kwargs):
        runtime = "process" if config.runtime == "process" else "docker"
        return [PreflightCheck("Runtime", "ok", "fixture", runtime_kind=runtime, check_id="runtime")]

    check = Mock(side_effect=ready)
    scan = Mock(return_value=[PreflightCheck("Docker", "ok", "fixture", check_id="docker")])
    monkeypatch.setattr("acprof.tui.run.preflight_controller.quick_preflight", check)
    monkeypatch.setattr("acprof.tui.run.preflight_controller.scan_environment", scan)
    return check, scan


def make_app(tmp_path, **changes):
    return AcprofTui(replace(RunConfig.smoke("fixture/model"), **{"runtime": "docker", **changes}),
                     settings_path=tmp_path / "settings.json")


async def ready(app, pilot):
    await wait_for_condition(pilot, lambda: not app.query_one("#start-run", Button).disabled,
                             description="current configuration ready to collect", timeout=5)


@pytest.mark.parametrize(("field", "value", "attribute"), [
    ("gpus", "on", "gpus"), ("cpuset-cpus", "0-1", "cpuset_cpus"),
    ("runtime", "process", "runtime"), ("cpus", "1,2", "cpus"),
    ("sniff-iface", "lo", "sniff_iface"), ("torch-profiler", True, "compute_profile_tool"),
])
async def test_relevant_edit_automatically_checks_without_repeating_inventory(tmp_path, diagnostics, field, value, attribute):
    check, scan = diagnostics
    app = make_app(tmp_path)
    async with app.run_test(size=(120, 30)) as pilot:
        await ready(app, pilot)
        app.query_one(f"#{field}").value = value
        await wait_for_condition(pilot, lambda: check.call_count == 2 and not app.preflight.check_running,
                                 description="automatic check of the edited configuration", timeout=3)
        await ready(app, pilot)
        assert getattr(check.call_args.args[0], attribute) == ("torch" if field == "torch-profiler" else value)
        scan.assert_called_once()


async def test_invalid_form_blocks_collection_and_shows_field_error_without_scanning(tmp_path, diagnostics):
    check, scan = diagnostics
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        await ready(app, pilot)
        app.query_one("#model", Input).value = ""
        await wait_for_condition(pilot, lambda: app.preflight.status.state == "unconfigured"
                                 and app.query_one("#start-run", Button).disabled,
                                 description="blank model stays pending and blocks start", timeout=2)
        assert not app.query_one("#model-error", Static).display
        assert app.query_one("#start-run", Button).disabled
        assert app.preflight.status.counts["error"] == 0
        app.experiments.request_run()
        await wait_for_condition(pilot, lambda: app.query_one("#model-error", Static).display,
                                 description="explicit attempt reports missing model", timeout=2)
        assert "模型 ID" in str(app.query_one("#model-error", Static).content)
        app.query_one("#gpus", Select).value = "on"
        await pilot.pause()
        check.assert_called_once()
        scan.assert_called_once()
        app.query_one("#model", Input).value = "fixture/repaired"
        await wait_for_condition(pilot, lambda: check.call_count == 2 and not app.preflight.check_running,
                                 description="automatic check after repairing the form", timeout=3)
        await ready(app, pilot)
        assert not app.query_one("#model-error", Static).display


async def debounce_elapsed(app):
    """Observe a complete debounce interval for negative scan assertions."""
    elapsed = asyncio.Event()
    app.set_timer(0.65, elapsed.set)
    await asyncio.wait_for(elapsed.wait(), 2)


async def test_burst_edits_merge_and_preserve_page_focus_input_and_silence(tmp_path, diagnostics):
    check, scan = diagnostics
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        await ready(app, pilot)
        app.configuration.show_run_form("advanced-form")
        field = app.query_one("#cpuset-cpus", Input)
        field.focus()
        await pilot.pause()
        with patch.object(app, "notify") as notify:
            for value in ("0", "0-1", "0-3"):
                field.value = value
                # Validate each edit in one event-loop turn so all three belong to the same burst.
                app.configuration.refresh_command_preview(notify=False)
                scheduled = app.preflight.auto_check_config
                assert scheduled is not None
                assert scheduled.cpuset_cpus == value
            check.assert_called_once()
            assert app.query_one("#start-run", Button).disabled
            await wait_for_condition(pilot, lambda: app.preflight.checked_config is not None
                                     and app.preflight.checked_config.cpuset_cpus == "0-3"
                                     and not app.preflight.check_running,
                                     description="completed preflight for the final edit", timeout=3)
            await ready(app, pilot)
            assert check.call_args.args[0].cpuset_cpus == "0-3"
            assert app.focused is field
            assert field.value == "0-3"
            assert app.query_one("#main-tabs", TabbedContent).active == "run-tab"
            assert len(app.screen_stack) == 1
            assert app.query_one("#run-log", SelectableLog).text == ""
            notify.assert_not_called()
        await debounce_elapsed(app)
        assert check.call_count == 2
        scan.assert_called_once()


async def test_workload_edits_only_refresh_the_plan(tmp_path, diagnostics):
    check, scan = diagnostics
    app = make_app(tmp_path)
    async with app.run_test(size=(120, 30)) as pilot:
        await ready(app, pilot)
        for name, value in (("warmup", "2"), ("repeat", "3"), ("repeat-in-window", "4")):
            app.query_one(f"#{name}", Input).value = value
        await wait_for_condition(pilot, lambda: "--repeat 3" in str(app.query_one("#command-preview", Static).content),
                                 description="updated workload command", timeout=2)
        await ready(app, pilot)
        await debounce_elapsed(app)
        check.assert_called_once()
        scan.assert_called_once()


@pytest.mark.parametrize("runtime", ["process", "auto"])
async def test_process_quota_edits_revalidate_resolved_runtime(tmp_path, diagnostics, runtime):
    check, scan = diagnostics
    check.side_effect = lambda *_args, **_kwargs: [
        PreflightCheck("Runtime", "ok", "fixture", runtime_kind="process", check_id="runtime"),
    ]
    app = make_app(tmp_path, runtime=runtime)
    async with app.run_test(size=(120, 30)) as pilot:
        await ready(app, pilot)
        app.query_one("#cpus", Input).value = "1"
        app.query_one("#mems", Input).value = "2"
        await wait_for_condition(pilot, lambda: check.call_count == 2 and not app.preflight.check_running,
                                 description="Process resource revalidation", timeout=3)
        await ready(app, pilot)
        assert (check.call_args.args[0].cpus, check.call_args.args[0].mems) == ("1", "2")
        scan.assert_called_once()


async def test_edits_during_worker_check_only_latest_snapshot_and_never_enable_stale_start(tmp_path, diagnostics):
    check, scan = diagnostics
    first_started, second_started = threading.Event(), threading.Event()
    first_release, second_release = threading.Event(), threading.Event()

    def controlled(config, **_kwargs):
        started, release = (first_started, first_release) if config.gpus == "off" else (second_started, second_release)
        started.set()
        if not release.wait(5):
            raise TimeoutError("test did not release diagnostic")
        return [PreflightCheck("Runtime", "ok", "fixture", runtime_kind="docker")]

    check.side_effect = controlled
    app = make_app(tmp_path)
    try:
        async with app.run_test(size=(120, 30)) as pilot:
            await wait_for_condition(pilot, first_started.is_set, description="first diagnostic", timeout=3)
            original_token = app.preflight.check_request
            app.configuration.apply_config(replace(app.configuration.initial_config, gpus="on", sniff_iface="lo"))
            app.configuration.apply_config(replace(app.configuration.initial_config, gpus="on", sniff_iface="eth1"))
            await debounce_elapsed(app)
            check.assert_called_once()
            app.activate_tab("monitor-tab")
            first_release.set()
            await wait_for_condition(pilot, second_started.is_set, description="latest diagnostic", timeout=3)
            assert check.call_count == 2
            assert check.call_args.args[0].sniff_iface == "eth1"
            assert app.query_one("#start-run", Button).disabled
            assert app.preflight.selected_runtime == ""
            assert app.preflight.collection_state == "checking"
            assert app.preflight.display_state == "checking"
            assert not app.preflight.inventory_running, "cached inventory is not being rescanned"
            assert app.query_one("#environment-pending").display
            app.preflight.show_quick_check([], "late duplicate", original_token)
            assert app.preflight.check_running
            assert "late duplicate" not in app.preflight.error
            second_release.set()
            await ready(app, pilot)
            assert app.query_one("#main-tabs", TabbedContent).active == "monitor-tab"
            assert app.preflight.checked_config.sniff_iface == "eth1"
            scan.assert_called_once()
    finally:
        first_release.set()
        second_release.set()


@pytest.mark.parametrize(("field", "invalid", "valid"), [("repeat", "oops", "3"), ("cpuset-cpus", "0-", "0-1")])
async def test_malformed_field_skips_checks_then_recovers(tmp_path, diagnostics, field, invalid, valid):
    check, scan = diagnostics
    app = make_app(tmp_path)
    async with app.run_test(size=(120, 30)) as pilot:
        await ready(app, pilot)
        app.query_one(f"#{field}", Input).value = invalid
        app.query_one("#gpus", Select).value = "on"
        await wait_for_condition(pilot, lambda: app.query_one(f"#{field}-error", Static).display,
                                 description="malformed field feedback", timeout=2)
        await debounce_elapsed(app)
        assert app.preflight.collection_state == "incomplete"
        assert app.query_one("#start-run", Button).disabled
        check.assert_called_once()
        app.query_one(f"#{field}", Input).value = valid
        await wait_for_condition(pilot, lambda: check.call_count == 2 and not app.preflight.check_running,
                                 description="repaired configuration check", timeout=3)
        await ready(app, pilot)
        assert not app.query_one(f"#{field}-error", Static).display
        scan.assert_called_once()


@pytest.mark.parametrize("failure", [OSError("fixture unavailable"), subprocess.TimeoutExpired("fixture", 1)])
async def test_failed_auto_check_stays_failed_until_manual_retry(tmp_path, diagnostics, failure):
    check, scan = diagnostics
    app = make_app(tmp_path)
    completed = asyncio.Queue()

    def observe_preflight(message):
        if (isinstance(message, Worker.StateChanged) and message.worker.group == "preflight"
                and message.state == WorkerState.SUCCESS):
            completed.put_nowait(app.preflight.checked_config)

    async with app.run_test(size=(120, 30), message_hook=observe_preflight) as pilot:
        initial = await asyncio.wait_for(completed.get(), timeout=5)
        assert initial is not None
        assert initial.gpus == "off"
        await ready(app, pilot)
        check.side_effect = failure
        app.query_one("#gpus", Select).value = "on"
        edited = await asyncio.wait_for(completed.get(), timeout=3)
        assert edited is not None
        assert edited.gpus == "on"
        assert app.preflight.collection_state == "failed"
        assert type(failure).__name__ in str(app.query_one("#preflight-run-reason", Button).label)
        assert app.query_one("#start-run", Button).disabled
        app.query_one("#repeat", Input).value = "3"
        await debounce_elapsed(app)
        assert check.call_count == 2
        check.side_effect = None
        check.return_value = [PreflightCheck("Runtime", "ok", "fixture", runtime_kind="docker", check_id="runtime")]
        app.preflight.quick_check()
        assert await asyncio.wait_for(completed.get(), timeout=3) == edited
        await ready(app, pilot)
        assert check.call_count == 3
        assert scan.call_count == 2


@pytest.mark.parametrize("phase", ["preparation", "measurement", "cleanup"])
async def test_pending_check_waits_for_exclusive_work_to_finish(tmp_path, diagnostics, phase):
    check, scan = diagnostics
    app = make_app(tmp_path)
    async with app.run_test(size=(120, 30)) as pilot:
        await ready(app, pilot)
        app.configuration.apply_config(replace(app.configuration.initial_config, gpus="on"))
        app.execution.session.kind = "run"
        app.execution.session.snapshot = ProgressSnapshot(
            measurement_active=phase == "measurement",
            stage={"preparation": Stage.PREPARING_IMAGE, "measurement": Stage.MEASURING,
                   "cleanup": Stage.CLEANUP_INCOMPLETE}[phase],
        )
        app.set_busy(True)
        try:
            await debounce_elapsed(app)
            assert app.preflight.auto_check_timer is None
            assert app.query_one("#start-run", Button).disabled
            check.assert_called_once()
            scan.assert_called_once()
        finally:
            app.execution.session.kind = ""
            app.execution.session.snapshot = ProgressSnapshot()
            app.set_busy(False)
        await wait_for_condition(pilot, lambda: check.call_count == 2 and not app.preflight.check_running,
                                 description="deferred diagnostic after exclusive work", timeout=3)
        await ready(app, pilot)


async def test_quit_cancels_pending_timer_and_drops_queued_timer_callback(tmp_path, diagnostics):
    check, scan = diagnostics
    app = make_app(tmp_path)
    async with app.run_test(size=(120, 30)) as pilot:
        await ready(app, pilot)
        app.configuration.apply_config(replace(app.configuration.initial_config, gpus="on"))
        assert app.preflight.auto_check_timer is not None
        app.action_request_quit()
    assert app.preflight.auto_check_timer is None
    assert app.preflight.auto_check_config is None
    with patch.object(app, "query_one", side_effect=AssertionError("late widget access")):
        app.preflight.auto_check_elapsed()
    check.assert_called_once()
    scan.assert_called_once()


async def test_quit_during_inventory_skips_remaining_probes(tmp_path, diagnostics):
    check, scan = diagnostics
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def inventory(**_kwargs):
        started.set()
        if not release.wait(5):
            raise TimeoutError("test did not release inventory")
        return []

    scan.side_effect = inventory
    app = make_app(tmp_path)
    execute = app.preflight._execute_quick_check_worker

    def run_worker(*args, **kwargs):
        try:
            return execute(*args, **kwargs)
        finally:
            finished.set()

    try:
        with patch.object(app.preflight, "_execute_quick_check_worker", side_effect=run_worker):
            async with app.run_test(size=(80, 24)) as pilot:
                await wait_for_condition(pilot, started.is_set, description="inventory start", timeout=3)
                worker = next(worker for worker in app.workers if worker.name == "execute_quick_check")
                app.action_request_quit()
            assert worker.is_cancelled
            release.set()
            assert await asyncio.to_thread(finished.wait, 2)
            check.assert_not_called()
    finally:
        release.set()


async def test_configuration_edit_cannot_bypass_unconfirmed_inventory_cleanup(tmp_path, diagnostics):
    check, scan = diagnostics
    scan.side_effect = ContainerCleanupError("a" * 64, [], final_state="present")
    app = make_app(tmp_path)
    async with app.run_test(size=(120, 30)) as pilot:
        await wait_for_condition(pilot, lambda: app.preflight.check_completed,
                                 description="unconfirmed inventory cleanup", timeout=3)
        app.query_one("#gpus", Select).value = "on"
        await debounce_elapsed(app)
        assert app.query_one("#start-run", Button).disabled
        assert "ContainerCleanupError" in app.preflight.preflight_run_reason()
        check.assert_not_called()
        scan.assert_called_once()
        scan.side_effect = None
        scan.return_value = []
        app.preflight.quick_check()
        await ready(app, pilot)
        check.assert_called_once()


async def test_external_measurement_blocks_auto_probe_without_discarding_inventory(tmp_path, diagnostics):
    check, scan = diagnostics
    app = make_app(tmp_path)
    async with app.run_test(size=(120, 30)) as pilot:
        await ready(app, pilot)
        inventory, completed_at = app.preflight.environment_checks, app.preflight.completed_at
        with MeasurementLock():
            app.configuration.apply_config(replace(app.configuration.initial_config, gpus="on"))
            await wait_for_condition(pilot, lambda: bool(app.preflight.error) and not app.preflight.check_running,
                                     description="measurement lock rejects automatic check", timeout=3)
            assert app.query_one("#start-run", Button).disabled
            assert "测量锁" in app.preflight.preflight_run_reason()
            assert app.preflight.environment_checks == inventory
            assert app.preflight.completed_at == completed_at
            check.assert_called_once()
            scan.assert_called_once()
        app.preflight.quick_check()
        await ready(app, pilot)
        assert check.call_count == scan.call_count == 2


@pytest.mark.parametrize(("language", "size"), [("zh", (80, 24)), ("en", (120, 30))])
async def test_global_status_distinguishes_invalid_pending_checking_limited_and_failed(tmp_path, diagnostics, language, size):
    check, _scan = diagnostics
    started, release = threading.Event(), threading.Event()

    def limited(*_args, **_kwargs):
        started.set()
        if not release.wait(5):
            raise TimeoutError("test did not release diagnostic")
        return [PreflightCheck("optional", "warn", "fixture unavailable")]

    app = make_app(tmp_path, repeat=3)
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    try:
        async with app.run_test(size=size) as pilot:
            await ready(app, pilot)
            status = app.query_one("#preflight-run-reason", Button)
            assert ("已就绪" if language == "zh" else "Ready") in str(status.label)
            assert not status.display
            assert not app.query_one("#start-run", Button).disabled
            app.query_one("#model", Input).value = ""
            await wait_for_condition(pilot, lambda: app.preflight.collection_state == "unconfigured",
                                     description="normal blank draft", timeout=2)
            assert app.preflight.status.counts["error"] == 0
            app.experiments.request_run()
            await wait_for_condition(pilot, lambda: app.preflight.collection_state == "incomplete",
                                     description="incomplete form status", timeout=2)
            assert app.preflight.status.counts["error"] == 1
            assert app.query_one("#start-run", Button).disabled
            check.side_effect = limited
            app.configuration.apply_config(replace(app.configuration.initial_config, gpus="on"))
            assert app.preflight.collection_state == "pending"
            assert ("待检查" if language == "zh" else "Pending") in str(status.label)
            await wait_for_condition(pilot, started.is_set, description="checking status", timeout=3)
            assert app.preflight.collection_state == "checking"
            assert ("正在检查环境" if language == "zh" else "Checking the environment") in str(status.label)
            release.set()
            await ready(app, pilot)
            assert app.preflight.collection_state == "limited"
            assert ("必要预检已通过" if language == "zh" else "Required checks passed") in str(status.label)
            check.side_effect = OSError("fixture failure")
            app.preflight.quick_check()
            await wait_for_condition(pilot, lambda: app.preflight.collection_state == "failed",
                                     description="failed diagnostic status", timeout=3)
            assert "fixture failure" in str(status.label)
            assert app.query_one("#start-run", Button).disabled
            assert len(app.screen_stack) == 1
    finally:
        release.set()



async def test_external_permission_change_invalidates_launch_and_debounces_recheck(tmp_path, diagnostics, monkeypatch):
    signature = Mock(return_value=("before",))
    monkeypatch.setattr("acprof.tui.run.preflight_controller.permission_signature", signature)
    check, scan = diagnostics
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        await ready(app, pilot)
        signature.return_value = ("changed",)
        assert not app.preflight.allow_collection()
        app.preflight.check_permission_changes()
        await wait_for_condition(pilot, lambda: check.call_count == 2 and not app.preflight.check_running,
                                 description="permission change rechecked", timeout=4)
        assert app.preflight.allow_collection()
        scan.assert_called_once()


async def test_idle_permission_change_rechecks_without_user_action(tmp_path, diagnostics, monkeypatch):
    signature = Mock(return_value=("before",))
    monkeypatch.setattr("acprof.tui.run.preflight_controller.permission_signature", signature)
    _check, scan = diagnostics
    app = make_app(tmp_path)
    app.PERMISSION_REFRESH_INTERVAL = 0.1
    async with app.run_test(size=(80, 24)) as pilot:
        await ready(app, pilot)
        signature.return_value = ("changed",)
        await wait_for_condition(pilot, lambda: app.preflight.checked_permissions == ("changed",) and not app.preflight.check_running,
                                 description="idle permission change rechecked", timeout=4)
        await ready(app, pilot)
        assert app.preflight.allow_collection()
        scan.assert_called_once()
