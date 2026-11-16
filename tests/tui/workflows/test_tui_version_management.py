"""Version manager actions use real widgets and controlled installer completion."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from textual.widgets import Button, Select, Static

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from acprof.tui.updates import UpdateController
from acprof.updates.installation import Installation
from acprof.updates.worker import UpdatePlan, atomic_json
from tests.support.tui_waits import wait_for_condition


@pytest.fixture
def version_manager(tmp_path, monkeypatch):
    monkeypatch.setattr("acprof.tui.run.preflight_controller.quick_preflight", lambda *a, **k: [])
    monkeypatch.setattr("acprof.tui.updates.__version__", "1.0.0")
    installation = Installation("uv tool", tmp_path / "tools/acprof", tmp_path / "tools/acprof/bin/python",
                                tmp_path / "bin/acprof", Path("/usr/bin/uv"))
    monkeypatch.setattr("acprof.tui.updates.detect_installation", lambda: installation)

    async def respond(_transport, request):
        return httpx.Response(200, json={"releases": {
            "1.0.0": [{"yanked": False}], "1.1.0a1": [{"yanked": False}],
        }})

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    app = AcprofTui(RunConfig.smoke(""), settings_path=tmp_path / "settings.json")
    launches = []
    active = []

    def plan(install, release, directory):
        return UpdatePlan(str(install.prefix), str(install.entrypoint), str(install.manager), "/usr/bin/python3",
                          str(release.current), str(release.latest), release.channel,
                          str(directory), str(directory / "jobs/fixture"))

    def launch(update):
        assert not active, "duplicate installation"
        launches.append(update)
        active.append(True)
        job = Path(update.job_dir)
        job.mkdir(parents=True)
        (job / "install.log").write_text("Resolved dependencies\nDownloading acprof\n")
        atomic_json(app.updates.directory / "status.json", {
            "status": "running", "phase": "download", "log": str(job / "install.log"), "job_dir": str(job),
        })
        return Mock(poll=lambda: None)

    monkeypatch.setattr("acprof.tui.updates.service.make_plan", plan)
    monkeypatch.setattr("acprof.tui.updates.service.launch_update", launch)
    monkeypatch.setattr("acprof.tui.updates.service.update_active", lambda directory: bool(active))
    return app, launches, active


@pytest.mark.parametrize(("language", "size"), [("zh", (80, 24)), ("en", (120, 30))])
async def test_keyboard_confirmation_progress_navigation_and_restart(version_manager, monkeypatch, language, size):
    app, launches, active = version_manager
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size, notifications=True) as pilot:
        await wait_for_condition(pilot, lambda: app.updates.release is not None, timeout=10, description="version check finished")
        app.activate_tab("settings-tab")
        await pilot.pause()
        channel = app.query_one("#update-channel", Select)
        channel.focus()
        await pilot.press("enter", "down", "enter")
        await pilot.pause()
        assert app.updates.channel == "preview"
        assert app.updates.effective_channel == "stable"
        assert not (app.updates.directory / "channel.json").exists()
        button = app.query_one("#version-apply", Button)
        button.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert "1.0.0 → 1.1.0a1" in str(app.screen.query_one("#confirm-message Static", Static).content)
        app.updates.request_install()  # A pending confirmation also excludes duplicates.
        assert len(app.screen_stack) == 2
        app.screen.query_one("#confirm-yes", Button).focus()
        await pilot.press("enter")
        await wait_for_condition(pilot, lambda: bool(launches), timeout=10, description="installer launched")
        assert len(launches) == 1
        assert app.updates.installing
        assert app.query_one("#start-run", Button).disabled
        assert app.query_one("#version-apply", Button).disabled
        app.updates.confirm_install(True, launches[0])
        assert len(launches) == 1
        app.activate_tab("monitor-tab")
        await pilot.pause()
        app.activate_tab("settings-tab")
        await pilot.pause()
        text = str(app.query_one("#version-status", Static).content)
        assert ("总进度未知" if language == "zh" else "Total progress unknown") in text
        app.updates.show_log()
        await pilot.pause()
        assert "Downloading acprof" in str(app.screen.query_one("#confirm-message Static", Static).content)
        await pilot.press("escape")
        atomic_json(app.updates.directory / "channel.json", {"channel": "preview"})
        atomic_json(app.updates.directory / "status.json", {"status": "success", "phase": "complete"})
        active.clear()
        await wait_for_condition(pilot, lambda: not app.updates.installing, timeout=10, description="installation verified")
        assert app.updates.effective_channel == "preview"
        assert app.updates.needs_restart
        assert app.query_one("#start-run", Button).disabled
        restart = app.query_one("#version-restart", Button)
        assert restart.display
        assert not restart.disabled
        monkeypatch.setattr("acprof.tui.updates.os.isatty", lambda descriptor: True)
        restart.focus()
        await pilot.press("enter")
    assert app.updates.restart_command == [str(app.updates.installation.entrypoint), "tui"]


async def test_downgrade_cancel_and_recheck_admission(version_manager):
    app, launches, active = version_manager
    async with app.run_test(size=(80, 24)) as pilot:
        await wait_for_condition(pilot, lambda: app.updates.release is not None, timeout=10, description="version check finished")
        from packaging.version import Version
        app.updates.current = Version("1.1.0rc1")
        app.updates.release = replace(app.updates.release, current=app.updates.current)
        app.updates.effective_channel = "preview"
        app.updates.request_install()
        await pilot.pause()
        assert str(app.screen.query_one("#confirm-yes", Button).label) == "降级"
        assert "1.1.0rc1 → 1.0.0" in str(app.screen.query_one("#confirm-message Static", Static).content)
        await pilot.press("escape")
        assert not launches
        assert not active
        app.updates.request_install()
        await pilot.pause()
        app.preflight.check_running = True
        app.screen.query_one("#confirm-yes", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert not launches
        app.preflight.check_running = False


@pytest.mark.parametrize("busy", ["preflight", "read", "maintenance", "measurement", "run"])
async def test_busy_tasks_reject_installation(version_manager, monkeypatch, busy):
    app, launches, _ = version_manager
    async with app.run_test(size=(80, 24)) as pilot:
        await wait_for_condition(pilot, lambda: app.updates.release is not None, timeout=10, description="version check finished")
        app.updates.select_channel("preview")
        if busy == "preflight":
            app.preflight.check_running = True
        elif busy == "read":
            app.read_jobs.add("fixture")
        elif busy == "maintenance":
            app.images.operation = "refresh"
        elif busy == "measurement":
            app.execution.session.snapshot = replace(app.execution.session.snapshot, measurement_active=True)
        else:
            app.execution.session.kind = "backfill"
        app.updates.install_state = {"log": "/fixture/log"}
        app.updates.refresh_controls()
        assert app.query_one("#version-apply", Button).disabled
        assert app.query_one("#version-log", Button).disabled
        with monkeypatch.context() as context:
            context.setattr(Path, "open", lambda *a, **k: pytest.fail("busy task must prevent log I/O"))
            app.updates.show_log()
        app.updates.request_install()
        await pilot.pause()
        assert not launches
        assert len(app.screen_stack) == 1
        app.preflight.check_running = False
        app.read_jobs.clear()
        app.images.operation = ""
        app.execution.session.kind = ""
        app.execution.session.snapshot = replace(app.execution.session.snapshot, measurement_active=False)


async def test_applied_channel_survives_new_controller(version_manager):
    app, _, _ = version_manager
    async with app.run_test(size=(80, 24)) as pilot:
        await wait_for_condition(pilot, lambda: app.updates.release is not None, timeout=10, description="version check finished")
        atomic_json(app.updates.directory / "channel.json", {"channel": "preview"})
        root = app.updates.directory.parent
        app.updates = UpdateController(app, state_root=root)
        app.updates.start_check()
        await wait_for_condition(pilot, lambda: app.updates.release is not None, timeout=10, description="restored channel check")
        assert app.updates.channel == "preview"
        assert str(app.updates.release.latest) == "1.1.0a1"


async def test_failed_install_reports_recovery_and_keeps_channel(version_manager):
    app, launches, active = version_manager
    async with app.run_test(size=(80, 24)) as pilot:
        await wait_for_condition(pilot, lambda: app.updates.release is not None, timeout=10, description="checked")
        app.updates.select_channel("preview")
        app.updates.request_install()
        await pilot.pause()
        app.screen.query_one("#confirm-yes", Button).focus()
        await pilot.press("enter")
        await wait_for_condition(pilot, lambda: bool(launches), timeout=10, description="install started")
        atomic_json(app.updates.directory / "status.json", {
            "status": "failed", "phase": "complete", "error_kind": "permission",
            "error": "Permission denied: tool path", "rollback": "restored",
        })
        active.clear()
        await wait_for_condition(pilot, lambda: not app.updates.installing, timeout=10, description="recovery finished")
        assert app.updates.effective_channel == "stable"
        assert not (app.updates.directory / "channel.json").exists()
        text = str(app.query_one("#version-status", Static).content)
        assert "安装权限不足" in text
        assert "原版本已恢复并验证" in text
        assert "Permission denied: tool path" in text
        assert app.updates.needs_restart
        assert not app.query_one("#version-restart", Button).disabled


async def test_unsupported_source_has_copyable_guidance(version_manager, monkeypatch):
    app, launches, _ = version_manager
    clipboard = []
    monkeypatch.setattr(app, "copy_to_clipboard", clipboard.append)
    async with app.run_test(size=(80, 24)) as pilot:
        await wait_for_condition(pilot, lambda: app.updates.release is not None, timeout=10, description="checked")
        app.updates.installation = replace(app.updates.installation, kind="source", reason="source")
        app.updates.select_channel("preview")
        app.activate_tab("settings-tab")
        await pilot.pause()
        assert app.query_one("#version-apply", Button).disabled
        app.updates.request_install()
        assert not launches
        app.query_one("#version-manual", Button).focus()
        await pilot.press("enter")
        assert clipboard == ["uv tool install acprof --prerelease allow"]


async def test_interrupted_state_blocks_actions_and_retains_log(version_manager):
    app, launches, _ = version_manager
    async with app.run_test(size=(80, 24)) as pilot:
        await wait_for_condition(pilot, lambda: app.updates.release is not None, timeout=10, description="checked")
        atomic_json(app.updates.directory / "status.json", {
            "status": "running", "phase": "install", "job_dir": "/fixture/backup", "log": "/fixture/install.log",
        })
        app.updates.start_check(manual=True)
        await wait_for_condition(pilot, lambda: not app.updates.checking, timeout=10, description="stale update detected")
        assert app.updates.needs_restart
        assert app.query_one("#start-run", Button).disabled
        assert "/fixture/backup" in str(app.query_one("#version-status", Static).content)
        assert not app.query_one("#version-log", Button).disabled
        assert not launches


async def test_reopened_tui_observes_worker_before_initial_state_is_published(version_manager, monkeypatch):
    app, _, active = version_manager
    async with app.run_test(size=(80, 24)) as pilot:
        await wait_for_condition(pilot, lambda: app.updates.release is not None, timeout=10, description="checked")
        installation = app.updates.installation
        app.updates = UpdateController(app, state_root=app.updates.directory.parent)
        active.append(True)
        app.updates.start_check()
        await wait_for_condition(pilot, lambda: app.updates.installing, timeout=10, description="adopt live updater")
        assert app.query_one("#start-run", Button).disabled
        job = app.updates.directory / "jobs/surviving"
        job.mkdir(parents=True)
        plan = UpdatePlan(str(installation.prefix), str(installation.entrypoint), str(installation.manager),
                          "/usr/bin/python3", "1.0.0", "1.1.0a1", "preview", str(app.updates.directory), str(job))
        atomic_json(job / "plan.json", asdict(plan))
        atomic_json(app.updates.directory / "status.json", {
            "status": "running", "phase": "install", "job_dir": str(job),
        })
        await wait_for_condition(pilot, lambda: app.updates.installation is not None, timeout=10, description="load updater identity")
        assert app.updates.installation.entrypoint == installation.entrypoint
        atomic_json(app.updates.directory / "channel.json", {"channel": "preview"})
        atomic_json(app.updates.directory / "status.json", {"status": "success", "target": "1.1.0a1"})
        active.clear()
        await wait_for_condition(pilot, lambda: not app.updates.installing, timeout=10, description="observed completion")
        assert app.updates.needs_restart
        assert app.updates.channel == app.updates.effective_channel == "preview"
