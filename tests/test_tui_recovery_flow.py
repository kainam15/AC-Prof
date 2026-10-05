"""Users can recover or restart from the failed experiment without editing paths."""
import asyncio
import json
import os
import threading
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, Static
from textual.worker import WorkerCancelled
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig, build_run_command
from acprof.host import run_state
from acprof.run_args import build_parser
from acprof.tui.commands import PendingLaunch
from acprof.tui.experiment_catalog import scan_experiments
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.recovery import review_recovery


@pytest.fixture
def saved_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "MEASUREMENT_LOCK_ROOT", tmp_path)
    monkeypatch.setattr(run_state, "host_identity", lambda _: {"source_sha256": "original"})
    config = replace(RunConfig.smoke("demo/model"), output_dir=str(tmp_path / "results"))
    command = build_run_command(config, project_dir=Path.cwd())
    args = build_parser().parse_args(command[command.index("--model"):])
    for name, value in asdict(RunConfig.from_namespace(args).validate(project_dir=Path.cwd())).items():
        setattr(args, name, value)
    directory = config.result_dir(Path.cwd())
    state = run_state.RunState(directory, run_state.run_options(args), resume=False, project_dir=Path.cwd())
    state.close("failed")
    (directory / "logs").mkdir()
    (directory / "logs/interface_validation.log").write_text("interrupted preparation\n")
    return config, directory


@pytest.mark.parametrize("language,size", [("zh", (80, 24)), ("en", (120, 30)), ("zh", (150, 45))])
async def test_start_existing_experiment_reviews_preparation_retry(saved_preparation, tmp_path, language, size):
    config, directory = saved_preparation
    original = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    app.ui_preferences = replace(app.ui_preferences, language=language)
    with patch.object(app, "_launch") as launch:
        async with app.run_test(size=size) as pilot:
            assert await pilot.click("#start-run")
            await app.workers.wait_for_complete()
            await pilot.pause()
            launch.assert_not_called()
            retry = app.screen.query_one("#recovery-resume", Button)
            assert not retry.disabled
            assert str(retry.label) == ("重试准备" if language == "zh" else "Retry preparation")
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            for button in app.screen.query(Button):
                assert button.region.width > 0
                assert button.region.right <= 80
                assert button.region.bottom <= 24
            assert await pilot.click("#recovery-resume")
            await app.workers.wait_for_complete()
            await pilot.pause()
            pending = launch.call_args.args[0]
            assert pending.config.resume
            assert "--resume" in pending.command
            assert pending.config.result_dir(Path.cwd()) == directory
    assert all(path.read_bytes() == content for path, content in original.items())


async def test_changed_source_offers_new_experiment_and_keeps_original(saved_preparation, tmp_path, monkeypatch):
    config, directory = saved_preparation
    original = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    monkeypatch.setattr(run_state, "host_identity", lambda _: {"source_sha256": "changed"})
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    with patch.object(app, "_launch") as launch:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.click("#start-run")
            await app.workers.wait_for_complete()
            await pilot.pause()
            launch.assert_not_called()
            assert app.screen.query_one("#recovery-resume", Button).disabled
            assert "源码" in str(app.screen.query_one("#recovery-detail", Static).content)
            assert await pilot.click("#recovery-new")
            await app.workers.wait_for_complete()
            await pilot.pause()
            pending = launch.call_args.args[0]
            assert not pending.config.resume
            assert "--resume" not in pending.command
            assert pending.config.model == config.model
            assert pending.config.repeat == config.repeat
            assert pending.config.output_dir != config.output_dir
            assert not pending.config.result_dir(Path.cwd()).exists()
    assert all(path.read_bytes() == content for path, content in original.items())


async def test_recovery_cancel_never_launches_or_changes_form(saved_preparation, tmp_path):
    config, _directory = saved_preparation
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    with patch.object(app, "_launch") as launch:
        async with app.run_test(size=(80, 24)) as pilot:
            before = app._collect_config()
            await pilot.click("#start-run")
            await app.workers.wait_for_complete()
            await pilot.pause()
            launch.assert_not_called()
            await pilot.press("escape")
            await pilot.pause()
            assert app._collect_config() == before
            assert not app._read_jobs
            assert not app.query_one("#start-run", Button).disabled


async def test_confirmation_rechecks_source_before_launch(saved_preparation, tmp_path, monkeypatch):
    config, _directory = saved_preparation
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    with patch.object(app, "_launch") as launch:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.click("#start-run")
            await app.workers.wait_for_complete()
            await pilot.pause()
            launch.assert_not_called()
            monkeypatch.setattr(run_state, "host_identity", lambda _: {"source_sha256": "changed"})
            await pilot.click("#recovery-resume")
            await app.workers.wait_for_complete()
            await pilot.pause()
            launch.assert_not_called()
            assert app.screen.query_one("#recovery-resume", Button).disabled


@pytest.mark.parametrize("changed_config", [False, True])
async def test_recovery_preflight_failure_or_edited_config_never_launches(
        saved_preparation, tmp_path, changed_config):
    config, directory = saved_preparation
    # Restoring a different profiling mode requires fresh environment checks.
    app = AcprofTui(replace(config, profiling_mode="full"), settings_path=tmp_path / "settings.json")
    record = scan_experiments([directory], max_depth=0).records[0]
    with patch.object(app, "_launch") as launch:
        async with app.run_test(size=(80, 24)) as pilot:
            with patch.object(app, "_execute_quick_check") as check:
                app._resume_experiment(record)
                await app.workers.wait_for_complete()
                await pilot.pause()
                await pilot.click("#recovery-resume")
                await app.workers.wait_for_complete()
                await pilot.pause()
                check.assert_called_once()
                assert app._recovery_after_check is not None
                launch.assert_not_called()
                if changed_config:
                    app.query_one("#repeat", Input).value = str(config.repeat + 1)
                    await pilot.pause()
                token = check.call_args.args[1]
                app._show_quick_check([], "" if changed_config else "preflight failed", token)
                await app.workers.wait_for_complete()
                await pilot.pause()
                launch.assert_not_called()
                assert app._recovery_after_check is None


def test_review_checks_active_writer_and_completed_experiment(saved_preparation):
    config, directory = saved_preparation
    pending = PendingLaunch(tuple(build_run_command(config, project_dir=Path.cwd())), "run", config)
    with run_state.ResultDirectoryLock(directory):
        review = review_recovery(pending, project_dir=Path.cwd(), python_executable=Path("/fixed/python"))
    assert review.resume is None
    assert "进程写入" in review.detail
    assert review.restart is not None
    path = directory / ".acprof/run_state.json"
    state = json.loads(path.read_text())
    state["status"] = "complete"
    path.write_text(json.dumps(state))
    review = review_recovery(pending, project_dir=Path.cwd(), python_executable=Path("/fixed/python"))
    assert review.phase == "complete"
    assert review.resume is None
    assert review.restart is not None


def test_review_rejects_hardlinked_result_lock(saved_preparation, tmp_path):
    config, directory = saved_preparation
    pending = PendingLaunch(tuple(build_run_command(config, project_dir=Path.cwd())), "run", config)
    lock_path = directory / ".acprof/result.lock"
    external = tmp_path / "external-result.lock"
    external.write_text("")
    lock_path.unlink()
    os.link(external, lock_path)

    review = review_recovery(pending, project_dir=Path.cwd(), python_executable=Path("/fixed/python"))

    assert review.resume is None
    assert "锁文件不可用" in review.detail
    assert review.restart is not None


def test_new_destination_review_creates_nothing(tmp_path):
    config = replace(RunConfig.smoke("demo/model"), output_dir=str(tmp_path / "new"))
    pending = PendingLaunch(tuple(build_run_command(config, project_dir=Path.cwd())), "run", config)
    assert review_recovery(pending, project_dir=Path.cwd(), python_executable=Path("/fixed/python")) is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("close_ui", [False, True])
async def test_cancel_or_close_discards_delayed_review(saved_preparation, tmp_path, close_ui):
    config, _directory = saved_preparation
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    def slow_review(*args, **kwargs):
        started.set()
        if not release.wait(5):
            raise TimeoutError("test did not release recovery read")
        try:
            return review_recovery(*args, **kwargs)
        finally:
            finished.set()
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    try:
        with patch.object(app, "_launch") as launch, patch("acprof.tui.recovery_actions.review_recovery", slow_review):
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.click("#start-run")
                assert await asyncio.to_thread(started.wait, 3)
                assert app._read_jobs
                assert app.query_one("#start-run", Button).disabled
                if close_ui:
                    app.action_request_quit()
                else:
                    app._cancel_result_reads()
                    assert app._read_jobs  # Reading owns the guard until the worker exits.
                release.set()
                try:
                    await app.workers.wait_for_complete()
                except WorkerCancelled:
                    assert close_ui
                assert await asyncio.to_thread(finished.wait, 3)
                launch.assert_not_called()
                if not close_ui:
                    await pilot.pause()
                    assert not app._read_jobs
                    assert not app.query_one("#start-run", Button).disabled
    finally:
        release.set()


async def test_failed_run_exposes_recovery_action_without_promising_resume(saved_preparation, tmp_path):
    config, _directory = saved_preparation
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        app._active_run_config = config
        app._process_finished("run", 1, None, "")
        app._activate_tab("monitor-tab")
        await pilot.pause()
        assert app.query_one("#run-recovery-actions").display
        assert "使用相同参数续跑" not in str(app.query_one("#status-detail", Static).content)
        assert await pilot.click("#review-run-recovery")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not app.screen.query_one("#recovery-resume", Button).disabled
