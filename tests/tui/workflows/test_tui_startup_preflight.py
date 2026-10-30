"""Startup diagnostics exercise real workers with isolated host probes."""
import asyncio
import threading
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from textual.widgets import Button, Input, Select, Static, TabbedContent

from acprof.experiments.configuration import RunConfig
from acprof.host.run_state import MeasurementLock
from acprof.tui.app import AcprofTui
from acprof.tui.commands import PendingLaunch
from acprof.tui.logging.widget import SelectableLog
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.run.progress import ProgressSnapshot


class TestStartupPreflight:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        temporary = tmp_path
        self.directory = Path(str(temporary))

    def make_app(self):
        return AcprofTui(RunConfig.smoke(""), settings_path=self.directory / "settings.json")

    @staticmethod
    async def finish_check(app, pilot):
        await pilot.pause()
        await asyncio.wait_for(app.workers.wait_for_complete(), 10)
        await pilot.pause()

    @pytest.mark.parametrize("language", ["zh", "en"])
    @pytest.mark.parametrize("runtime", ["docker", "process"])
    async def test_completed_preflight_preserves_smoke_plan(self, language, runtime):
        app = self.make_app()
        app.configuration.preferences = replace(app.configuration.preferences, language=language)
        checks = [PreflightCheck("Runtime", "ok", "ready", runtime_kind=runtime)]
        with patch("acprof.tui.run.preflight_controller.quick_preflight", return_value=checks):
            async with app.run_test(size=(120, 30)) as pilot:
                await self.finish_check(app, pilot)
                summary = app.query_one("#run-plan-summary", Static).content
                assert isinstance(summary, str)
                runtime_label = "运行方式：" if language == "zh" else "Runtime: "
                assert f"{runtime_label}{runtime}" in summary
                assert ("预热 0 / 正式 1 窗口" if language == "zh" else
                        "0 warmup / 1 measured windows") in summary
                assert ("最小单一尺度" if language == "zh" else "Smallest single scale") in (
                    app.query_one("#input-scales", Input).placeholder
                )
                config = app.configuration.collect_config(allow_empty_model=True)
                assert (config.input_scale_policy, config.warmup, config.repeat) == ("minimal", 0, 1)

    @pytest.mark.parametrize("repeat", [7, 0], ids=["edited-plan", "invalid-plan"])
    async def test_preflight_completion_uses_edits_made_while_checking(self, repeat):
        release = threading.Event()

        def slow_check(*_args, **_kwargs):
            if not release.wait(10):
                raise TimeoutError("test did not release diagnostics")
            return [PreflightCheck("Runtime", "ok", "ready", runtime_kind="docker")]

        app = self.make_app()
        with patch("acprof.tui.run.preflight_controller.quick_preflight", side_effect=slow_check):
            async with app.run_test(size=(120, 30)) as pilot:
                try:
                    await pilot.pause()
                    assert app.preflight.check_running
                    app.configuration.apply_config(replace(
                        RunConfig.smoke("fixture/model"), input_scales="16,32", warmup=3, repeat=repeat,
                    ))
                finally:
                    release.set()
                await self.finish_check(app, pilot)
                summary = app.query_one("#run-plan-summary", Static).content
                command = app.query_one("#command-preview", Static).content
                assert isinstance(summary, str)
                assert isinstance(command, str)
                if repeat:
                    assert "输入 2 档" in summary
                    assert "预热 6 / 正式 14 窗口" in summary
                    assert "--repeat 7" in command
                else:
                    assert "配置尚未完成" in command
                    assert "每条件计划 5 个正式窗口" not in summary

    async def test_startup_is_once_async_and_success_is_silent(self):
        release = threading.Event()

        def slow_check(*_args, **_kwargs):
            if not release.wait(10):
                raise TimeoutError("test did not release diagnostics")
            return [PreflightCheck("Docker", "ok", "ready")]

        app = self.make_app()
        with patch("acprof.tui.run.preflight_controller.quick_preflight", side_effect=slow_check) as check:
            async with app.run_test(size=(80, 24), notifications=True) as pilot:
                try:
                    await pilot.pause()
                    assert (check.call_count) == (1), "startup must schedule diagnostics"
                    assert (app.query_one("#main-tabs", TabbedContent).active) == ("run-tab")
                    assert (app.query_one("#start-run", Button).disabled)
                    assert not (app.query_one("#model", Input).disabled)
                    assert (app.query_one("#open-environment-settings", Button).disabled)
                    assert (app.query_one("#open-model-store", Button).disabled)
                    await pilot.press("a", "b", "c")
                    assert (app.query_one("#model", Input).value) == ("abc")
                    assert not (app.query("#quick-check"))
                    assert not (app.query_one("#environment-status").display)
                    app.preflight.quick_check()
                    assert (check.call_count) == (1), "no duplicate worker while checking"
                    app.clear_notifications()
                finally:
                    release.set()
                await self.finish_check(app, pilot)
                assert not (app.query_one("#start-run", Button).disabled)
                assert not (app.query_one("#open-environment-settings", Button).disabled)
                assert not (app.query_one("#open-model-store", Button).disabled)
                assert (app.query_one("#run-log", SelectableLog).text) == ("")
                assert not (app._notifications)
                assert not (app.query_one("#environment-status").display)
                app.activate_tab("settings-tab")
                app.activate_tab("run-tab")
                await pilot.resize_terminal(120, 30)
                await pilot.pause()
                assert (check.call_count) == (1)

    async def test_warning_entry_opens_details_and_retry_clears_it(self):
        app = self.make_app()
        with patch("acprof.tui.run.preflight_controller.quick_preflight", side_effect=[
            [PreflightCheck("optional collector", "warn", "not supported [detail]")],
            [PreflightCheck("Docker", "ok", "ready")],
        ]) as check:
            async with app.run_test(size=(80, 24)) as pilot:
                await self.finish_check(app, pilot)
                assert not (app.query_one("#start-run", Button).disabled)
                assert (await pilot.click("#environment-status"))
                await pilot.pause()
                assert ("not supported [detail]") in (str(app.screen.query_one("#preflight-details", Static).content))
                assert (await pilot.click("#preflight-retry"))
                await self.finish_check(app, pilot)
                assert (check.call_count) == (2)
                assert (len(app.screen_stack)) == (1)
                assert not (app.query_one("#environment-status").display)

    async def test_success_does_not_reapply_initial_preset_or_notify(self):
        app = self.make_app()
        app.configuration.initial_config = replace(app.configuration.initial_config, output_dir="results/custom-startup")
        with patch("acprof.tui.run.preflight_controller.quick_preflight", return_value=[]), patch.object(app, 'notify') as notify:
            async with app.run_test(size=(80, 24)) as pilot:
                await self.finish_check(app, pilot)
                notify.assert_not_called()
                assert (app.query_one("#output-dir", Input).value) == ("results/custom-startup")
                # User selection still applies presets after initial events settle.
                app.query_one("#run-preset", Select).value = "main"
                await pilot.pause()
                assert (app.query_one("#cpus", Input).value) == ("auto")
                notify.assert_called_once()

    async def test_error_blocks_all_launch_paths_until_retry_succeeds(self):
        app = self.make_app()
        app.configuration.initial_config = RunConfig.smoke("demo/model")
        with patch("acprof.tui.run.preflight_controller.quick_preflight", side_effect=[
            OSError("diagnostic failed"), [PreflightCheck("Docker", "ok", "ready")],
        ]):
            async with app.run_test(size=(120, 30)) as pilot:
                await self.finish_check(app, pilot)
                assert (app.query_one("#start-run", Button).disabled)
                assert ("diagnostic failed") in (str(app.query_one("#preflight-run-reason", Static).content))
                with patch.object(app.process, 'execute_command') as launch:
                    await pilot.click("#start-run")
                    await pilot.press("f5")
                    field = app.query_one("#slash-command", Input)
                    field.value = "/run"
                    field.focus()
                    await pilot.press("enter")
                    app.experiments.pending_launch = PendingLaunch(("collector",), "run", app.configuration.initial_config)
                    app.experiments.confirmed_launch(True)
                    launch.assert_not_called()
                    assert (len(app.screen_stack)) == (1)
                app.preflight.quick_check()
                await self.finish_check(app, pilot)
                assert not (app.query_one("#start-run", Button).disabled)

    async def test_blocking_result_and_relevant_configuration_changes_require_recheck(self):
        app = self.make_app()
        with patch("acprof.tui.run.preflight_controller.quick_preflight", side_effect=[
            [PreflightCheck("Docker", "fail", "daemon unavailable")], [], [],
        ]) as check:
            async with app.run_test(size=(120, 30)) as pilot:
                await self.finish_check(app, pilot)
                assert (app.query_one("#start-run", Button).disabled)
                assert ("daemon unavailable") in (str(app.query_one("#preflight-run-reason", Static).content))
                app.preflight.quick_check()
                await self.finish_check(app, pilot)
                app.query_one("#model", Input).value = "changed/model"
                await pilot.pause()
                assert not (app.query_one("#start-run", Button).disabled)
                app.query_one("#gpus", Select).value = "on"
                # Form changes debounce their preview. An idle event loop does
                # not imply that the timer has fired, especially without debug.
                async def configuration_invalidated():
                    while not app.query_one("#start-run", Button).disabled:
                        await pilot.pause()

                await asyncio.wait_for(configuration_invalidated(), timeout=3)
                assert (app.query_one("#start-run", Button).disabled)
                assert (app.query_one("#environment-status").display)
                assert (check.call_count) == (2), "form edits do not start more probes"
                app.preflight.quick_check()
                await self.finish_check(app, pilot)
                assert (check.call_args.args[0].gpus) == ("on")
                assert not (app.query_one("#start-run", Button).disabled)

    async def test_issue_entry_and_retry_fit_languages_and_resize(self):
        app = self.make_app()
        with patch("acprof.tui.run.preflight_controller.quick_preflight", return_value=[
            PreflightCheck("Docker", "fail", "daemon unavailable\n" + "details\n" * 50),
        ]):
            async with app.run_test(size=(80, 24)) as pilot:
                await self.finish_check(app, pilot)
                for language in ("zh", "en"):
                    app.configuration.preferences = replace(app.configuration.preferences, language=language)
                    app.configuration.apply_ui_preferences()
                    for width, height in ((80, 24), (120, 30), (150, 45)):
                        await pilot.resize_terminal(width, height)
                        await pilot.pause()
                        entry = app.query_one("#environment-status", Button)
                        assert (entry.region.width) > (0)
                        assert (str(entry.label)) == ("环境异常" if language == "zh" else "Environment error")
                        assert (await pilot.click(entry))
                        await pilot.pause()
                        retry = app.screen.query_one("#preflight-retry", Button)
                        assert (str(retry.label)) == ("重新检查" if language == "zh" else "Check again")
                        assert (retry.region.height) > (0)
                        assert (retry.region.right) <= (width)
                        assert (retry.region.bottom) <= (height)
                        await pilot.press("escape")
                        await pilot.pause()

    async def test_form_changes_during_check_do_not_accept_old_results(self):
        release = threading.Event()

        def slow_check(*_args, **_kwargs):
            if not release.wait(10):
                raise TimeoutError("test did not release diagnostics")
            return []

        app = self.make_app()
        with patch("acprof.tui.run.preflight_controller.quick_preflight", side_effect=slow_check) as check:
            async with app.run_test(size=(120, 30)) as pilot:
                try:
                    await pilot.pause()
                    token = app.preflight.check_request
                    app.query_one("#gpus", Select).value = "on"
                    app.preflight.show_quick_check([], "obsolete error", object())
                    assert (app.preflight.check_running)
                finally:
                    release.set()
                await self.finish_check(app, pilot)
                assert (app.query_one("#start-run", Button).disabled)
                assert (app.query_one("#environment-status").display)
                assert ("obsolete error") not in (app.preflight.preflight_run_reason())
                check.assert_called_once()
                with patch.object(app, 'set_busy') as update:
                    app.preflight.show_quick_check([], "duplicate error", token)
                    update.assert_not_called()

    async def test_external_measurement_lock_prevents_probes_and_can_retry(self):
        app = self.make_app()
        with patch("acprof.host.run_state.MEASUREMENT_LOCK_ROOT", self.directory), patch(
            "acprof.tui.run.preflight_controller.quick_preflight", return_value=[],
        ) as check, ExitStack() as owner:
            owner.enter_context(MeasurementLock())
            async with app.run_test(size=(120, 30)) as pilot:
                await self.finish_check(app, pilot)
                assert (app.query_one("#start-run", Button).disabled)
                assert ("测量锁") in (app.preflight.preflight_run_reason())
                check.assert_not_called()
                owner.close()
                app.preflight.quick_check()
                await self.finish_check(app, pilot)
                check.assert_called_once()
                assert not (app.query_one("#start-run", Button).disabled)

    async def test_measurement_disables_retry_without_losing_cached_issues(self):
        app = self.make_app()
        with patch("acprof.tui.run.preflight_controller.quick_preflight", return_value=[
            PreflightCheck("optional", "warn", "unavailable"),
        ]) as check:
            async with app.run_test(size=(120, 30)) as pilot:
                await self.finish_check(app, pilot)
                app.execution.session.snapshot = ProgressSnapshot(measurement_active=True)
                app.set_busy(True)
                app.preflight.quick_check()
                assert (await pilot.click("#environment-status"))
                await pilot.pause()
                assert (app.screen.query_one("#preflight-retry", Button).disabled)
                check.assert_called_once()
                await pilot.press("escape")
                app.execution.session.snapshot = ProgressSnapshot()
                app.set_busy(False)
                assert not (app.query_one("#start-run", Button).disabled)

    async def test_quit_during_check_drops_late_callback(self):
        release = threading.Event()
        finished = threading.Event()

        def slow_check(*_args, **_kwargs):
            try:
                if not release.wait(10):
                    raise TimeoutError("test did not release diagnostics")
                return []
            finally:
                finished.set()

        app = self.make_app()
        with patch("acprof.tui.run.preflight_controller.quick_preflight", side_effect=slow_check):
            try:
                async with app.run_test(size=(120, 30)) as pilot:
                    await pilot.pause()
                    token = app.preflight.check_request
                    app.action_request_quit()
                    assert app._task is not None
                    await asyncio.wait_for(app._task, 5)
                callback = Mock(side_effect=AssertionError("late UI access"))
                with patch.object(app, app.query_one.__name__, callback):
                    release.set()
                    assert (await asyncio.to_thread(finished.wait, 5))
                    app.preflight.show_quick_check([], "late error", token)
                    callback.assert_not_called()
            finally:
                release.set()
