"""用户入口、结果归属和迟到读取的行为回归；不启动真实采集。"""
import asyncio
import csv
import json
import sys
import tempfile
import threading
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from textual.widgets import Button, Input, Select, Static, TabbedContent
from tui_fixtures import AcprofTui

from acprof.artifact_layout import ArtifactLayout
from acprof.experiment import RunConfig
from acprof.tui.commands import PendingLaunch
from acprof.tui.diagnostics import quick_preflight, summarize_result_csv
from acprof.tui.i18n import translate
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.run_form import infer_preset, matches_preset
from acprof.tui.run_results import RunArtifacts, inspect_run_result


def write_csv(path, latencies=(0.01, 0.2)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["cpu_cores", "mem_cap_gb", "gpu_mode",
            "input_scale", "warmup", "repeat_idx", "status", "latency_app_s"])
        writer.writeheader()
        for scale, latency in zip((64, 128), latencies):
            writer.writerow(dict(cpu_cores=1, mem_cap_gb=4, gpu_mode="off", input_scale=scale,
                                 warmup=0, repeat_idx=0, status="ok", latency_app_s=latency))


def test_execution_settings_do_not_change_preset():
    config = replace(RunConfig.smoke("demo/model"), output_dir="elsewhere", model_store="cache",
                     download_mode="direct", notify="auto", resume=True, skip_build=True)
    assert (infer_preset(config)) == ("smoke")
    assert (matches_preset(config, "smoke"))
    assert not (matches_preset(replace(config, cpus="2"), "smoke"))

def test_summary_keeps_input_scales_separate_and_single_windows_insufficient():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "result.csv"
        write_csv(path)
        summary = summarize_result_csv(path)
        groups = getattr(summary, "groups", ())
        assert (len(groups)) == (2), "两个输入规模必须分别显示"
        assert ([group["mean"] for group in groups]) == ([0.01, 0.2])
        assert (all(group["reason"] == "insufficient_windows" for group in groups))
        assert (all(group["ci_low"] is None for group in groups))

def test_unrequested_rapl_is_not_a_pass():
    with patch("acprof.tui.diagnostics.shutil.which", return_value=None):
        checks = quick_preflight(RunConfig.smoke("demo/model"))
    rapl = next(check for check in checks if check.label == "CPU RAPL")
    assert (rapl.status) == ("not_requested")


class TestRunAttribution:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        temporary = tmp_path
        self.root = Path(str(temporary)) / "run"
        self.before = RunArtifacts.read(self.root)
        self.layout = ArtifactLayout.for_new_run(self.root)
        self.layout.initialize()
        self.state: dict = dict(schema_version=1, layout_version=2, run_id="run-1", status="complete", outcome="ok",
            attempts=[dict(pid=123, started_at="2026-10-02T01:00:00Z", ended_at="2026-10-02T01:01:00Z")],
            options=dict(cpus="1", mems="4", gpus="off", warmup=0, repeat=1),
            runtime=dict(planned=dict(scales=[64, 128])),
            cases={"case.csv": dict(status="complete", completed_at="2026-10-02T01:01:00Z")})
        self.save_state()
        write_csv(self.layout.result_csv)
        self.layout.path("capability_report.json").write_text(json.dumps({"requested_measurements_complete": True}))

    def save_state(self):
        path = self.layout.path("run_state.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state))

    def test_complete_requires_matching_attempt_and_audited_coverage(self):
        result = inspect_run_result(self.before, 123)
        assert (result.stage(0, False, "")) == ("已完成")
        assert (result.result_csv) == (str(self.layout.result_csv))
        assert ((result.completed_cases, result.total_cases, result.new_cases)) == ((1, 1, 1))
        assert (inspect_run_result(self.before, 999).stage(0, False, "")) == ("失败")
        self.state["runtime"]["planned"]["scales"].append(256)
        self.save_state()
        assert (inspect_run_result(self.before, 123).stage(0, False, "")) == ("部分完成")

    def test_zero_exit_with_missing_requested_metrics_is_partial(self):
        self.layout.path("capability_report.json").write_text(json.dumps({"requested_measurements_complete": False}))
        assert (inspect_run_result(self.before, 123).stage(0, False, "")) == ("部分完成")

    @pytest.mark.parametrize(("content", "reason", "english"), [
        ("{", "JSON 格式无效", "Invalid JSON"),
        ("[]", "最外层必须是 JSON 对象", "A JSON object is required at the top level"),
        ('{"requested_measurements_complete": NaN}', "包含非有限数值", "Contains a non-finite number"),
        ('{"requested_measurements_complete": 1e999}', "包含非有限数值", "Contains a non-finite number"),
    ])
    def test_invalid_capability_evidence_retains_attributed_results(self, content, reason, english):
        self.layout.path("capability_report.json").write_text(content)

        result = inspect_run_result(self.before, 123)

        assert result.belongs_to_attempt
        assert (result.result_csv) == (str(self.layout.result_csv))
        assert (result.stage(0, False, "")) == ("部分完成")
        assert reason in result.detail
        assert english in translate(result.detail, "en")

    def test_oversized_capability_evidence_is_not_read_as_complete(self):
        content = json.dumps({
            "requested_measurements_complete": True,
            "padding": "x" * (4 * 1024 * 1024),
        })
        self.layout.path("capability_report.json").write_text(content)

        result = inspect_run_result(self.before, 123)

        assert result.belongs_to_attempt
        assert (result.result_csv) == (str(self.layout.result_csv))
        assert (result.stage(0, False, "")) == ("部分完成")
        assert "超过 4 MiB 读取上限" in result.detail
        assert "Exceeds the 4 MiB read limit" in translate(result.detail, "en")

    def test_preflight_failure_and_unchanged_resume_never_promote_history(self):
        before = RunArtifacts.read(self.root)
        assert not (inspect_run_result(before, 123).belongs_to_attempt)
        self.state["attempts"].append(dict(pid=124, started_at="new", ended_at="end"))
        self.save_state()
        result = inspect_run_result(before, 124)
        assert (result.belongs_to_attempt)
        assert (result.result_csv) == ("")
        assert (result.new_cases) == (0)

    def test_stopped_case_is_retained_without_merged_csv(self):
        self.layout.result_csv.unlink()
        self.state["status"] = "interrupted"
        self.save_state()
        result = inspect_run_result(self.before, 123)
        assert (result.stage(0, True, "")) == ("已停止")
        assert (result.stage(1, False, "")) == ("部分完成")
        assert (result.new_cases) == (1)
        assert (result.result_csv) == ("")

    def test_corrupt_post_run_state_reports_unverified_csv_without_attribution(self):
        state_path = self.layout.path("run_state.json")
        state_path.write_text("{")

        result = inspect_run_result(self.before, 123)

        assert not result.belongs_to_attempt
        assert (result.result_csv) == ("")
        assert (result.stage(0, False, "")) == ("失败")
        assert "无法核验本次运行状态" in result.detail
        assert str(self.layout.result_csv) in result.detail
        assert "Could not verify this attempt's run state" in translate(result.detail, "en")


class TestTuiReliability:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        temporary = tmp_path
        self.directory = Path(str(temporary))

    def make_app(self) -> AcprofTui:
        return AcprofTui(replace(RunConfig.smoke("demo/model"), output_dir=str(self.directory)),
                         settings_path=self.directory / "settings.json")

    async def test_preflight_blocks_button_f5_and_command_and_recovers_on_error(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            with patch.object(app, "_execute_quick_check"), patch.object(app, "_launch") as launch:
                app.action_quick_check()
                assert (app.query_one("#start-run", Button).disabled)
                assert (app.query_one("#stop-run", Button).disabled)
                app._activate_tab("run-tab")
                await pilot.pause()
                await pilot.click("#start-run")
                await pilot.press("f5")
                field = app.query_one("#slash-command", Input)
                field.value = "/run"
                field.focus()
                await pilot.press("enter")
                await pilot.pause()
                launch.assert_not_called()
                assert (len(app.screen_stack)) == (1)
                app._show_quick_check([], "preflight failed", app._check_request)
                assert (app.query_one("#start-run", Button).disabled)
                assert not (app._is_busy())

    async def test_failed_attempt_does_not_promote_old_csv(self):
        app = self.make_app()
        config = app.initial_config
        path = config.result_csv(self.directory)
        write_csv(path)
        original = path.read_bytes()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            with patch.object(app, "_execute_command"), patch("acprof.tui.app.summarize_result_csv", wraps=summarize_result_csv) as read:
                app._launch(PendingLaunch(("collector",), "run", config))
                app._process_finished("run", 2, None, "")
                await pilot.pause()
                read.assert_not_called()
                assert ("本次未产生结果") in (str(app.query_one("#result-summary", Static).content))
        assert (path.read_bytes()) == (original)

    async def test_report_a_finishing_after_b_cannot_replace_b(self):
        from acprof.tui.reports import ReportView
        app = self.make_app()
        started, release = threading.Event(), threading.Event()
        self._request.addfinalizer(partial(release.set))
        a, b = self.directory / "a.json", self.directory / "b.json"
        finished = []
        def read(path):
            if path == a:
                started.set()
                release.wait(10)
            finished.append(path)
            return ReportView(path, "test", (), (), "")
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            with patch("acprof.tui.app.read_report", side_effect=read):
                app._open_report(str(a))
                assert (await asyncio.to_thread(started.wait, 2))
                app._open_report(str(b))
                for _ in range(100):
                    await pilot.pause()
                    if app._report_view is not None:
                        break
                    await asyncio.sleep(0.01)
                release.set()
                await app.workers.wait_for_complete()
                await pilot.pause()
            assert (app._report_view) is not None
            assert app._report_view is not None
            assert (app._report_view.source) == (b)
            assert (finished) == ([b, a])

    async def test_zero_exit_without_completion_evidence_is_not_success(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app._process_kind = "run"
            with patch.object(app, "notify") as notify:
                app._process_finished("run", 0, None, "")
            assert ("任务已完成") not in ([str(call.args[0]) for call in notify.call_args_list])
            assert (app._latest_snapshot.stage) == ("失败")

    async def test_final_result_audit_does_not_offer_a_stop_action(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            app._process_kind = "run"  # Child exited; its result audit still owns the task.
            app._set_busy(True)
            assert (app.query_one("#start-run", Button).disabled)
            assert (app.query_one("#stop-run", Button).disabled)
            app.action_request_stop()
            assert (len(app.screen_stack)) == (1)
            assert not (app._stop_requested)
            app._consume_process_line("Profiling complete!", ProgressSnapshot(stage="已完成"), True)
            assert (app._latest_snapshot.stage) == ("核验产物")

    async def test_background_summary_keeps_ui_responsive_and_only_shows_latest_selection(self):
        app = self.make_app()
        a, b = self.directory / "a.csv", self.directory / "b.csv"
        write_csv(a)
        write_csv(b, (0.03, 0.04))
        started, release = threading.Event(), threading.Event()
        self._request.addfinalizer(partial(release.set))
        ui_thread = threading.get_ident()
        def read(path, **kwargs):
            assert (threading.get_ident()) != (ui_thread)
            if path == a:
                started.set()
                release.wait(10)
            return summarize_result_csv(path, **kwargs)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            with patch("acprof.tui.app.summarize_result_csv", side_effect=read):
                app._update_result_summary(str(a))
                assert (await asyncio.to_thread(started.wait, 2))
                assert ("正在读取") in (str(app.query_one("#result-summary", Static).content))
                await pilot.press("f2")
                await pilot.pause()
                app._update_result_summary(str(b))
                for _ in range(100):
                    await pilot.pause()
                    if "40 ms" in str(app.query_one("#result-summary", Static).content):
                        break
                    await asyncio.sleep(0.01)
                assert ("40 ms") in (str(app.query_one("#result-summary", Static).content))
                assert (app.query_one("#start-run", Button).disabled), "旧读取结束前不能启动采集"
                release.set()
                await app.workers.wait_for_complete()
                await pilot.pause()
            assert (str(b)) in (str(app.query_one("#result-summary", Static).content))
            assert (str(a)) not in (str(app.query_one("#result-summary", Static).content))
            assert not (app._is_busy())

    async def test_cancelled_and_failed_reads_release_controls(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            app._update_result_summary(str(self.directory / "missing.csv"))
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert ("无法读取结果") in (str(app.query_one("#result-summary", Static).content))
            assert not (app._is_busy())
            with patch.object(app, "_execute_summary_read"):
                app._update_result_summary("cancel.csv")
                token = app._summary_request
                assert token is not None
                app._cancel_result_reads()
                assert (token.is_set())
                assert (app._is_busy())
                app._show_result_summary(Path("cancel.csv"), token, None, "cancelled", True)
            assert ("读取已取消") in (str(app.query_one("#result-summary", Static).content))
            assert not (app._is_busy())

    async def test_late_checks_and_shutdown_callbacks_do_not_touch_widgets(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            with patch.object(app, "_execute_quick_check"):
                app.action_quick_check()
                token = app._check_request
                app._show_quick_check([], "old error", object())
                assert (app._check_running)
                app._show_quick_check([], "", token)
                assert not (app._is_busy())
            app.action_request_quit()
            callback = Mock(side_effect=AssertionError("unmounted UI access"))
            app._deliver_ui_callback(callback)
            callback.assert_not_called()
            await asyncio.wait_for(app._task, timeout=5)
        with patch.object(app, app.query_one.__name__, side_effect=AssertionError("unmounted UI access")):
            app._show_quick_check([], "late error", token)
            app._show_result_summary(self.directory, object(), None, "late error", False)
            app._show_report(None, "late error", object())

    async def test_child_attempt_is_audited_then_updates_current_result(self):
        app = self.make_app()
        directory = app.initial_config.result_dir(self.directory)
        script = f'''
import csv, json, os
from pathlib import Path
from acprof.artifact_layout import ArtifactLayout
layout = ArtifactLayout.for_new_run(Path({str(directory)!r}))
layout.initialize()
state = dict(schema_version=1, layout_version=2, run_id="child-run", status="complete", outcome="ok",
             attempts=[dict(pid=os.getpid(), started_at="start", ended_at="end")],
             options=dict(cpus="1", mems="4", gpus="off", warmup=0, repeat=1),
             runtime=dict(planned=dict(scales=[64])), cases={{"case.csv": dict(status="complete")}})
layout.path("run_state.json").parent.mkdir(parents=True, exist_ok=True)
layout.path("run_state.json").write_text(json.dumps(state))
layout.path("capability_report.json").write_text(json.dumps({{"requested_measurements_complete": True}}))
layout.result_csv.write_text("cpu_cores,mem_cap_gb,gpu_mode,input_scale,warmup,repeat_idx,status,latency_app_s\\n1,4,off,64,0,0,ok,0.02\\n")
print("Profiling complete!", flush=True)
'''
        async with app.run_test(size=(150, 45)) as pilot:
            await pilot.pause()
            app._launch(PendingLaunch((sys.executable, "-u", "-c", script), "run", app.initial_config))
            for _ in range(3):
                await app.workers.wait_for_complete()
                await pilot.pause()
            assert (app._latest_snapshot.stage) == ("已完成")
            assert (app.query_one("#result-csv", Input).value) == (str(directory / "result_all.csv"))
            assert ("20 ms") in (str(app.query_one("#result-summary", Static).content))
            assert not (app._is_busy())

    async def test_quit_during_summary_read_does_not_wait_for_io_or_apply_late_data(self):
        path = self.directory / "slow.csv"
        write_csv(path)
        started, release, delivered = threading.Event(), threading.Event(), threading.Event()
        app = self.make_app()
        original_callback = app._safe_process_callback

        def read(*args, **kwargs):
            started.set()
            release.wait(10)
            return summarize_result_csv(*args, **kwargs)

        def deliver(*args):
            try:
                original_callback(*args)
            finally:
                delivered.set()

        with patch("acprof.tui.app.summarize_result_csv", side_effect=read), patch.object(
            app, "_safe_process_callback", side_effect=deliver,
        ):
            try:
                async with app.run_test(size=(120, 30)) as pilot:
                    await pilot.pause()
                    app._update_result_summary(str(path))
                    assert (await asyncio.to_thread(started.wait, 5))
                    assert (app._is_busy())
                    app.action_request_quit()
                    await asyncio.wait_for(app._task, timeout=5)
                    assert not (delivered.is_set()), "退出不应等待 CSV 线程返回"
                with patch.object(app, app.query_one.__name__) as query:
                    release.set()
                    assert (await asyncio.to_thread(delivered.wait, 5))
                    query.assert_not_called()
            finally:
                release.set()

    async def test_large_csv_allows_navigation_and_summary_preserves_window_count(self):
        path = self.directory / "large.csv"
        with path.open("w") as stream:
            stream.write("cpu_cores,mem_cap_gb,gpu_mode,input_scale,warmup,repeat_idx,status,latency_app_s\n")
            stream.writelines(f"1,4,off,64,0,{index},ok,0.01\n" for index in range(100_000))
        app = self.make_app()
        ui_thread = threading.get_ident()
        def read(path, **kwargs):
            assert (threading.get_ident()) != (ui_thread)
            return summarize_result_csv(path, **kwargs)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            with patch("acprof.tui.app.summarize_result_csv", side_effect=read):
                app._update_result_summary(str(path))
                await pilot.press("f2")
                assert (app.query_one("#main-tabs", TabbedContent).active) == ("settings-tab")
                await app.workers.wait_for_complete()
                await pilot.pause()
            assert ("有效窗口 100000") in (str(app.query_one("#result-summary", Static).content))
            assert ("10 ms") in (str(app.query_one("#result-summary", Static).content))
            assert not (app._is_busy())

    async def test_preset_labels_and_quiet_unselected_checks_in_both_languages_and_sizes(self):
        from acprof.tui.diagnostics import PreflightCheck
        from acprof.tui.log import SelectableLog
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            for language in ("zh", "en"):
                app.ui_preferences = replace(app.ui_preferences, language=language)
                app._apply_ui_preferences()
                for width, height in ((80, 24), (120, 30), (150, 45)):
                    await pilot.resize_terminal(width, height)
                    app._apply_config(RunConfig.smoke("demo/model"), preset="smoke")
                    app.query_one("#output-dir", Input).value = "results/changed"
                    app._refresh_command_preview(notify=False, sync_preset=True)
                    label = app.query_one("#run-preset SelectCurrent #label", Static)
                    assert ("adjusted") not in (str(label.content))
                    assert ("已调整") not in (str(label.content))
                    app.query_one("#cpus", Input).value = "2"
                    app._refresh_command_preview(notify=False, sync_preset=True)
                    assert (app.query_one("#run-preset", Select).value) == ("smoke")
                    assert ("已调整" if language == "zh" else "adjusted") in (str(label.content))
                    with patch.object(app, "_execute_quick_check"):
                        app.action_quick_check()
                        app._show_quick_check([PreflightCheck("Docker", "ok", "ready"),
                            PreflightCheck("CPU RAPL", "not_requested", "basic", "not_requested")], "", app._check_request)
                    await pilot.pause()
                    assert ("CPU RAPL") not in (app.query_one("#run-log", SelectableLog).text)
                    assert not (app.query_one("#environment-status").display)
                    assert (app.query_one("#main-tabs", TabbedContent).active) == ("run-tab")
                    assert (app.query_one("#run-form").region.height) > (0)
                    assert (app.query_one("#stop-run", Button).disabled)
                    app.action_clear_log()
