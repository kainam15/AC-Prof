"""Preflight, preparation dialogs and bounded retry interactions.

Only the App owns the active process, measurement-window state and worker
cancellation; this mixin handles presentation and preflight requests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from textual import on, work
from textual.message_pump import MessagePump
from textual.widgets import Button, Static

from acprof.experiment import RunConfig
from acprof.host.run_state import MeasurementLock
from acprof.messages import join_messages, message
from acprof.tui import run_planning
from acprof.tui.diagnostics import PreflightCheck
from acprof.tui.views import EnvironmentPreflightScreen


class PreflightActions(MessagePump):
    @on(Button.Pressed, "#environment-status")
    def show_environment_status(self) -> None:
        lines = []
        reason = self._preflight_run_reason()
        if reason:
            lines.append(reason)
        lines.extend(message("[{0}] {1}: {2}",
                             message("失败" if check.status == "fail" else "警告"),
                             check.label, check.detail)
                     for check in self._preflight_checks if check.status in {"warn", "fail"})
        if not lines:
            return
        self.push_screen(EnvironmentPreflightScreen(
            join_messages("\n\n", lines),
            retry_disabled=not self._operation_state().allows("check"),
        ), self._preflight_dialog_closed)

    def _preflight_dialog_closed(self, retry: bool | None) -> None:
        if retry:
            self.action_quick_check()

    def _network_preflight_report(self, line: str) -> None:
        report = json.loads(line.split(" ", 1)[1])
        from acprof.tui.downloads import download_summary
        self.query_one("#network-download-summary", Static).update(download_summary(report, self.tr))
        from acprof.tui.presentation import format_bytes
        self.query_one("#download-estimate", Static).update(format_bytes(report.get("expected_download_bytes")))

    def _network_download_report(self, line: str) -> None:

        from acprof.tui.downloads import download_result_summary
        report = json.loads(line.split(" ", 1)[1])
        summary = self.query_one("#network-download-summary", Static)
        summary.update(f"{summary.content}\n{self.tr(download_result_summary(report))}")

    @on(Button.Pressed, "#open-model-store")
    def open_model_store(self):
        if self._is_busy() or self._check_running or self._latest_snapshot.measurement_active:
            return
        from acprof.host.model_store import store_root
        from acprof.tui.model_store import ModelStoreScreen
        root = Path(self._input("model-store")).expanduser().resolve() if self._input("model-store") else store_root()
        self._environment_open = True
        self._set_busy(True)
        self.push_screen(ModelStoreScreen(root, self._input("model-store-max")), self._environment_closed)

    def _show_preparation(self, event: dict) -> None:
        from acprof.tui.preparation import PreparationScreen
        if self._preparation_screen is None:
            self._preparation_screen = PreparationScreen(event, respond=self._preparation_answered)
            self.push_screen(self._preparation_screen)
        else:
            self._preparation_screen.update_event(event)

    def _close_preparation(self) -> None:
        screen = self._preparation_screen
        self._preparation_screen = None
        self._preparation_request = None
        if screen is not None:
            screen.close()

    def _preparation_event(self, event: dict) -> None:
        if event.get("input_plan") is not None and self._active_run_config is not None:
            if self._latest_snapshot.measurement_active:
                raise RuntimeError("input plan update inside measurement window")
            plan = event["input_plan"]
            from acprof.preparation_events import validate_input_plan
            validate_input_plan(plan)
            self._planned_input = plan
            self._planned_input_identity = run_planning.input_identity(self._active_run_config)
            self._update_plan_summary(self._active_run_config)
        if self._stop_requested:
            return
        if event["stage"] == "runtime" and event.get("status") == "passed":
            self._write_log(self.tr("✓ 模型检测完成"))
            self._close_preparation()
            return
        request = event.get("request")
        if request:
            if self._latest_snapshot.measurement_active or self._preparation_request is not None:
                raise RuntimeError("unexpected preparation request")
            process = self._lifecycle.process
            if process is None or process.poll() is not None:
                return
            if request["kind"] == "error":
                self._write_log(f"[preparation][ERROR] {event['stage']}: {request.get('detail', '')}")
            self._preparation_request = (process, request["id"])
        if request:
            self._show_preparation(event)

    def _preparation_answered(self, result) -> None:
        from acprof.preparation_events import encode_reply
        if result["action"] == "request-stop":
            self.action_request_stop()
            return
        pending = self._preparation_request
        if result["action"] == "switch-source":
            self._pending_source_change = result
            result = {"action": "cancel"}
        self._preparation_request = None
        if result["action"] == "cancel":
            self._preparation_cancelled = True
            self._stop_requested = True
            if pending is None:
                self._stop_process_gracefully(self._process_token, self._lifecycle.process)
            self._close_preparation()
        if pending is None:
            return
        process, request_id = pending
        try:
            self._lifecycle.reply(process, encode_reply(request_id, **result))
        except (OSError, ValueError, RuntimeError) as exc:
            self.notify(str(exc), severity="error")
            self._stop_requested = True
            self._stop_process_gracefully(self._process_token, process)
        else:
            if result["action"] != "cancel":
                self._close_preparation()

    def _preflight_config(self) -> RunConfig:
        # Diagnostics remain usable without a model or valid resource matrix.
        return RunConfig(
            model="preflight-only",
            profiling_mode=self._select("profiling-mode"),
            gpus=self._select("gpus"),
            sniff_iface=self._input("sniff-iface"),
            compute_profile_tool=self._select("compute-profile-tool"),
            execution_profile_tool=self._select("execution-profile-tool"),
        )

    def _preflight_run_reason(self) -> str:
        if not self.is_running or not self._form_ready or self._ui_closing:
            return message("正在检查环境；完成后可开始采集。")
        if self._check_running or not self._check_completed:
            return message("正在检查环境；完成后可开始采集。")
        if self._preflight_error:
            return message("环境检查失败：{0}", self._preflight_error)
        if self._preflight_config() != self._checked_config:
            return message("环境相关配置已更改，请重新检查。")
        failures = [message("{0}: {1}", check.label, check.detail)
                    for check in self._preflight_checks if check.status == "fail"]
        return message("采集已阻止：{0}", join_messages("；", failures)) if failures else ""

    def _allow_collection(self) -> bool:
        reason = self._preflight_run_reason()
        if reason:
            self._refresh_preflight_state()
            self.notify(reason, title="环境状态", severity="warning", timeout=8)
            return False
        return True

    def _refresh_preflight_state(self) -> None:
        if not self.is_running or not self._form_ready or self._ui_closing:
            return
        reason = self._preflight_run_reason()
        issues = [check for check in self._preflight_checks if check.status in {"warn", "fail"}]
        stale = self._check_completed and self._preflight_config() != self._checked_config
        entry = self.query_one("#environment-status", Button)
        entry.display = bool(issues or self._preflight_error or stale)
        failed = bool(self._preflight_error or any(check.status == "fail" for check in issues))
        self._set_text(entry, "环境异常" if failed else "环境警告", "label")
        entry.variant = "error" if failed else "warning"
        self._set_text(entry, "查看环境问题与重新检查", "tooltip")
        detail = self.query_one("#preflight-run-reason", Static)
        detail.display = bool(reason)
        self._set_text(detail, reason)
        start = self.query_one("#start-run", Button)
        start.disabled = bool(reason) or not self._operation_state().allows("run")
        self._set_text(start, reason or "开始采集", "tooltip")
        self.refresh_bindings()

    def action_quick_check(self) -> None:
        if not self.is_running or not self._form_ready or self._ui_closing:
            return
        if self._check_running:
            return
        if not self._allow_operation("check"):
            return
        self._check_config = self._preflight_config()
        self._check_running = True
        self._check_request = object()
        # Keep the current page and editable form. Only conflicting work waits.
        self._set_busy(False)
        self._execute_quick_check(self._check_config, self._check_request)

    @work(thread=True, group="preflight", exclusive=True, exit_on_error=False)
    def _execute_quick_check(self, config: RunConfig, token: object) -> None:
        try:
            # Perf probes and other diagnostics must never overlap a formal
            # measurement, including one owned by another AC-Prof process.
            with MeasurementLock():
                checks = self._quick_preflight(config, project_dir=self._project_dir)
            error = ""
        except Exception as exc:
            checks = []
            error = f"{type(exc).__name__}: {exc}"
        self._safe_process_callback(self._show_quick_check, checks, error, token)

    def _show_quick_check(
        self,
        checks: Sequence[PreflightCheck],
        error: str,
        token: object,
    ) -> None:
        if not self.is_running or not self._form_ready or self._ui_closing or token is not self._check_request:
            return
        self._check_running = False
        self._check_request = None
        self._check_completed = True
        self._checked_config = self._check_config
        self._preflight_checks = tuple(checks)
        self._preflight_error = error
        self._set_busy(False)
        self._finish_recovery_check()
