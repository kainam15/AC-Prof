"""Preflight, preparation dialogs and bounded retry interactions.

Only the App owns the active process, measurement-window state and worker
cancellation; this controller owns presentation and preflight requests.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from textual.widgets import Button, Static
from textual.worker import Worker

from acprof.download_events import DownloadProgress
from acprof.experiment import RunConfig
from acprof.host.run_state import MeasurementLock
from acprof.messages import join_messages, message
from acprof.tui import run_form, run_planning
from acprof.tui.diagnostics import PreflightCheck, quick_preflight
from acprof.tui.i18n import message_id
from acprof.tui.views import EnvironmentPreflightScreen

if TYPE_CHECKING:
    from acprof.tui.preparation import PreparationScreen



if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui

class PreflightController:
    """Own preflight state and operations; the App owns lifecycle and scheduling."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app
        self.check_running = False
        self.check_request: object | None = None
        self.check_completed = False
        self.check_config: RunConfig | None = None
        self.checked_config: RunConfig | None = None
        self.checks: tuple[PreflightCheck, ...] = ()
        self.error = ""
        self.preparation_screen: PreparationScreen | None = None
        self.preparation_cancelled = False
        self.pending_source_change: dict[str, str] | None = None
        self.preparation_request: tuple[subprocess.Popen[str], int] | None = None
        self.latest_download: DownloadProgress | None = None

    def reset_download(self) -> None:
        self.latest_download = None
        self.app.query_one("#download-progress", Static).display = False

    def download_progress(self, progress: DownloadProgress) -> None:
        if self.app.latest_snapshot.measurement_active:
            raise RuntimeError("download progress inside measurement window")
        from acprof.tui.downloads import live_download_summary
        self.latest_download = progress
        widget = self.app.query_one("#download-progress", Static)
        widget.display = True
        widget.update(live_download_summary(progress, self.app.localization.tr))

    def finish_download(self, *, cancelled: bool) -> None:
        if self.latest_download is not None and self.latest_download.state in {"downloading", "verifying"}:
            self.download_progress(replace(self.latest_download, state="cancelled" if cancelled else "failed",
                                           rate_bytes_s=None))

    @property
    def selected_runtime(self) -> str:
        return next((check.runtime_kind for check in self.checks if check.runtime_kind), "")

    def show_environment_status(self) -> None:
        lines = []
        reason = self.preflight_run_reason()
        if reason:
            lines.append(reason)
        lines.extend(message("[{0}] {1}: {2}",
                             check.capability_status or check.status,
                             check.label, check.detail)
                     for check in self.checks)
        if not lines:
            return
        self.app.push_screen(EnvironmentPreflightScreen(
            join_messages("\n\n", lines),
            retry_disabled=not self.app.operation_state().allows("check"),
        ), self.preflight_dialog_closed)

    def preflight_dialog_closed(self, retry: bool | None) -> None:
        if retry:
            self.quick_check()

    def network_preflight_report(self, line: str) -> None:
        report = json.loads(line.split(" ", 1)[1])
        from acprof.tui.downloads import download_summary
        self.app.query_one("#network-download-summary", Static).update(download_summary(report, self.app.localization.tr))
        from acprof.tui.presentation import format_bytes
        self.app.query_one("#download-estimate", Static).update(format_bytes(report.get("expected_download_bytes")))

    def network_download_report(self, line: str) -> None:

        from acprof.tui.downloads import download_result_summary
        report = json.loads(line.split(" ", 1)[1])
        summary = self.app.query_one("#network-download-summary", Static)
        summary.update(f"{summary.content}\n{self.app.localization.tr(download_result_summary(report))}")

    def open_model_store(self):
        if self.app.is_busy() or self.check_running or self.app.latest_snapshot.measurement_active:
            return
        from acprof.host.model_store import store_root
        from acprof.tui.model_store import ModelStoreScreen
        root = Path(self.app.configuration.input("model-store")).expanduser().resolve() if self.app.configuration.input("model-store") else store_root()
        self.app.configuration.environment_open = True
        self.app.set_busy(True)
        self.app.push_screen(ModelStoreScreen(root, self.app.configuration.input("model-store-max")), self.app.configuration.environment_closed)

    def show_preparation(self, event: dict) -> None:
        from acprof.tui.preparation import PreparationScreen
        if self.preparation_screen is None:
            self.preparation_screen = PreparationScreen(event, respond=self.preparation_answered)
            self.app.push_screen(self.preparation_screen)
        else:
            self.preparation_screen.update_event(event)

    def close_preparation(self) -> None:
        screen = self.preparation_screen
        self.preparation_screen = None
        self.preparation_request = None
        if screen is not None:
            screen.close()

    def preparation_event(self, event: dict) -> None:
        if event.get("input_plan") is not None and self.app.active_run_config is not None:
            if self.app.latest_snapshot.measurement_active:
                raise RuntimeError("input plan update inside measurement window")
            plan = event["input_plan"]
            from acprof.preparation_events import validate_input_plan
            validate_input_plan(plan)
            self.app.configuration.planned_input = plan
            self.app.configuration.planned_input_identity = run_planning.input_identity(self.app.active_run_config)
            self.app.configuration.update_plan_summary(self.app.active_run_config)
        if self.app.stop_requested:
            return
        if event["stage"] == "runtime" and event.get("status") == "passed":
            self.app.write_log(self.app.localization.tr(message_id('ui.0060')))
            self.close_preparation()
            return
        request = event.get("request")
        if request:
            if self.app.latest_snapshot.measurement_active or self.preparation_request is not None:
                raise RuntimeError("unexpected preparation request")
            process = self.app.lifecycle.process
            if process is None or process.poll() is not None:
                return
            if request["kind"] == "error":
                self.app.write_log(f"[preparation][ERROR] {event['stage']}: {request.get('detail', '')}")
            self.preparation_request = (process, request["id"])
        if request:
            self.show_preparation(event)

    def preparation_answered(self, result) -> None:
        from acprof.preparation_events import encode_reply
        if result["action"] == "request-stop":
            self.app.action_request_stop()
            return
        pending = self.preparation_request
        if result["action"] == "switch-source":
            self.pending_source_change = result
            result = {"action": "cancel"}
        self.preparation_request = None
        if result["action"] == "cancel":
            self.preparation_cancelled = True
            self.app.stop_requested = True
            if pending is None:
                self.app.stop_process_gracefully(self.app.process_token, self.app.lifecycle.process)
            self.close_preparation()
        if pending is None:
            return
        process, request_id = pending
        try:
            self.app.lifecycle.reply(process, encode_reply(request_id, **result))
        except (OSError, ValueError, RuntimeError) as exc:
            self.app.notify(str(exc), severity="error")
            self.app.stop_requested = True
            self.app.stop_process_gracefully(self.app.process_token, process)
        else:
            if result["action"] != "cancel":
                self.close_preparation()

    def preflight_config(self) -> RunConfig:
        # Diagnostics remain usable without a model or valid resource matrix.
        compute_tool, execution_tool = run_form.profiler_tools_from_checks(
            {key: self.app.configuration.checked(key) for key in run_form.PROFILER_CHECKBOXES}
        )
        return RunConfig(
            model="preflight-only",
            runtime=self.app.configuration.select("runtime"),
            cpus=self.app.configuration.input("cpus"), mems=self.app.configuration.input("mems"),
            cpuset_cpus=self.app.configuration.input("cpuset-cpus"),
            profiling_mode=self.app.configuration.select("profiling-mode"),
            gpus=self.app.configuration.select("gpus"),
            sniff_iface=self.app.configuration.input("sniff-iface"),
            compute_profile_tool=compute_tool,
            execution_profile_tool=execution_tool,
        )

    def preflight_run_reason(self) -> str:
        if not self.app.is_running or not self.app.form_ready or self.app.ui_closing:
            return message_id('ui.0599')
        if self.check_running or not self.check_completed:
            return message_id('ui.0599')
        if self.error:
            return message_id('ui.0600', self.error)
        if not self.config_matches():
            return message_id('ui.0601')
        failures = [message("{0}: {1}", check.label, check.detail)
                    for check in self.checks if check.status == "fail"]
        return message_id('ui.0602', join_messages("；", failures)) if failures else ""

    def config_matches(self) -> bool:
        current, checked = self.preflight_config(), self.checked_config
        if checked is None:
            return False
        if current.runtime != "process" and self.selected_runtime != "process":
            # Docker preflight does not validate numeric matrix enforcement;
            # form validation and the runner retain that responsibility.
            current = replace(current, cpus="auto", mems="auto", cpuset_cpus="")
            checked = replace(checked, cpus="auto", mems="auto", cpuset_cpus="")
        return current == checked

    def allow_collection(self) -> bool:
        reason = self.preflight_run_reason()
        if reason:
            self.refresh_preflight_state()
            self.app.notify(reason, title="环境状态", severity="warning", timeout=8)
            return False
        return True

    def refresh_preflight_state(self) -> None:
        if not self.app.is_running or not self.app.form_ready or self.app.ui_closing:
            return
        reason = self.preflight_run_reason()
        issues = [check for check in self.checks if check.status in {"warn", "fail"}]
        stale = self.check_completed and not self.config_matches()
        entry = self.app.query_one("#environment-status", Button)
        entry.display = bool(issues or self.error or stale or self.selected_runtime == "process")
        failed = bool(self.error or any(check.status == "fail" for check in issues))
        warning = bool(issues or stale)
        self.app.localization.set_text(entry, "环境异常" if failed else "环境警告" if warning else "环境能力", "label")
        entry.variant = "error" if failed else "warning" if warning else "default"
        self.app.localization.set_text(entry, "查看环境问题与重新检查", "tooltip")
        detail = self.app.query_one("#preflight-run-reason", Static)
        detail.display = bool(reason)
        self.app.localization.set_text(detail, reason)
        inherited = self.app.query_one("#process-inherited-resources", Button)
        inherited.display = self.selected_runtime == "process" or self.app.configuration.select("runtime") == "process"
        probe = self.app.query_one("#probe-largest", Button)
        if self.selected_runtime == "process" or self.app.configuration.select("runtime") == "process":
            probe.disabled = True
            self.app.localization.set_text(probe, "Process 请使用输入规模扫描；最小内存探测需要 Docker。", "tooltip")
        start = self.app.query_one("#start-run", Button)
        start.disabled = bool(reason) or not self.app.operation_state().allows("run")
        self.app.localization.set_text(start, reason or "开始采集", "tooltip")
        self.app.refresh_bindings()

    def quick_check(self) -> None:
        if not self.app.is_running or not self.app.form_ready or self.app.ui_closing:
            return
        if self.check_running:
            return
        if not self.app.allow_operation("check"):
            return
        self.check_config = self.preflight_config()
        self.check_running = True
        self.check_request = object()
        # Keep the current page and editable form. Only conflicting work waits.
        self.app.set_busy(False)
        self.execute_quick_check(self.check_config, self.check_request)

    def execute_quick_check(self, config: RunConfig, token: object) -> Worker[None]:
        """Schedule this operation on the owning App worker manager."""
        return self.app.run_worker(
            partial(self._execute_quick_check_worker, config, token),
            name="execute_quick_check", thread=True, group='preflight', exclusive=True, exit_on_error=False,
        )

    def _execute_quick_check_worker(self, config: RunConfig, token: object) -> None:
        try:
            # Perf probes and other diagnostics must never overlap a formal
            # measurement, including one owned by another AC-Prof process.
            with MeasurementLock():
                checks = quick_preflight(config, project_dir=self.app.project_dir)
            error = ""
        except Exception as exc:
            checks = []
            error = f"{type(exc).__name__}: {exc}"
        self.app.process.safe_process_callback(self.show_quick_check, checks, error, token)

    def show_quick_check(
        self,
        checks: Sequence[PreflightCheck],
        error: str,
        token: object,
    ) -> None:
        if not self.app.is_running or not self.app.form_ready or self.app.ui_closing or token is not self.check_request:
            return
        self.check_running = False
        self.check_request = None
        self.check_completed = True
        self.checked_config = self.check_config
        self.checks = tuple(checks)
        self.error = error
        self.app.set_busy(False)
        if self.checked_config is not None:
            self.app.configuration.update_plan_summary(self.checked_config)
        self.app.recovery.finish_recovery_check()
