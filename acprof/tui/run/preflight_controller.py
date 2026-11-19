"""Preflight, preparation dialogs and bounded retry interactions.

Only the App owns the active process, measurement-window state and worker
cancellation; this controller owns presentation and preflight requests.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from textual.timer import Timer
from textual.widgets import Button, Static
from textual.worker import Worker, get_current_worker

from acprof.events.download import DownloadProgress
from acprof.events.image import ImageProgress
from acprof.experiments.configuration import RunConfig, RunConfigError
from acprof.host.preparation.preparation_log import redact_text
from acprof.host.run_state import RunStateError, offline_analysis_lock
from acprof.host.runtimes.contracts import RuntimeCleanupError
from acprof.messages import join_messages, message
from acprof.tui.i18n import message_id
from acprof.tui.run import form as run_form, planning as run_planning
from acprof.tui.run.diagnostics import PreflightCheck, permission_signature, quick_preflight
from acprof.tui.run.environment_checks import scan_environment
from acprof.tui.run.global_status import GlobalStatus, build_status
from acprof.tui.run.monitor import MonitorPhase, monitor_phase
from acprof.tui.ui.environment import EnvironmentIndicators, EnvironmentPopover

if TYPE_CHECKING:
    from acprof.tui.run.preparation import PreparationScreen



if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui

class PreflightController:
    """Own preflight state and operations; the App owns lifecycle and scheduling."""

    AUTO_CHECK_DELAY = 0.5
    PERMISSION_CHECK_IDS = frozenset({'cpu_instructions', 'cpu_energy', 'tcpdump', 'capture_permissions'})

    def __init__(self, app: AcprofTui) -> None:
        self.app = app
        self.check_running = False
        self.resource_phase = "checking"
        self.check_request: object | None = None
        self.check_completed = False
        self.checked_permissions: tuple | None = None
        self.check_config: RunConfig | None = None
        self.check_inventory = True
        self.checked_config: RunConfig | None = None
        self.auto_check_config: RunConfig | None = None
        self.auto_check_timer: Timer | None = None
        self.auto_check_due = False
        self.checks: tuple[PreflightCheck, ...] = ()
        self.error = ""
        self.observed_checks: dict[str, PreflightCheck] = {}
        self.environment_checks: tuple[PreflightCheck, ...] = ()
        self.environment_error = ""
        self.cleanup_error = ""
        self.completed_at: datetime | None = None
        self.lock_acquired = False
        self.preparation_screen: PreparationScreen | None = None
        self.preparation_cancelled = False
        self.pending_source_change: dict[str, str] | None = None
        self.preparation_request: tuple[subprocess.Popen[str], int] | None = None
        self.latest_download: DownloadProgress | ImageProgress | None = None

    def collection_issue(self, event: dict) -> None:
        name = event["collector"]
        identifier = "cgroup_metrics" if name == "cgroup" else name
        detail = event["detail"] + " · " + ", ".join(event["fields"])
        self.observed_checks[identifier] = PreflightCheck(name, "warn", detail, check_id=identifier,
                                                         source="result_layers.json")
        self.refresh_preflight_state()

    def reset_download(self) -> None:
        self.latest_download = None
        from acprof.tui.ui.progress_widgets import DownloadProgressPanel, ImageProgressContent
        self.app.query_one(ImageProgressContent).reset_accounting()
        self.app.query_one(DownloadProgressPanel).update_plan("")
        self.app.query_one("#download-progress").display = False

    def download_progress(self, progress: DownloadProgress | ImageProgress) -> None:
        if self.app.execution.session.snapshot.measurement_active:
            raise RuntimeError("download progress inside measurement window")
        from acprof.tui.ui.progress_widgets import DownloadProgressPanel
        previous = self.latest_download
        if (isinstance(progress, ImageProgress) and isinstance(previous, ImageProgress)
                and not progress.operation_id and previous.component == progress.component):
            # Content checks/fallback/terminal events carry no new byte evidence.
            progress = replace(progress, **{key: getattr(previous, key) for key in (
                "operation_id", "transferred_bytes", "known_bytes", "unknown_items", "cached_bytes",
                "cached_items", "reused_steps", "transfer_status", "elapsed_seconds", "idle_seconds",
                "stall_after_seconds", "uncovered_sources",
            )})
            if progress.image == previous.image or progress.phase in {"fallback", "verifying", "complete", "failed", "cancelled"}:
                retained = {key: getattr(progress, key) or getattr(previous, key) for key in (
                    "source_image", "source_registry", "source_domain", "source_kind", "source_category",
                    "network_owner", "detail",
                )}
                progress = replace(progress, activity=previous.activity, **retained)
        if isinstance(progress, ImageProgress) and progress.phase in {"failed", "cancelled"}:
            progress = replace(progress,
                               transfer_status="partial" if progress.transferred_bytes is not None else "unavailable",
                               uncovered_sources=tuple(dict.fromkeys((*progress.uncovered_sources, "incomplete_stream"))))
        self.latest_download = progress
        widget = self.app.query_one("#download-progress", DownloadProgressPanel)
        widget.display = monitor_phase(self.app.execution.session.snapshot) is MonitorPhase.PREPARATION
        widget.update_progress(progress)

    def finish_download(self, *, cancelled: bool) -> None:
        if isinstance(self.latest_download, ImageProgress) and self.latest_download.phase not in {"complete", "failed", "cancelled"}:
            self.download_progress(replace(self.latest_download, phase="cancelled" if cancelled else "failed",
                                           current=None, total=None))
        elif isinstance(self.latest_download, DownloadProgress) and self.latest_download.state in {"downloading", "verifying"}:
            self.download_progress(replace(self.latest_download, state="cancelled" if cancelled else "failed",
                                           rate_bytes_s=None))

    @property
    def selected_runtime(self) -> str:
        if self.checked_config is None:
            return ""
        # A cached auto resolution belongs to its configuration, including edits
        # made while a worker was running. Never reuse it in a new run plan.
        if not self.config_matches():
            return ""
        return next((check.runtime_kind for check in self.checks if check.runtime_kind), "")

    @property
    def panel(self) -> EnvironmentPopover:
        return self.app.query_one(EnvironmentPopover)

    @property
    def inventory_running(self) -> bool:
        return self.check_running and self.check_inventory

    @property
    def status(self) -> GlobalStatus:
        return build_status(
            self.preflight_config(), issues=self.app.configuration.config_issues,
            warnings=self.app.configuration.config_warnings,
            validation_pending=self.app.configuration.validation_pending,
            checks=(*self.checks, *self.observed_checks.values()) if self.config_matches() else self.checks, inventory=self.environment_checks,
            completed=self.check_completed, current=self.config_matches(),
            running=self.check_running, inventory_running=self.inventory_running,
            error=self.error, inventory_error=self.environment_error, cleanup_error=self.cleanup_error,
        )

    @property
    def display_state(self) -> str:
        return self.status.state

    def network_preflight_report(self, line: str) -> None:
        if self.app.execution.session.snapshot.measurement_active:
            raise RuntimeError("download preflight inside measurement window")
        report = json.loads(line.split(" ", 1)[1])
        from acprof.tui.run.downloads import download_estimate, download_summary
        self.app.query_one("#network-download-summary", Static).update(download_summary(report, self.app.localization.tr))
        self.app.query_one("#download-estimate", Static).update(self.app.localization.tr(download_estimate(report)))
        from acprof.tui.ui.progress_widgets import DownloadProgressPanel
        self.app.query_one(DownloadProgressPanel).update_plan(self.app.localization.tr(
            message_id('downloads.size.expected_detail', download_estimate(report))))

    def network_download_report(self, line: str) -> None:

        from acprof.tui.run.downloads import download_result_summary
        report = json.loads(line.split(" ", 1)[1])
        summary = self.app.query_one("#network-download-summary", Static)
        summary.update(f"{summary.content}\n{self.app.localization.tr(download_result_summary(report))}")

    def open_model_store(self):
        if self.app.is_busy() or self.check_running or self.app.execution.session.snapshot.measurement_active:
            return
        from acprof.host.model_store.store import store_root
        from acprof.tui.models.store import ModelStoreScreen
        root = Path(self.app.configuration.input("model-store")).expanduser().resolve() if self.app.configuration.input("model-store") else store_root()
        self.app.configuration.environment_open = True
        self.app.set_busy(True)
        self.app.push_screen(ModelStoreScreen(root, self.app.configuration.input("model-store-max")), self.app.configuration.environment_closed)

    def show_preparation(self, event: dict) -> None:
        from acprof.tui.run.preparation import PreparationScreen
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
        if "image_progress" in event:
            self.download_progress(ImageProgress(**event["image_progress"]))
        if event.get("input_plan") is not None and self.app.execution.session.config is not None:
            if self.app.execution.session.snapshot.measurement_active:
                raise RuntimeError("input plan update inside measurement window")
            plan = event["input_plan"]
            from acprof.events.preparation import validate_input_plan
            validate_input_plan(plan)
            self.app.configuration.planned_input = plan
            self.app.configuration.planned_input_identity = run_planning.input_identity(self.app.execution.session.config)
            self.app.configuration.update_plan_summary(self.app.execution.session.config)
        if self.app.execution.session.stop_requested:
            return
        if event["stage"] == "runtime" and event.get("status") == "passed":
            self.app.write_log(self.app.localization.tr(message_id('models.preparation.check_complete')))
            self.close_preparation()
            return
        request = event.get("request")
        if request:
            if self.app.execution.session.snapshot.measurement_active or self.preparation_request is not None:
                raise RuntimeError("unexpected preparation request")
            process = self.app.execution.lifecycle.process
            if process is None or process.poll() is not None:
                return
            if request["kind"] == "error":
                self.app.write_log(f"[preparation][ERROR] {event['stage']}: {request.get('detail', '')}")
            self.preparation_request = (process, request["id"])
        if request:
            self.show_preparation(event)

    def preparation_answered(self, result) -> None:
        from acprof.events.preparation import encode_reply
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
            self.app.execution.request_stop(self.app.execution.session.token)
            if pending is None:
                self.app.stop_process_gracefully(self.app.execution.session.token, self.app.execution.lifecycle.process)
            self.close_preparation()
        if pending is None:
            return
        process, request_id = pending
        try:
            self.app.execution.lifecycle.reply(process, encode_reply(request_id, **result))
        except (OSError, ValueError, RuntimeError) as exc:
            self.app.notify(str(exc), severity="error")
            self.app.execution.request_stop(self.app.execution.session.token)
            self.app.stop_process_gracefully(self.app.execution.session.token, process)
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
            profiling_mode=self.app.configuration.recovery_policy,
            extra_options={key: value for key, value in self.app.configuration.extra_options.items()
                           if key in {"gpu_device", "dram_energy"}},
            gpus=self.app.configuration.select("gpus"),
            sniff_iface=self.app.configuration.input("sniff-iface"),
            compute_profile_tool=compute_tool,
            execution_profile_tool=execution_tool,
        )

    def preflight_run_reason(self, status: GlobalStatus | None = None) -> str:
        if not self.app.is_running or not self.app.form_ready or self.app.ui_closing:
            return message_id('run.preflight.checking_notice')
        status = self.status if status is None else status
        if status.can_start:
            return ""
        failures = [message("{0}: {1}", redact_text(self.app.tr(item.check.label)),
                            redact_text(self.app.tr(item.check.detail))) for item in status.blockers
                    if item.group == "error"]
        if failures:
            key = ('run.preflight.configuration_incomplete' if status.state == 'incomplete'
                   else 'run.preflight.collection_blocked')
            return message_id(key, join_messages("；", failures))
        if status.state == 'checking':
            return message_id('run.preflight.checking_notice')
        if self.app.configuration.validation_pending:
            return message_id('run.preflight.validation_pending')
        return message_id('run.preflight.configuration_changed')

    def config_matches(self) -> bool:
        return self.same_requirements(self.preflight_config(), self.checked_config, self.checks)

    @staticmethod
    def same_requirements(current: RunConfig, checked: RunConfig | None,
                          checks: Sequence[PreflightCheck] = ()) -> bool:
        if checked is None:
            return False
        cached_runtime = next((check.runtime_kind for check in checks if check.runtime_kind), "")
        if current.profiling_mode != "best_effort" and (current.runtime == "docker" or (current.runtime == "auto" and cached_runtime == "docker")):
            # Docker preflight does not validate numeric matrix enforcement;
            # form validation and the runner retain that responsibility.
            current = replace(current, cpus="auto", mems="auto")
            checked = replace(checked, cpus="auto", mems="auto")
        return current == checked

    def cancel_auto_check(self) -> None:
        self.pause_auto_check()
        self.auto_check_config = None
        self.auto_check_due = False

    def pause_auto_check(self) -> None:
        if self.auto_check_timer is not None:
            self.auto_check_timer.stop()
            self.auto_check_timer = None

    def schedule_auto_check(self) -> None:
        """Coalesce valid edits; inventory and manual retries remain explicit."""
        if not self.app.is_running or not self.app.form_ready or self.app.ui_closing:
            return
        if not self.check_completed and not self.check_running:
            # Mount/result restoration owns the initial inventory request.
            return
        if self.app.configuration.config_issues or self.app.configuration.validation_pending or self.cleanup_error:
            self.cancel_auto_check()
            return
        config = self.preflight_config()
        if ((not self.check_running and self.check_completed and self.config_matches())
                or (self.check_running and self.same_requirements(config, self.check_config))):
            self.cancel_auto_check()
            return
        if not self.same_requirements(config, self.auto_check_config):
            self.cancel_auto_check()
            self.auto_check_config = config
        # Debounce can elapse while a worker is running, but never during a
        # measurement or preparation. Completion resumes only the latest edit.
        if not self.app.is_busy() and self.auto_check_timer is None and not self.auto_check_due:
            self.auto_check_timer = self.app.set_timer(self.AUTO_CHECK_DELAY, self.auto_check_elapsed)
        self.resume_auto_check()

    def auto_check_elapsed(self) -> None:
        self.auto_check_timer = None
        if self.auto_check_config is None or self.app.ui_closing or not self.app.form_ready:
            return
        self.auto_check_due = True
        self.resume_auto_check()

    def resume_auto_check(self) -> None:
        if (self.auto_check_config is None or not self.app.is_running or not self.app.form_ready
                or self.app.ui_closing or self.app.configuration.validation_pending
                or self.app.configuration.config_issues or not self.app.operation_state().allows("check")):
            return
        if self.auto_check_due:
            self.quick_check(automatic=True)
        elif self.auto_check_timer is None:
            self.auto_check_timer = self.app.set_timer(self.AUTO_CHECK_DELAY, self.auto_check_elapsed)

    @property
    def collection_state(self) -> str:
        return self.status.state

    def check_permission_changes(self) -> None:
        if self.checked_permissions is None or self.app.is_busy() or self.app.ui_closing:
            return
        try:
            with offline_analysis_lock():
                current = permission_signature(self.app.project_dir)
        except RunStateError:
            current = None
        if current != self.checked_permissions:
            self.checked_permissions = None
            self.checked_config = None
            self.schedule_auto_check()
            self.refresh_preflight_state()

    def allow_collection(self) -> bool:
        self.check_permission_changes()
        status = self.status
        reason = self.preflight_run_reason(status)
        if reason:
            self.refresh_preflight_state(status)
            return False
        return True

    def refresh_preflight_state(self, status: GlobalStatus | None = None) -> None:
        if not self.app.is_running or not self.app.form_ready or self.app.ui_closing:
            return
        status = self.status if status is None else status
        reason = self.preflight_run_reason(status)
        entry = self.app.query_one(EnvironmentIndicators)
        entry.update_cached(status)
        self.panel.update_cached(status)
        detail = self.app.query_one("#preflight-run-reason", Static)
        state = status.state
        detail.display = False  # The global center and Start tooltip own admission feedback.
        summary = message_id('run.preflight.state.limited') if state == 'limited' else message_id('run.preflight.state.ready')
        self.app.localization.set_text(detail, reason or summary)
        detail.set_classes('preflight-' + state)
        process_runtime = self.selected_runtime == "process" or self.app.configuration.select("runtime") == "process"
        from acprof.tui.run import runtime_view
        if not self.app.execution.session.snapshot.measurement_active:
            runtime_view.refresh(self)
            from acprof.tui.run.issue_view import IssueDiagnosticScreen
            for screen in self.app.screen_stack:
                if isinstance(screen, IssueDiagnosticScreen) and screen.is_mounted:
                    screen.update_cached(status)
            self.app.configuration.update_field_errors(self.app.configuration.config_issues)
        probe = self.app.query_one("#probe-largest", Button)
        if process_runtime:
            probe.disabled = True
            self.app.localization.set_text(probe, "Process 请使用输入规模扫描；最小内存探测需要 Docker。", "tooltip")
        start = self.app.query_one("#start-run", Button)
        start.disabled = not status.can_start or not self.app.operation_state().allows("run")
        self.app.localization.set_text(start, reason or "开始采集", "tooltip")
        self.app.refresh_bindings()

    def quick_check(self, *, automatic: bool = False) -> None:
        if not self.app.is_running or not self.app.form_ready or self.app.ui_closing:
            return
        if self.check_running:
            return
        if automatic and (self.app.configuration.validation_pending or self.app.configuration.config_issues or self.cleanup_error
                          or (self.check_completed and self.config_matches())):
            self.cancel_auto_check()
            return
        if not self.app.operation_state().allows("check"):
            if not automatic:
                self.app.allow_operation("check")
            return
        if automatic:
            # Changed messages can still be queued behind this timer. Validate
            # the actual form before admitting probes for its current snapshot.
            try:
                self.app.configuration.collect_config()
            except RunConfigError as exc:
                self.cancel_auto_check()
                self.app.configuration.update_field_errors(exc.issues)
                self.refresh_preflight_state()
                return
        self.cancel_auto_check()
        self.check_config = self.preflight_config()
        self.check_inventory = not automatic
        self.check_running = True
        self.resource_phase = "checking"
        self.check_request = object()
        # Keep the current page and editable form. Only conflicting work waits.
        self.app.set_busy(False)
        self.execute_quick_check(self.check_config, self.check_request)

    def execute_quick_check(self, config: RunConfig, token: object) -> Worker[None]:
        """Schedule this operation on the owning App worker manager."""
        return self.app.run_worker(
            partial(self._execute_quick_check_worker, config, token, refresh_environment=self.check_inventory),
            name="execute_quick_check", thread=True, group='preflight', exclusive=True, exit_on_error=False,
        )

    def _execute_quick_check_worker(self, config: RunConfig, token: object, *, refresh_environment: bool = True) -> None:
        worker = get_current_worker()
        lock_acquired = False
        environment_checks: Sequence[PreflightCheck] | None = None
        environment_error = ""
        cleanup_error = "" if refresh_environment else None
        permissions = None
        try:
            # Perf probes and other diagnostics must never overlap a formal
            # Linux measurement, including one owned by another AC-Prof process.
            with offline_analysis_lock():
                lock_acquired = True
                if worker.is_cancelled:
                    return
                if refresh_environment:
                    environment_checks = ()
                    try:
                        environment_checks = scan_environment(project_dir=self.app.project_dir)
                    except RuntimeCleanupError:
                        raise
                    except Exception as exc:
                        environment_error = f"{type(exc).__name__}: {exc}"
                if worker.is_cancelled:
                    return
                if config.runtime in {"process", "auto"} and (config.cpus != "auto" or config.mems != "auto"):
                    self.app.process.safe_process_callback(self.show_resource_repair, token)
                checks = quick_preflight(config, project_dir=self.app.project_dir)
                if config.profiling_mode == "best_effort":
                    permissions = permission_signature(self.app.project_dir)
            error = ""
        except Exception as exc:
            checks = []
            error = f"{type(exc).__name__}: {exc}"
            if refresh_environment and (not lock_acquired or isinstance(exc, RuntimeCleanupError)):
                environment_error = error
            if isinstance(exc, RuntimeCleanupError):
                cleanup_error = error
        if worker.is_cancelled:
            return
        self.app.process.safe_process_callback(
            self.show_quick_check, checks, error, token, lock_acquired, environment_checks, environment_error, cleanup_error, permissions,
        )

    def show_resource_repair(self, token: object) -> None:
        if token is self.check_request and self.check_running and not self.app.execution.session.snapshot.measurement_active:
            self.resource_phase = "repairing"
            self.refresh_preflight_state()

    def invalidate_permission_checks(self) -> None:
        """System changes invalidate their evidence before any repair is attempted."""
        if self.app.execution.session.snapshot.measurement_active:
            return
        def invalidate(rows: tuple[PreflightCheck, ...]) -> tuple[PreflightCheck, ...]:
            return tuple(replace(row, status='pending') if row.check_id in self.PERMISSION_CHECK_IDS else row
                         for row in rows)
        self.checks = invalidate(self.checks)
        self.environment_checks = invalidate(self.environment_checks)
        self.refresh_preflight_state()

    def begin_permission_check(self) -> object | None:
        if self.check_running or self.app.execution.session.snapshot.measurement_active or self.app.ui_closing:
            return None
        self.cancel_auto_check()
        self.check_config = self.preflight_config()
        self.check_request = object()
        self.check_running = True
        self.check_inventory = False
        self.invalidate_permission_checks()
        return self.check_request

    def finish_permission_check(self, checks: Sequence[PreflightCheck], error: str, token: object,
                                *, cleanup_error: bool = False) -> None:
        """Merge only measured permission capabilities, never approve a whole run."""
        if token is not self.check_request or self.app.ui_closing:
            return
        current = self.config_matches() and self.same_requirements(self.preflight_config(), self.check_config, self.checks)
        updates = {row.check_id: row for row in checks if row.check_id in self.PERMISSION_CHECK_IDS}
        missing_tcpdump = 'tcpdump' in updates and updates['tcpdump'].status == 'fail'

        def merged(rows: tuple[PreflightCheck, ...], *, inventory: bool) -> tuple[PreflightCheck, ...]:
            existing = {row.check_id for row in rows}
            kept = tuple(row for row in rows if row.check_id not in updates and row.check_id != 'permission_check'
                         and not (missing_tcpdump and row.check_id == 'capture_permissions'))
            return kept + tuple(row for key, row in updates.items() if inventory or key in existing
                                or (key == 'tcpdump' and missing_tcpdump and 'capture_permissions' in existing))

        self.environment_checks = merged(self.environment_checks, inventory=True)
        if current:
            self.checks = merged(self.checks, inventory=False)
        if error:
            self.environment_checks += (PreflightCheck(message_id('settings.permissions.check_failed'), 'warn', error,
                                                       check_id='permission_check', source='permission_check'),)
        if cleanup_error:
            self.cleanup_error = error
        self.check_running = False
        self.check_request = None
        if self.check_config and self.check_config.profiling_mode == "best_effort":
            self.checked_config = None
            self.schedule_auto_check()
        self.refresh_preflight_state()

    def show_quick_check(
        self,
        checks: Sequence[PreflightCheck],
        error: str,
        token: object,
        lock_acquired: bool = False,
        environment_checks: Sequence[PreflightCheck] | None = None,
        environment_error: str = "",
        cleanup_error: str | None = None,
        permissions: tuple | None = None,
    ) -> None:
        if not self.app.is_running or not self.app.form_ready or self.app.ui_closing or token is not self.check_request:
            return
        self.check_running = False
        self.check_request = None
        self.check_completed = True
        self.observed_checks.clear()
        self.checked_config = self.check_config
        self.checked_permissions = permissions
        self.checks = tuple(checks)
        self.error = error
        if cleanup_error is not None:
            self.cleanup_error = cleanup_error
        # Inventory is informational; only requirements of this configuration block.
        if environment_checks is not None or environment_error:
            self.environment_checks = tuple(environment_checks or ())
            self.environment_error = environment_error
            self.completed_at = datetime.now().astimezone()
            self.lock_acquired = lock_acquired
        self.app.updates.start_check()
        self.app.set_busy(False)
        # Preflight only retains diagnostic fields, not the user's workload or
        # window counts. Re-read the editable form with the resolved runtime.
        self.app.configuration.refresh_command_preview(notify=False)
        self.app.recovery.finish_recovery_check()
