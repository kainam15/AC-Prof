"""AC-Prof TUI 事件处理与子进程生命周期。"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Literal

from acprof.tui.i18n import message_id

try:
    from rich.console import Console
    from textual import events, on, work
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.geometry import Size
    from textual.theme import Theme
    from textual.timer import Timer
    from textual.widgets import (
        Button,
        Checkbox,
        Collapsible,
        DataTable,
        Header,
        Input as TextualInput,
        Select,
        Static,
        TabbedContent,
        Tabs,
        Tree,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - exercised before tests install deps
    if exc.name == "textual":
        raise SystemExit(
            "AC-Prof TUI 需要 Textual。请运行：\n"
            "  uv sync\n"
            "发行安装请运行 uv tool install --reinstall acprof。"
        ) from None
    raise

from acprof.experiment import RunConfig
from acprof.messages import join_messages, message
from acprof.tui.catalog_controller import CatalogController
from acprof.tui.commands import (
    OperationState,
    PendingLaunch,
    format_command,
)
from acprof.tui.configuration_controller import ConfigurationController
from acprof.tui.experiment_controller import ExperimentController
from acprof.tui.image_controller import ImageController
from acprof.tui.input import BarCursorApp, BarCursorInput as Input
from acprof.tui.localization_controller import LocalizationController
from acprof.tui.log import SelectableLog
from acprof.tui.log_controller import LogController
from acprof.tui.model_controller import ModelController
from acprof.tui.preflight_controller import PreflightController
from acprof.tui.presentation import NOT_APPLICABLE, UNKNOWN
from acprof.tui.process import ProcessLifecycle, StopResult
from acprof.tui.process_controller import ProcessController
from acprof.tui.profile_controller import ProfileController
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.recovery_controller import RecoveryController
from acprof.tui.rendering import CjkScreen
from acprof.tui.result_controller import ResultController
from acprof.tui.run_results import RunResult
from acprof.tui.slash_controller import SlashController
from acprof.tui.stages import Stage
from acprof.tui.table import ResizableDataTable
from acprof.tui.themes import THEME_CATALOG
from acprof.tui.views import (
    ConfirmActionScreen,
    compose_images_tab,
    compose_monitor_tab,
    compose_plot_tab,
    compose_profile_tab,
    compose_reports_tab,
    compose_run_tab,
    compose_settings_tab,
)

PROJECT_DIR = Path.cwd()
# Keep the virtual-environment path. Resolving this symlink would turn
# ``.venv/bin/python`` into the system interpreter and lose the venv.
PYTHON_EXECUTABLE = Path(sys.executable).absolute()


class AcprofTui(BarCursorApp):
    """Full-screen controller for AC-Prof collection and diagnostics."""

    TITLE = "AC-Prof"
    SUB_TITLE = "推理实验控制台"
    ENABLE_COMMAND_PALETTE = False
    ALLOW_IN_MAXIMIZED_VIEW = "Header"

    BINDINGS = [
        ("f5", "request_run", "开始采集"),
        Binding("f6", "quick_check", "环境检查", priority=True),
        ("f8", "toggle_log_view", "放大日志"),
        ("f2", "show_settings", "应用设置"),
        Binding("ctrl+x", "request_stop", "终止任务", priority=True),
        ("ctrl+l", "clear_log", "清空日志"),
        ("ctrl+q", "request_quit", "退出"),
    ]

    CSS_PATH = Path(__file__).with_name("tui.tcss")
    IMAGE_REFRESH_INTERVAL = 5.0

    @property
    def project_dir(self) -> Path:
        return PROJECT_DIR

    @property
    def python_executable(self) -> Path:
        return PYTHON_EXECUTABLE


    def __init__(
        self, initial_config: RunConfig | None = None, *, settings_path: Path | None = None,
        color_system: Literal["truecolor", "256", "auto"] = "truecolor",
    ):
        super().__init__(ansi_color=False)
        if color_system != "auto":
            # SSH often omits COLORTERM even when the client supports RGB.
            # Configure only this renderer; keep the subprocess environment intact.
            self.console: Console = Console(
                color_system=color_system,
                file=self.console.file,
                markup=True,
                highlight=False,
                emoji=False,
                legacy_windows=False,
                force_terminal=True,
                safe_box=False,
                soft_wrap=False,
                # Textual applies its own NO_COLOR filter to the whole frame.
                no_color=False,
            )
        self.animation_level = "none"
        self.lifecycle = ProcessLifecycle()
        self.process_kind = ""
        self.active_run_config: RunConfig | None = None
        self.active_command: tuple[str, ...] = ()
        self.started_monotonic = 0.0
        self.stop_requested = False
        self.latest_snapshot = ProgressSnapshot()
        self.ui_closing: bool = False
        self.read_jobs: set[object] = set()
        self.process_token: object | None = None
        self.run_result = RunResult()
        self.form_ready: bool = False
        self.elapsed_timer: Timer | None = None
        self.configuration = ConfigurationController(self, initial_config, settings_path=settings_path)
        self.localization = LocalizationController(self)
        self.experiments = ExperimentController(self)
        self.preflight = PreflightController(self)
        self.results = ResultController(self)
        self.profiles = ProfileController(self)
        self.recovery = RecoveryController(self)
        self.images = ImageController(self)
        self.models = ModelController(self)
        self.catalog = CatalogController(self)
        self.process = ProcessController(self)
        self.logs = LogController(self)
        self.slash = SlashController(self)
        for palette in THEME_CATALOG:
            self.register_theme(Theme(**palette.theme_kwargs()))
        self.theme = self.configuration.preferences.theme

    def get_default_screen(self) -> CjkScreen:
        return CjkScreen(id="_default")

    def compose(self) -> ComposeResult:
        # A ticking clock would force periodic redraws during RAPL windows.
        with Header(show_clock=False):
            yield self.localization.localized_widget(Button(
                "×", id="quit-app", name="退出", tooltip="退出（Ctrl+Q）", compact=True,
            ))
            yield Button("", id="environment-status", compact=True)
        with TabbedContent(initial="run-tab", id="main-tabs"):
            yield from compose_run_tab(self)
            yield from compose_monitor_tab(self)
            yield from compose_plot_tab(self)
            yield from compose_reports_tab(self)
            yield from compose_profile_tab(self)
            yield from compose_images_tab(self)
            yield from compose_settings_tab(self)

        with Vertical(id="bottom-panel"):
            with Horizontal(id="slash-command-bar"):
                yield self.localization.localized_widget(Input(
                    placeholder="快捷命令：输入 /help 查看可用命令，按 Enter 执行",
                    id="slash-command",
                ))

    def on_mount(self) -> None:
        self.configuration.configure_interaction()
        self.configuration.configure_scrollbars()
        self.localization.capture_language_text()
        self.configuration.apply_ui_preferences()
        self.localization.set_text(self.query_one('#settings-location', Static), message_id('ui.0729', self.configuration.settings_path))
        self.update_responsive_layout()
        self.configuration.selected_preset = self.configuration.select("run-preset")
        self.form_ready = True
        self.configuration.refresh_command_preview(notify=False)
        self.call_after_refresh(self.preflight.quick_check)
        self.query_one("#model", Input).focus()
        self.images.refresh_timer = self.set_interval(
            self.IMAGE_REFRESH_INTERVAL, self.images.refresh_images, pause=True,
        )
        if self.configuration.settings_warning:
            self.notify(self.configuration.settings_warning, title="设置读取提示", severity="warning", timeout=8)

    def on_resize(self, event: events.Resize) -> None:
        # App.size can still refer to the previous frame while Resize is
        # dispatched. Use the event's new dimensions for responsive classes.
        self.update_responsive_layout(event.size)
        if (self.form_ready and self.images.inventory is not None and not self.images.images_unavailable(allow_refresh=True)
                and self.query_one("#main-tabs", TabbedContent).active == "images-tab"):
            self.images.render_images(width=event.size.width)

    def update_responsive_layout(self, size: Size | None = None) -> None:
        size = self.size if size is None else size
        self.set_class(size.width < 110, "narrow")
        self.set_class(size.height < 35, "short")


    def is_busy(self) -> bool:
        return self.operation_state().busy

    def operation_state(self) -> OperationState:
        process = self.lifecycle.process
        return OperationState(
            process=process is not None or bool(self.process_kind),
            stoppable=process is not None and process.poll() is None,
            checking=self.preflight.check_running, reading=bool(self.read_jobs),
            maintenance=bool(self.images.operation) or self.images.storage_loading,
            configuring=self.configuration.environment_open or self.catalog.picker_open,
            measuring=self.latest_snapshot.measurement_active, closing=self.ui_closing,
        )

    def allow_operation(self, operation: str) -> bool:
        if self.operation_state().allows(operation):
            return True
        self.notify("请等待当前任务完成", severity="warning")
        return False

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        operation = {"request_run": "run", "request_probe": "probe",
                     "quick_check": "check", "request_stop": "stop"}.get(action)
        if action == "request_run" and self.preflight.preflight_run_reason():
            return False
        return self.operation_state().allows(operation) if operation else True

    def set_busy(self, busy: bool) -> None:
        if not self.is_running or not self.form_ready or self.ui_closing:
            return
        state = self.operation_state()
        busy = busy or state.busy
        # Configuration changes during a run can queue preview redraws and
        # make the visible settings differ from the running subprocess.
        if busy:
            self.configuration.cancel_preview_timer()
        for table in self.query(ResizableDataTable):
            refreshing_image = self.images.operation == "refresh" and table.has_class("image-control")
            table.resize_enabled = not ((busy and not refreshing_image) or self.latest_snapshot.measurement_active)
        for widget in self.query(
            ".config-control, .experiment-picker, #model-candidates, #run-preset, .ui-preference, .profile-tool, .report-control, "
            "#save-run-default, #restore-ui-defaults, #save-ui-settings, #open-environment-settings"
        ):
            if widget.id in {"report-source", "report-open"} and state.allows("report"):
                widget.disabled = False
                continue
            if busy and widget.has_focus:
                self.screen.set_focus(self.query_one("#main-tabs", TabbedContent).query_one(Tabs), scroll_visible=False)
            widget.disabled = busy or (state.checking and widget.id in {
                "open-environment-settings", "open-model-store",
            })
        for widget in self.query(".image-control"):
            widget.disabled = (busy and self.images.operation != "refresh") or self.latest_snapshot.measurement_active
        for selector, operation in {
            "#start-run": "run", "#probe-largest": "probe",
            "#review-run-recovery": "run",
            "#summarize-results": "summary", "#plot-results": "plot",
            "#profile-dry-run": "profile", "#profile-run": "profile", "#report-open": "report",
            "#report-calculate": "stats", "#stop-run": "stop",
            "#html-report-generate": "html-report", "#html-report-open": "html-open",
            "#html-report-export": "html-export",
        }.items():
            self.query_one(selector, Button).disabled = not state.allows(operation)
        self.query_one("#html-report-cancel", Button).disabled = not (
            self.process_kind == "html-report" and state.allows("stop"))
        for selector in ("#result-csv", "#report-source"):
            self.query_one(selector, Input).disabled = not state.allows("summary")
        self.refresh_bindings()
        if not busy:
            self.set_input_cursor_blink_enabled(True)
        self.images.update_image_controls()
        self.images.sync_image_refresh_timer()
        self.preflight.refresh_preflight_state()

    def activate_tab(self, tab_id: str) -> None:
        if self.screen.maximized is not None:
            self.screen.minimize()
        tabs = self.query_one("#main-tabs", TabbedContent)
        # Move focus before hiding the outgoing pane. Widget.focus() defers
        # this change, allowing an old pane's focus event to undo the switch.
        self.screen.set_focus(tabs.query_one(Tabs), scroll_visible=False)
        tabs.active = tab_id


    def launch(self, pending: PendingLaunch) -> None:
        if pending.kind == "run" and not self.preflight.allow_collection():
            return
        if not self.allow_operation("run"):
            return
        self.logs.flush()
        if pending.kind in {"run", "probe"}:
            with self.prevent(Checkbox.Changed):
                self.query_one("#rebuild-image", Checkbox).value = False
        model = pending.config.model if pending.config is not None else ""
        # Intended output is not evidence of a result. Keep the selected/history
        # paths until a matching run attempt has actually published artifacts.
        if pending.kind == "run":
            self.query_one("#run-recovery-actions").display = False
            self.configuration.planned_input = None
            self.configuration.planned_input_identity = None
            self.configuration.remember_last_used(model=model)
            self.run_result = RunResult()
            self.results.summary_request = None
            self.localization.set_text(self.query_one("#result-summary", Static),
                           "本次尚未产生结果；历史结果可通过原路径查看。")
        else:
            self.configuration.remember_last_used(model=model, result_dir=pending.result_dir, result_csv=pending.result_csv)
        self.process_token = object()
        self.active_run_config = pending.config if pending.kind == "run" else None
        self.results.comparison_report_path = Path(pending.report_path) if pending.kind == "compare" else None
        if pending.kind == "html-report":
            self.results.html_report_started(pending)
        self.active_command = pending.command
        self.process_kind = pending.kind
        self.started_monotonic = time.monotonic()
        self.stop_requested = False
        self.preflight.preparation_cancelled = False
        self.latest_snapshot = ProgressSnapshot(stage=Stage.STARTING, detail="正在创建子进程")
        self.preflight.reset_download()
        self.set_busy(True)
        self.activate_tab("monitor-tab")
        # Moving focus away from inputs stops the native cursor's blink timer.
        self.query_one("#stop-run", Button).focus()
        # Start the elapsed-time ticker; it self-gates on measurement_active
        # to avoid any redraws during formal energy/latency windows.
        if self.elapsed_timer is not None:
            self.elapsed_timer.stop()
        self.elapsed_timer = self.set_interval(1.0, self.tick_elapsed)
        log = self.query_one("#run-log", SelectableLog)
        log.write(f"$ {format_command(pending.command)}")
        log.write(self.localization.tr(message_id('ui.0766')))
        self.render_snapshot(self.latest_snapshot)
        self.process.execute_command(list(pending.command), pending.kind, config=pending.config)

    def tick_elapsed(self) -> None:
        """Update elapsed time display; skipped during measurement windows."""
        # Textual stops the app before removing widgets, ahead of on_unmount.
        if not self.is_running:
            return
        if self.latest_snapshot.measurement_active:
            return  # Zero redraws during RAPL/latency measurement windows.
        if not self.started_monotonic or not self.is_busy():
            return
        elapsed = self.format_elapsed(time.monotonic() - self.started_monotonic)
        self.localization.set_text(self.query_one('#status-elapsed', Static), elapsed)

    _STAGE_CSS_CLASS = {
        Stage.WAITING: "stage-idle",
        Stage.SERVER_READY: "stage-success",
        Stage.COMPLETED: "stage-success",
        Stage.PROBE_COMPLETED: "stage-success",
        Stage.MEMORY_FOUND: "stage-success",
        Stage.CASE_COMPLETED: "stage-success",
        Stage.FAILED: "stage-error",
        Stage.CLEANUP_INCOMPLETE: "stage-error",
        Stage.PROBE_FAILED: "stage-error",
        Stage.TASK_UNSUPPORTED: "stage-error",
        Stage.TERMINATED: "stage-error",
        Stage.STOPPED: "stage-error",
        Stage.PARTIAL: "stage-running",
        Stage.MEASURING: "stage-measuring",
        Stage.MAX_SCALE_PROBE: "stage-measuring",
    }
    _STAGE_CLASSES = frozenset({
        "stage-idle", "stage-running", "stage-measuring",
        "stage-success", "stage-error",
    })

    def process_cleanup_incomplete(self, result: StopResult) -> None:
        detail = message_id('ui.0259', result.pid, result.error)
        self.latest_snapshot = replace(self.latest_snapshot, stage=Stage.CLEANUP_INCOMPLETE, detail=detail)
        self.render_snapshot(self.latest_snapshot)
        self.write_log(detail)
        self.set_busy(True)
        self.notify(detail, severity="error", timeout=10)

    def process_started(self, pid: int, kind: str) -> None:
        self.set_busy(True)
        self.query_one("#run-log", SelectableLog).write(
            self.localization.tr(message_id('ui.0782', kind, pid))
        )

    def show_suppressed_count(self, count: int) -> None:
        self.query_one("#run-log", SelectableLog).write(
            self.localization.tr(message_id('ui.0783', count))
        )

    def show_deferred_lines(self, lines: tuple[str, ...]) -> None:
        log = self.query_one("#run-log", SelectableLog)
        log.write(self.localization.tr(message_id('ui.0784')))
        log.write("\n".join(lines))

    def write_log(self, line: str) -> None:
        self.query_one("#run-log", SelectableLog).write(self.localization.tr(line))

    def consume_process_line(
        self,
        line: str,
        snapshot: ProgressSnapshot | None,
        state_changed: bool,
    ) -> None:
        if self.process_kind == "html-report" and line:
            self.results.consume_html_line(line)
        if self.process_kind == "stats" and line.startswith("ACPROF_STATS "):
            receipt = json.loads(line.removeprefix("ACPROF_STATS "))
            if (not isinstance(receipt, dict) or not isinstance(receipt.get("report_path"), str)
                    or not receipt["report_path"] or type(receipt.get("reused")) is not bool):
                raise ValueError("统计进程返回的报告信息无效")
            self.results.stats_report_path = Path(receipt["report_path"])
            self.results.stats_report_reused = receipt["reused"]
            line = self.localization.tr(message(
                "已有相同报告：{0}" if self.results.stats_report_reused else "统计报告已保存：{0}",
                self.results.stats_report_path,
            ))
        if snapshot is not None:
            if self.process_kind == "run" and snapshot.stage is Stage.COMPLETED:
                snapshot = replace(snapshot, stage=Stage.VERIFYING_ARTIFACT, detail="进程已报告完成，等待本次产物核验。")
            was_measuring = self.latest_snapshot.measurement_active
            self.latest_snapshot = snapshot
            if was_measuring != snapshot.measurement_active:
                self.logs.suspend(snapshot.measurement_active)
            self.set_input_cursor_blink_enabled(not snapshot.measurement_active)
            self.images.sync_image_refresh_timer()
            if self.elapsed_timer is not None:
                if snapshot.measurement_active:
                    self.elapsed_timer.pause()
                elif was_measuring:
                    self.elapsed_timer.resume()
            if state_changed:
                self.render_snapshot(snapshot)
        if line:
            self.logs.append(line)

    @staticmethod
    def format_elapsed(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def render_snapshot(self, snapshot: ProgressSnapshot) -> None:
        self.logs.stage_changed(snapshot)
        from acprof.tui.preparation import phase_summary
        self.localization.set_text(self.query_one("#status-preparation", Static), phase_summary(snapshot))
        elapsed = (
            self.format_elapsed(time.monotonic() - self.started_monotonic)
            if self.started_monotonic
            else NOT_APPLICABLE
        )
        # Stage text with visual category coloring.
        stage_widget = self.query_one("#status-stage", Static)
        self.localization.set_text(stage_widget, snapshot.stage.label)
        stage_widget.set_classes(
            self._STAGE_CSS_CLASS.get(snapshot.stage, "stage-running")
        )
        self.localization.set_text(self.query_one('#status-elapsed', Static), elapsed)
        self.localization.set_text(self.query_one("#status-case", Static), message_id(
            'ui.0785', snapshot.current_case or NOT_APPLICABLE,
            snapshot.completed_cases, snapshot.total_cases,
        ))
        missing = UNKNOWN if snapshot.current_case else NOT_APPLICABLE
        cpu = snapshot.cpu if snapshot.cpu != "-" else missing
        mem = f"{snapshot.mem}GB" if snapshot.mem != "-" else missing
        gpu = snapshot.gpu if snapshot.gpu != "-" else missing
        self.localization.set_text(self.query_one("#status-resource", Static), message("CPU={0}  MEM={1}  GPU={2}", cpu, mem, gpu))
        self.localization.set_text(self.query_one('#status-errors', Static), f'{snapshot.warnings} / {snapshot.errors}')
        self.localization.set_text(self.query_one('#status-detail', Static), snapshot.detail)

    def process_finished(
        self,
        kind: str,
        returncode: int,
        snapshot: ProgressSnapshot | None,
        launch_error: str,
    ) -> None:
        # The child has exited; flush preparation output before its final outcome.
        self.logs.suspend(False)
        cancelled = self.stop_requested
        if kind == "html-report" and not cancelled:
            launch_error = self.results.html_report_error(returncode, launch_error)
        if kind == "stats" and returncode == 0 and not launch_error and self.results.stats_report_path is None:
            launch_error = "统计进程未返回报告路径"
        if kind == "probe" and returncode == 0 and not launch_error and not self.stop_requested:
            if (snapshot is None or snapshot.stage is not Stage.PROBE_COMPLETED
                    or not snapshot.probe_result or not snapshot.probe_summary):
                launch_error = self.localization.tr(message_id('ui.0183', "missing structured probe result"))
        comparison_ready = (kind == "compare" and returncode in {0, 1} and not launch_error
                            and not self.stop_requested and self.results.comparison_report_path is not None
                            and self.results.comparison_report_path.is_file())
        if kind == "compare" and returncode == 0 and not launch_error and not self.stop_requested and not comparison_ready:
            launch_error = "比较进程未返回报告路径"
        self.preflight.close_preparation()
        if snapshot is not None and snapshot.measurement_status == "running":
            snapshot = replace(snapshot, measurement_status=(
                "cancelled" if self.stop_requested else "failed" if returncode else "passed"))
        # Stop the elapsed-time ticker.
        if self.elapsed_timer is not None:
            self.elapsed_timer.stop()
            self.elapsed_timer = None
        # Show the final elapsed time.
        if self.started_monotonic:
            final_elapsed = self.format_elapsed(
                time.monotonic() - self.started_monotonic
            )
            self.localization.set_text(self.query_one('#status-elapsed', Static), final_elapsed)
        if kind in {"stats", "compare", "html-report"}:
            # Move focus before disabling the monitor page's focused Stop button;
            # its queued focus event could otherwise reactivate that old page.
            self.activate_tab("reports-tab")
        log = self.query_one("#run-log", SelectableLog)
        unsupported_task = (
            snapshot is not None and snapshot.stage is Stage.TASK_UNSUPPORTED
            and returncode != 0 and not self.stop_requested and not launch_error
        )
        if launch_error:
            log.write(self.localization.tr(message_id('ui.0786', launch_error)))
            self.notify(launch_error, title=message_id('ui.0986') if kind == "html-report" else "任务启动失败", severity="error", timeout=8)
        elif comparison_ready:
            notice: str = message_id('ui.0488')
            log.write(self.localization.tr(notice))
            self.notify(notice, severity="information", timeout=8)
        elif returncode == 0 and kind != "run" and not self.stop_requested:
            log.write(self.localization.tr(message_id('ui.0788', kind)))
            notice = (message_id('ui.0571', self.results.stats_report_path)
                      if kind == "stats" and self.results.stats_report_reused else "任务已完成")
            self.notify(notice, severity="information", timeout=8 if self.results.stats_report_reused else 5)
        elif self.stop_requested:
            log.write(self.localization.tr(message_id('ui.0790', returncode)))
            self.notify(message_id('ui.0962') if kind == "html-report" else "任务已终止；部分 case 结果可能仍可续跑", severity="warning", timeout=7)
        elif unsupported_task and snapshot is not None:
            log.write(self.localization.tr(snapshot.detail))
            log.write(self.localization.tr(message_id(
                'ui.0587'
            )))
            self.notify(snapshot.detail, title="任务不支持", severity="error", timeout=12)
        elif kind != "run":
            log.write(self.localization.tr(message_id('ui.0792', kind, returncode)))
            self.notify(message_id('ui.0793', returncode), severity="error", timeout=8)

        current_csv = ""
        if unsupported_task and snapshot is not None:
            self.latest_snapshot = replace(snapshot, measurement_active=False, measurement_status="failed")
            self.render_snapshot(self.latest_snapshot)
        elif kind == "run":
            result = self.run_result
            stage = result.stage(returncode, self.stop_requested, launch_error)
            final_state = snapshot or self.latest_snapshot
            detail: str = message_id('ui.0191', result.completed_cases,
                             result.total_cases or final_state.total_cases,
                             message_id('ui.0048') if stage is not Stage.COMPLETED else message_id('ui.0192'))
            self.query_one("#run-recovery-actions").display = stage is not Stage.COMPLETED
            if result.detail:
                detail = join_messages("\n", (detail, result.detail))
            if result.new_cases and result.retained_dir:
                detail = join_messages("\n", (detail, message_id('ui.0184', result.retained_dir)))
            if result.complete and not result.result_csv and not result.new_cases:
                detail = join_messages("\n", (detail, message_id('ui.0185')))
            self.latest_snapshot = replace(final_state, stage=stage, detail=detail,
                completed_cases=result.completed_cases, total_cases=result.total_cases or final_state.total_cases,
                measurement_active=False, measurement_status=("passed" if stage is Stage.COMPLETED else
                    "cancelled" if self.stop_requested else "failed"))
            self.render_snapshot(self.latest_snapshot)
            current_csv = result.result_csv
            if current_csv:
                self.configuration.remember_last_used(result_csv=current_csv, result_dir=str(Path(current_csv).parent))
            else:
                summary_text = (message_id('ui.0183', result.detail)
                                if result.detail and not result.belongs_to_attempt else
                                message_id('ui.0190', result.new_cases))
                self.localization.set_text(self.query_one("#result-summary", Static), summary_text)
            self.notify(message_id('ui.0193', stage.label, detail),
                        severity="information" if stage is Stage.COMPLETED else "warning" if stage in {Stage.PARTIAL, Stage.STOPPED} else "error")
        elif kind == "probe":
            final_state = snapshot or self.latest_snapshot
            if launch_error or (returncode != 0 and not self.stop_requested):
                detail = launch_error or (
                    final_state.detail
                    if final_state.stage is Stage.PROBE_FAILED
                    else message_id('ui.0796', returncode)
                )
                final_state = replace(
                    final_state,
                    stage=Stage.PROBE_FAILED,
                    detail=detail,
                    measurement_active=False,
                )
            elif self.stop_requested:
                final_state = replace(
                    final_state,
                    stage=Stage.TERMINATED,
                    detail="用户终止了最大尺度探测",
                    measurement_active=False,
                )
            self.latest_snapshot = final_state
            self.render_snapshot(final_state)
        else:
            if launch_error or (returncode != 0 and not self.stop_requested and not comparison_ready):
                stage = Stage.FAILED
                detail = launch_error or message_id('ui.0799', kind, returncode)
            elif self.stop_requested:
                stage = Stage.TERMINATED
                detail = message_id('ui.0800', kind)
            else:
                stage = Stage.COMPLETED
                detail = message_id('ui.0801', kind)
            self.latest_snapshot = ProgressSnapshot(stage=stage, detail=detail)
            self.render_snapshot(self.latest_snapshot)

        self.preflight.finish_download(cancelled=self.stop_requested)
        self.logs.flush()
        self.active_command = ()
        self.process_kind = ""
        self.stop_requested = False
        self.set_busy(self.is_busy())
        if self.preflight.preparation_cancelled and not self.is_busy():
            self.activate_tab("run-tab")
            self.preflight.preparation_cancelled = False
            if self.preflight.pending_source_change is not None:
                from uuid import uuid4
                change, self.preflight.pending_source_change = self.preflight.pending_source_change, None
                config = self.active_run_config or self.configuration.initial_config
                self.configuration.apply_config(replace(config, model=change["model_id"], revision=change["revision"],
                    model_source="modelscope", download_mode="auto", task="", task_family="", backend="",
                    model_spec="", resume=False, rebuild_image=False,
                    output_dir=str(Path(config.output_dir) / ("modelscope-" + uuid4().hex[:8]))))
        if current_csv:
            self.results.update_result_summary(current_csv, notify=False)
        self.images.sync_image_refresh_timer()
        if kind == "stats":
            report_path, self.results.stats_report_path = self.results.stats_report_path, None
            self.results.stats_report_reused = False
            if returncode == 0 and not launch_error and report_path is not None:
                self.results.open_report(str(report_path))
            else:
                self.activate_tab("reports-tab")
                self.results.clear_report(message_id('ui.0570'))
        elif kind == "html-report":
            self.results.html_report_finished(error=launch_error, cancelled=cancelled)
        elif kind == "compare":
            report_path, self.results.comparison_report_path = self.results.comparison_report_path, None
            # The comparison CLI uses exit 1 for a valid incompatible/unknown
            # report too; its reasons are the primary result to display.
            if comparison_ready:
                self.results.open_report(str(report_path))
            else:
                self.activate_tab("reports-tab")
                self.results.clear_report(message_id('ui.0486'))

    @on(Button.Pressed, "#stop-run")
    def stop_button(self) -> None:
        self.action_request_stop()

    def action_request_stop(self) -> None:
        if not self.operation_state().allows("stop"):
            self.notify("当前没有运行中的任务", severity="warning")
            return
        token = self.process_token
        html_report = self.process_kind == "html-report"
        self.push_screen(
            ConfirmActionScreen(
                message_id('ui.0984') if html_report else "终止当前任务？",
                message_id('ui.0985') if html_report else "将先向整个采集进程组发送 SIGINT，允许容器和监控器清理；"
                "超时后才会升级为 SIGTERM。已写入的 case 结果不会删除。",
                "终止任务",
                variant="error",
            ),
            lambda confirmed: self.confirmed_stop(confirmed) if token is self.process_token else None,
        )

    def confirmed_stop(self, confirmed: bool | None) -> None:
        if not confirmed or not self.operation_state().allows("stop"):
            return
        self.stop_requested = True
        self.preflight.close_preparation()
        self.stop_process_gracefully(self.process_token, self.lifecycle.process)

    @work(thread=True, group="stop", exclusive=True, exit_on_error=False)
    def stop_process_gracefully(self, token: object, process) -> None:
        self.process.safe_process_callback(self.process.deliver_process_callback, token, self.write_log,
                                    "[TUI] 正在请求采集进程安全停止……")
        if process is None:
            return  # The child may exit between confirmation and this worker.
        result = self.lifecycle.stop(expected_process=process)
        if not result.complete:
            self.process.safe_process_callback(self.process.deliver_process_callback, token, self.process_cleanup_incomplete, result)


    @on(Button.Pressed, "#quit-app")
    def action_request_quit(self) -> None:
        if self.images.operation:
            self.notify("镜像操作尚未完成，请稍候", severity="warning")
            return
        if not self.operation_state().allows("quit"):
            self.notify("任务仍在运行，请先使用 /stop 安全终止", severity="warning", timeout=6)
            return
        if self.results.summary_request is not None:
            self.results.summary_request.set()
        self.ui_closing = True
        self.exit()

    async def on_unmount(self) -> None:
        """Use the same bounded cleanup policy even after widgets are gone."""
        self.form_ready = False
        self.ui_closing = True
        if self.recovery.recovery_request is not None:
            self.recovery.recovery_request.set()
        self.recovery.recovery_after_check = None
        if self.results.summary_request is not None:
            self.results.summary_request.set()
        if self.results.report_request is not None:
            self.results.report_request.set()
        self.results.summary_request = self.results.report_request = self.preflight.check_request = None
        self.configuration.cancel_preview_timer()
        if self.elapsed_timer is not None:
            self.elapsed_timer.stop()
            self.elapsed_timer = None
        if self.images.refresh_timer is not None:
            self.images.refresh_timer.stop()
        result = await asyncio.to_thread(self.lifecycle.stop, closing=True)
        if not result.complete:
            print(f"[TUI] cleanup incomplete: PID={result.pid}; {result.error}", file=sys.stderr)

    def tr(self, source: str) -> str:
        return self.localization.tr(source)

    def notify(self, message: str, *, title: str = "", **kwargs) -> None:
        super().notify(self.tr(message), title=self.tr(title), **kwargs)

    def action_show_settings(self) -> None:
        self.configuration.show_settings()

    def action_toggle_log_view(self) -> None:
        self.configuration.toggle_log_view()

    def action_request_probe(self) -> None:
        self.experiments.request_probe()

    def action_request_run(self) -> None:
        self.experiments.request_run()

    def action_show_images(self) -> None:
        self.images.show_images()

    def action_quick_check(self) -> None:
        self.preflight.quick_check()

    def action_clear_log(self) -> None:
        self.results.clear_log()

    @on(Button.Pressed, '#model-candidates')
    def on_models_open_model_candidates(self) -> None:
        self.models.open_model_candidates()

    @on(Input.Changed, '#model')
    @on(Select.Changed, '#model-source')
    def on_models_candidate_identity_changed(self, event: TextualInput.Changed | Select.Changed) -> None:
        self.models.candidate_identity_changed(event)

    @on(Button.Pressed, '.experiment-picker')
    def on_catalog_select_experiment(self, event: Button.Pressed) -> None:
        self.catalog.select_experiment(event)

    @on(Button.Pressed, '#review-run-recovery')
    def on_recovery_review_failed_run(self) -> None:
        self.recovery.review_failed_run()

    @on(TabbedContent.TabActivated, '#main-tabs')
    def on_images_image_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        self.images.image_tab_activated(event)

    @on(Button.Pressed, '#image-view-tree, #image-view-list, #image-view-layers')
    def on_images_image_view_changed(self, event: Button.Pressed) -> None:
        self.images.image_view_changed(event)

    @on(DataTable.HeaderSelected, '#image-table')
    def on_images_sort_image_table(self, event: DataTable.HeaderSelected) -> None:
        self.images.sort_image_table(event)

    @on(Input.Changed, '#image-search')
    @on(Select.Changed, '#image-scope')
    def on_images_image_filter_changed(self) -> None:
        self.images.image_filter_changed()

    @on(DataTable.RowHighlighted, '#image-table')
    @on(DataTable.RowHighlighted, '#image-layer-table')
    @on(Tree.NodeHighlighted, '#image-tree')
    def on_images_show_image_detail(self) -> None:
        self.images.show_image_detail()

    @on(Collapsible.Expanded, '#image-dependencies')
    def on_images_expand_image_dependencies(self) -> None:
        self.images.expand_image_dependencies()

    @on(DataTable.RowSelected, '#image-table')
    @on(Tree.NodeSelected, '#image-tree')
    @on(Button.Pressed, '#image-toggle')
    def on_images_toggle_image_selection(self) -> None:
        self.images.toggle_image_selection()

    @on(Button.Pressed, '#image-clear')
    def on_images_clear_image_selection(self) -> None:
        self.images.clear_image_selection()

    @on(Button.Pressed, '#image-model')
    def on_images_select_model_images(self) -> None:
        self.images.select_model_images()

    @on(Button.Pressed, '#image-storage')
    def on_images_open_storage(self) -> None:
        self.images.open_storage()

    @on(Button.Pressed, '#image-delete')
    def on_images_request_delete_images(self) -> None:
        self.images.request_delete_images()

    @on(Select.Changed, '.ui-preference')
    @on(Checkbox.Changed, '.ui-preference')
    def on_configuration_ui_preference_changed(self) -> None:
        self.configuration.ui_preference_changed()

    @on(Button.Pressed, '#restore-ui-defaults')
    def on_configuration_restore_ui_defaults(self) -> None:
        self.configuration.restore_ui_defaults()

    @on(Button.Pressed, '#save-ui-settings')
    def on_configuration_save_ui_settings(self) -> None:
        self.configuration.save_ui_settings()

    @on(Button.Pressed, '#save-run-default')
    def on_configuration_save_run_default(self) -> None:
        self.configuration.save_run_default()

    @on(Button.Pressed, '#open-environment-settings')
    def on_configuration_open_environment_settings(self) -> None:
        self.configuration.open_environment_settings()

    @on(Button.Pressed, '#open-run-settings')
    def on_configuration_open_run_settings(self) -> None:
        self.configuration.open_run_settings()

    @on(Button.Pressed, '#copy-log')
    def on_configuration_copy_log_selection(self) -> None:
        self.configuration.copy_log_selection()

    @on(SelectableLog.Updated)
    def on_log_updated(self) -> None:
        self.logs.refresh_status()

    @on(Input.Changed, '#log-search')
    @on(Select.Changed, '#log-level')
    @on(Checkbox.Changed, '#log-details')
    def on_log_view_changed(self) -> None:
        self.logs.view_changed()

    @on(Button.Pressed, '#log-next')
    @on(Input.Submitted, '#log-search')
    def on_log_next(self) -> None:
        self.logs.find()

    @on(Button.Pressed, '#log-previous')
    def on_log_previous(self) -> None:
        self.logs.find(previous=True)

    @on(Button.Pressed, '#copy-log-all')
    def on_log_copy_all(self) -> None:
        self.logs.copy_all()

    @on(Button.Pressed, '#log-tools')
    def on_log_tools(self) -> None:
        self.logs.toggle_tools()

    @on(Button.Pressed, '#export-log')
    def on_log_export(self) -> None:
        self.logs.export()

    @on(Button.Pressed, '#follow-log')
    def on_configuration_follow_latest_log(self) -> None:
        self.configuration.follow_latest_log()

    @on(Button.Pressed, '#expand-log')
    @on(Button.Pressed, '#restore-log')
    def on_configuration_toggle_log_view_button(self) -> None:
        self.configuration.toggle_log_view_button()

    @on(Input.Changed, '.config-control')
    @on(Select.Changed, '.config-control')
    @on(Checkbox.Changed, '.config-control')
    def on_configuration_configuration_changed(self) -> None:
        self.configuration.configuration_changed()

    @on(Select.Changed, '#run-preset')
    def on_configuration_run_preset_changed(self, event: Select.Changed) -> None:
        self.configuration.run_preset_changed(event)

    @on(Button.Pressed, '#clear-log')
    def on_results_clear_log_button(self) -> None:
        self.results.clear_log_button()

    @on(Input.Changed, '#result-csv')
    def on_results_result_source_changed(self) -> None:
        self.results.result_source_changed()

    @on(Button.Pressed, '#summarize-results')
    def on_results_summarize_results_button(self) -> None:
        self.results.summarize_results_button()

    @on(Button.Pressed, '#plot-results')
    def on_results_plot_results_button(self) -> None:
        self.results.plot_results_button()

    @on(Input.Changed, '#report-source')
    def on_results_report_source_changed(self) -> None:
        self.results.report_source_changed()

    @on(Button.Pressed, '#report-open')
    def on_results_open_report_button(self) -> None:
        self.results.open_report_button()

    @on(DataTable.RowHighlighted, '#report-table')
    def on_results_report_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self.results.report_row_highlighted(event)

    @on(Button.Pressed, '#report-calculate')
    def on_results_calculate_report_button(self) -> None:
        self.results.calculate_report_button()

    @on(Button.Pressed, '#report-compare')
    def on_results_compare_experiments_button(self) -> None:
        self.results.compare_experiments_button()

    @on(Button.Pressed, '#html-report-generate')
    def on_results_generate_html(self) -> None:
        self.results.launch_html_report()

    @on(Button.Pressed, '#html-report-open')
    def on_results_open_html(self) -> None:
        self.results.open_html_report()

    @on(Button.Pressed, '#html-report-export')
    def on_results_export_html(self) -> None:
        self.results.export_html_report()

    @on(Button.Pressed, '#html-report-cancel')
    def on_results_cancel_html(self) -> None:
        if self.process_kind == "html-report":
            self.action_request_stop()

    @on(Button.Pressed, '#environment-status')
    def on_preflight_show_environment_status(self) -> None:
        self.preflight.show_environment_status()

    @on(Button.Pressed, '#open-model-store')
    def on_preflight_open_model_store(self) -> None:
        self.preflight.open_model_store()

    @on(Button.Pressed, '#profile-dry-run')
    def on_profiles_profile_dry_run_button(self) -> None:
        self.profiles.profile_dry_run_button()

    @on(Button.Pressed, '#profile-run')
    def on_profiles_profile_run_button(self) -> None:
        self.profiles.profile_run_button()

    @on(Input.Submitted, '#slash-command')
    def on_slash_slash_command_submitted(self, event: TextualInput.Submitted) -> None:
        self.slash.slash_command_submitted(event)

    @on(Button.Pressed, '#start-run')
    def on_experiments_start_run_button(self) -> None:
        self.experiments.start_run_button()

    @on(Button.Pressed, '#probe-largest')
    def on_experiments_probe_largest_button(self) -> None:
        self.experiments.probe_largest_button()
