"""AC-Prof TUI 事件处理与子进程生命周期。"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Literal

try:
    from rich.console import Console
    from textual import events, on, work
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.geometry import Size
    from textual.theme import Theme
    from textual.widget import Widget
    from textual.widgets import (
        Button,
        Header,
        Select,
        Static,
        TabbedContent,
        Tabs,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - exercised before tests install deps
    if exc.name == "textual":
        raise SystemExit(
            "AC-Prof TUI 需要 Textual。请运行：\n"
            "  .venv/bin/python -m pip install --require-hashes -r requirements/host.lock\n"
            "发行安装请重新运行项目的 setup.sh。"
        ) from None
    raise

from acprof.experiment import RunConfig
from acprof.host.image_management import (
    ImageInventory,
    ImageLayer,
    ManagedImage,
)
from acprof.messages import join_messages, message
from acprof.platform import detect_environment
from acprof.tui import run_form
from acprof.tui.catalog_actions import CatalogActions
from acprof.tui.commands import (
    OperationState,
    PendingLaunch,
    format_command,
)
from acprof.tui.configuration_actions import ConfigurationActions
from acprof.tui.diagnostics import (
    PreflightCheck,
    quick_preflight,
    summarize_result_csv,
)
from acprof.tui.experiment_actions import ExperimentActions
from acprof.tui.image_actions import ImageActions
from acprof.tui.input import BarCursorApp, BarCursorInput as Input
from acprof.tui.localization_actions import LocalizationActions
from acprof.tui.log import SelectableLog
from acprof.tui.model_actions import ModelActions
from acprof.tui.preflight_actions import PreflightActions
from acprof.tui.presentation import NOT_APPLICABLE, UNKNOWN
from acprof.tui.process import ProcessLifecycle, StopResult
from acprof.tui.profile_actions import ProfileActions
from acprof.tui.progress import ProgressSnapshot, RunProgressTracker
from acprof.tui.recovery_actions import RecoveryActions
from acprof.tui.rendering import CjkScreen
from acprof.tui.reports import ReportView, read_report
from acprof.tui.result_actions import ResultActions
from acprof.tui.run_results import RunArtifacts, RunResult, inspect_run_result
from acprof.tui.settings import (
    default_settings_path,
    load_settings,
    save_settings,
)
from acprof.tui.slash_actions import SlashCommandActions
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


class AcprofTui(
    ModelActions,
    CatalogActions,
    RecoveryActions,
    ImageActions,
    LocalizationActions,
    ConfigurationActions,
    ResultActions,
    PreflightActions,
    ProfileActions,
    SlashCommandActions,
    ExperimentActions,
    BarCursorApp,
):
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
    def _project_dir(self) -> Path:
        return PROJECT_DIR

    @property
    def _python_executable(self) -> Path:
        return PYTHON_EXECUTABLE

    @staticmethod
    def _summarize_result_csv(*args, **kwargs):
        return summarize_result_csv(*args, **kwargs)

    @staticmethod
    def _quick_preflight(*args, **kwargs):
        return quick_preflight(*args, **kwargs)

    @staticmethod
    def _read_report(*args, **kwargs):
        return read_report(*args, **kwargs)

    @staticmethod
    def _save_local_settings(*args, **kwargs):
        return save_settings(*args, **kwargs)

    @staticmethod
    def _current_environment():
        return detect_environment()

    def __init__(
        self, initial_config: RunConfig | None = None, *, settings_path: Path | None = None,
        color_system: Literal["truecolor", "256", "auto"] = "truecolor",
    ):
        super().__init__(ansi_color=False)
        if color_system != "auto":
            # SSH often omits COLORTERM even when the client supports RGB.
            # Configure only this renderer; keep the subprocess environment intact.
            self.console = Console(
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
        self.settings_path = settings_path or default_settings_path(PROJECT_DIR)
        self._saved_settings, self._settings_warning = load_settings(
            self.settings_path, PROJECT_DIR,
        )
        self.ui_preferences = self._saved_settings.ui
        self._localized_text: dict[tuple[Widget, str], str] = {}
        self._localized_selects: dict[Select, tuple] = {}
        self._applied_language: str | None = None
        config = self._saved_settings.run_defaults or RunConfig.smoke()
        if self._saved_settings.last_model:
            config = replace(config, model=self._saved_settings.last_model,
                             revision=config.revision if config.model == self._saved_settings.last_model else "")
        self.initial_config = initial_config if initial_config is not None else config
        self._extra_run_options = dict(self.initial_config.extra_options)
        self._planned_input: dict | None = None
        self._planned_input_identity: tuple | None = None
        for palette in THEME_CATALOG:
            self.register_theme(Theme(**palette.theme_kwargs()))
        self.theme = self.ui_preferences.theme
        self._lifecycle = ProcessLifecycle()
        self._process_kind = ""
        self._pending_launch: PendingLaunch | None = None
        self._active_run_config: RunConfig | None = None
        self._active_command: tuple[str, ...] = ()
        self._started_monotonic = 0.0
        self._stop_requested = False
        self._latest_snapshot = ProgressSnapshot()
        self._check_running = False
        self._ui_closing = False
        self._check_request = None
        self._check_completed = False
        self._check_config: RunConfig | None = None
        self._checked_config: RunConfig | None = None
        self._preflight_checks: tuple[PreflightCheck, ...] = ()
        self._preflight_error = ""
        self._summary_request: Event | None = None
        self._recovery_request: Event | None = None
        self._recovery_after_check = None
        self._summary_path: Path | None = None
        self._report_request: Event | None = None
        self._report_path: Path | None = None
        self._read_jobs: set[object] = set()
        self._process_token = None
        self._run_result = RunResult()
        self._environment_open = False
        self._preparation_screen = None
        self._preparation_cancelled = False
        self._pending_source_change = None
        self._preparation_request: tuple[subprocess.Popen[str], int] | None = None
        self._form_ready = False
        self._config_issues = ()
        self._field_errors_visible = False
        self._applying_config = False
        self._preview_timer = None
        self._selected_preset: str | None = None
        self._initial_preset = run_form.infer_preset(self.initial_config)
        self._elapsed_timer = None
        self._report_view: ReportView | None = None
        self._stats_report_path: Path | None = None
        self._comparison_report_path: Path | None = None
        self._stats_report_reused = False
        self._image_inventory: ImageInventory | None = None
        self._image_operation = ""
        self._image_refresh_timer = None
        self._image_refresh_error = ""
        self._storage_screen = None
        self._storage_loading = False
        self._selected_image_ids: set[str] = set()
        self._visible_images: tuple[ManagedImage, ...] = ()
        self._visible_image_layers: tuple[ImageLayer, ...] = ()
        self._image_view = "tree"
        self._image_sort = ("name", False)
        self._focused_image_id = ""

    def get_default_screen(self) -> CjkScreen:
        return CjkScreen(id="_default")

    def compose(self) -> ComposeResult:
        # A ticking clock would force periodic redraws during RAPL windows.
        with Header(show_clock=False):
            yield self._localized_widget(Button(
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
                yield self._localized_widget(Input(
                    placeholder="快捷命令：输入 /help 查看可用命令，按 Enter 执行",
                    id="slash-command",
                ))

    def on_mount(self) -> None:
        self._configure_interaction()
        self._configure_scrollbars()
        self._capture_language_text()
        self._apply_ui_preferences()
        self._set_text(self.query_one('#settings-location', Static), message('保存位置：{0}', self.settings_path))
        self._update_responsive_layout()
        self._selected_preset = self._select("run-preset")
        self._form_ready = True
        self._refresh_command_preview(notify=False)
        self.call_after_refresh(self.action_quick_check)
        self.query_one("#model", Input).focus()
        self._image_refresh_timer = self.set_interval(
            self.IMAGE_REFRESH_INTERVAL, self.refresh_images, pause=True,
        )
        if self._settings_warning:
            self.notify(self._settings_warning, title="设置读取提示", severity="warning", timeout=8)

    def on_resize(self, event: events.Resize) -> None:
        # App.size can still refer to the previous frame while Resize is
        # dispatched. Use the event's new dimensions for responsive classes.
        self._update_responsive_layout(event.size)
        if (self._form_ready and self._image_inventory is not None and not self._images_unavailable(allow_refresh=True)
                and self.query_one("#main-tabs", TabbedContent).active == "images-tab"):
            self._render_images(width=event.size.width)

    def _update_responsive_layout(self, size: Size | None = None) -> None:
        size = self.size if size is None else size
        self.set_class(size.width < 110, "narrow")
        self.set_class(size.height < 35, "short")



    def _is_busy(self) -> bool:
        return self._operation_state().busy

    def _operation_state(self) -> OperationState:
        process = self._lifecycle.process
        return OperationState(
            process=process is not None or bool(self._process_kind),
            stoppable=process is not None and process.poll() is None,
            checking=self._check_running, reading=bool(self._read_jobs),
            maintenance=bool(self._image_operation) or self._storage_loading,
            configuring=self._environment_open or getattr(self, "_picker_open", False),
            measuring=self._latest_snapshot.measurement_active, closing=self._ui_closing,
        )

    def _allow_operation(self, operation: str) -> bool:
        if self._operation_state().allows(operation):
            return True
        self.notify("请等待当前任务完成", severity="warning")
        return False

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        operation = {"request_run": "run", "request_probe": "probe",
                     "quick_check": "check", "request_stop": "stop"}.get(action)
        if action == "request_run" and self._preflight_run_reason():
            return False
        return self._operation_state().allows(operation) if operation else True

    def _set_busy(self, busy: bool) -> None:
        if not self.is_running or not self._form_ready or self._ui_closing:
            return
        state = self._operation_state()
        busy = busy or state.busy
        # Configuration changes during a run can queue preview redraws and
        # make the visible settings differ from the running subprocess.
        if busy:
            self._cancel_preview_timer()
        for table in self.query(ResizableDataTable):
            refreshing_image = self._image_operation == "refresh" and table.has_class("image-control")
            table.resize_enabled = not ((busy and not refreshing_image) or self._latest_snapshot.measurement_active)
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
            widget.disabled = (busy and self._image_operation != "refresh") or self._latest_snapshot.measurement_active
        for selector, operation in {
            "#start-run": "run", "#probe-largest": "probe",
            "#review-run-recovery": "run",
            "#summarize-results": "summary", "#plot-results": "plot",
            "#profile-dry-run": "profile", "#profile-run": "profile", "#report-open": "report",
            "#report-calculate": "stats", "#stop-run": "stop",
        }.items():
            self.query_one(selector, Button).disabled = not state.allows(operation)
        for selector in ("#result-csv", "#report-source"):
            self.query_one(selector, Input).disabled = not state.allows("summary")
        self.refresh_bindings()
        if not busy:
            self.set_input_cursor_blink_enabled(True)
        self._update_image_controls()
        self._sync_image_refresh_timer()
        self._refresh_preflight_state()

    def _activate_tab(self, tab_id: str) -> None:
        if self.screen.maximized is not None:
            self.screen.minimize()
        tabs = self.query_one("#main-tabs", TabbedContent)
        # Move focus before hiding the outgoing pane. Widget.focus() defers
        # this change, allowing an old pane's focus event to undo the switch.
        self.screen.set_focus(tabs.query_one(Tabs), scroll_visible=False)
        tabs.active = tab_id


    def _launch(self, pending: PendingLaunch) -> None:
        if pending.kind == "run" and not self._allow_collection():
            return
        if not self._allow_operation("run"):
            return
        model = pending.config.model if pending.config is not None else ""
        # Intended output is not evidence of a result. Keep the selected/history
        # paths until a matching run attempt has actually published artifacts.
        if pending.kind == "run":
            self.query_one("#run-recovery-actions").display = False
            self._planned_input = None
            self._planned_input_identity = None
            self._remember_last_used(model=model)
            self._run_result = RunResult()
            self._summary_request = None
            self._set_text(self.query_one("#result-summary", Static),
                           "本次尚未产生结果；历史结果可通过原路径查看。")
        else:
            self._remember_last_used(model=model, result_dir=pending.result_dir, result_csv=pending.result_csv)
        self._process_token = object()
        self._active_run_config = pending.config if pending.kind == "run" else None
        self._comparison_report_path = Path(pending.report_path) if pending.kind == "compare" else None
        self._active_command = pending.command
        self._process_kind = pending.kind
        self._started_monotonic = time.monotonic()
        self._stop_requested = False
        self._preparation_cancelled = False
        self._latest_snapshot = ProgressSnapshot(stage="启动中", detail="正在创建子进程")
        self._set_busy(True)
        self._activate_tab("monitor-tab")
        # Moving focus away from inputs stops the native cursor's blink timer.
        self.query_one("#stop-run", Button).focus()
        # Start the elapsed-time ticker; it self-gates on measurement_active
        # to avoid any redraws during formal energy/latency windows.
        if self._elapsed_timer is not None:
            self._elapsed_timer.stop()
        self._elapsed_timer = self.set_interval(1.0, self._tick_elapsed)
        log = self.query_one("#run-log", SelectableLog)
        log.write(f"$ {format_command(pending.command)}")
        log.write(self.tr("[TUI] 子进程输出通过管道读取；tmux pane 捕获已对该子进程禁用。"))
        self._render_snapshot(self._latest_snapshot)
        self._execute_command(list(pending.command), pending.kind)

    def _tick_elapsed(self) -> None:
        """Update elapsed time display; skipped during measurement windows."""
        # Textual stops the app before removing widgets, ahead of on_unmount.
        if not self.is_running:
            return
        if self._latest_snapshot.measurement_active:
            return  # Zero redraws during RAPL/latency measurement windows.
        if not self._started_monotonic or not self._is_busy():
            return
        elapsed = self._format_elapsed(time.monotonic() - self._started_monotonic)
        self._set_text(self.query_one('#status-elapsed', Static), elapsed)

    _STAGE_CSS_CLASS = {
        "等待": "stage-idle",
        "服务就绪": "stage-success",
        "已完成": "stage-success",
        "探测完成": "stage-success",
        "找到最低可用内存": "stage-success",
        "case 完成": "stage-success",
        "失败": "stage-error",
        "清理未完成": "stage-error",
        "探测失败": "stage-error",
        "任务不支持": "stage-error",
        "已终止": "stage-error",
        "已停止": "stage-error",
        "部分完成": "stage-running",
        "正式测量": "stage-measuring",
        "最大尺度探测": "stage-measuring",
    }
    _STAGE_CLASSES = frozenset({
        "stage-idle", "stage-running", "stage-measuring",
        "stage-success", "stage-error",
    })

    @work(thread=True, group="process", exclusive=True, exit_on_error=False)
    def _execute_command(self, command: list[str], kind: str) -> None:
        token = self._process_token
        def deliver(callback, *args):
            self.call_from_thread(self._deliver_process_callback, token, callback, *args)
        run_before = None
        if kind == "run" and self._active_run_config is not None:
            try:
                run_before = RunArtifacts.read(self._active_run_config.result_dir(PROJECT_DIR))
            except (OSError, ValueError, RuntimeError):
                pass  # CLI reports invalid history; it cannot become current results.
        from acprof.preparation_events import PREFIX, parse_event as parse_preparation_event
        tracker = RunProgressTracker(structured=(kind == "run")) if kind in {"run", "probe"} else None
        suppressed_lines = 0
        deferred_important_lines: list[str] = []
        process: subprocess.Popen[str] | None = None
        launch_error = ""
        returncode = 1
        diagnostic_continuation = False
        try:
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["MPLBACKEND"] = "Agg"
            child_env["ACPROF_TUI"] = "1"
            child_env.pop("ACPROF_INTERACTIVE_PREPARATION", None)
            if kind == "run":
                child_env["ACPROF_INTERACTIVE_PREPARATION"] = "1"
            # acprof run otherwise captures the entire full-screen pane, including
            # ANSI redraws, when the TUI itself is launched inside tmux.
            child_env.pop("TMUX", None)
            child_env.pop("TMUX_PANE", None)
            process = self._lifecycle.start(command, cwd=PROJECT_DIR, env=child_env)
            if self._stop_requested:
                self._lifecycle.stop()
            deliver(self._process_started, process.pid, kind)
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip("\r\n")
                before = tracker.snapshot if tracker is not None else None
                snapshot = tracker.feed(line) if tracker is not None else None
                state_changed = snapshot != before if snapshot is not None else False
                important = (
                    "[ERROR]" in line
                    or "[WARN]" in line
                    or line.startswith("Traceback")
                )
                # perf and tracebacks emit untagged continuation lines. Retain
                # the complete diagnostic until the next structured log entry.
                if line.startswith("["):
                    diagnostic_continuation = important
                elif important:
                    diagnostic_continuation = True
                important = important or diagnostic_continuation
                if (
                    before is not None
                    and before.measurement_active
                    and snapshot is not None
                    and snapshot.measurement_active
                ):
                    if important:
                        deferred_important_lines.append(line)
                    else:
                        suppressed_lines += 1
                    continue
                if (
                    before is not None
                    and before.measurement_active
                    and snapshot is not None
                    and not snapshot.measurement_active
                ):
                    if deferred_important_lines:
                        deliver(
                            self._show_deferred_lines,
                            tuple(deferred_important_lines),
                        )
                        deferred_important_lines.clear()
                    if suppressed_lines:
                        deliver(
                            self._show_suppressed_count,
                            suppressed_lines,
                        )
                        suppressed_lines = 0
                deliver(
                    self._consume_process_line,
                    "" if line.startswith(("ACPROF_EVENT ", PREFIX)) else line,
                    snapshot,
                    state_changed,
                )
                preparation = parse_preparation_event(line)
                if line.startswith("[network-preflight] "):
                    deliver(self._network_preflight_report, line)
                elif line.startswith("[network-download] "):
                    deliver(self._network_download_report, line)
                if preparation is not None:
                    deliver(self._preparation_event, preparation)
            returncode = process.wait()
            process.stdout.close()
        except Exception as exc:  # process errors must become visible in the UI
            launch_error = f"{type(exc).__name__}: {exc}"
            if process is not None:
                result = self._lifecycle.stop()
                if not result.complete:
                    self._safe_process_callback(self._process_cleanup_incomplete, result)
        finally:
            final_snapshot = tracker.snapshot if tracker is not None else None
            if process is not None and process.poll() is None:
                # Keep both ownership and the output reader until exit. A failed
                # UI callback must not orphan a collector or block its pipe.
                self._watch_failed_process(process, kind, final_snapshot, launch_error, run_before, token)
            else:
                if process is not None:
                    if process.stdout is not None:
                        process.stdout.close()
                    self._lifecycle.release(process)
                    returncode = process.returncode
                if kind == "run" and process is not None and run_before is not None:
                    self._inspect_finished_run(run_before, process.pid, token)
                if suppressed_lines:
                    self._safe_process_callback(self._show_suppressed_count, suppressed_lines)
                if deferred_important_lines:
                    self._safe_process_callback(self._show_deferred_lines, tuple(deferred_important_lines))
                self._safe_process_callback(
                    self._process_finished, kind, returncode, final_snapshot, launch_error,
                )

    def _deliver_process_callback(self, token, callback, *args) -> None:
        if self.is_running and self._form_ready and not self._ui_closing and token is self._process_token:
            callback(*args)

    def _accept_run_result(self, token, result: RunResult) -> None:
        if token is self._process_token:
            self._run_result = result

    def _inspect_finished_run(self, before: RunArtifacts, pid: int, token: object) -> None:
        self._safe_process_callback(self._deliver_process_callback, token, self._set_busy, True)
        try:
            result = inspect_run_result(before, pid)
        except Exception as exc:
            result = RunResult(detail=f"{type(exc).__name__}: {exc}")
        self._safe_process_callback(self._accept_run_result, token, result)

    def _safe_process_callback(self, callback, *args) -> None:
        if self._ui_closing or not self.is_running:
            return
        try:
            self.call_from_thread(self._deliver_ui_callback, callback, *args)
        except Exception as exc:
            # Widgets may already have been pruned during shutdown. Process
            # cleanup and ownership must not depend on their availability.
            print(f"[TUI] callback failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    def _deliver_ui_callback(self, callback, *args) -> None:
        if self.is_running and self._form_ready and not self._ui_closing:
            callback(*args)

    @work(thread=True, group="process-reap", exit_on_error=False)
    def _watch_failed_process(self, process, kind, snapshot, error, before, token) -> None:
        try:
            if process.stdout is not None:
                for _ in process.stdout:
                    pass
        except (OSError, ValueError):
            pass
        process.wait()
        if process.stdout is not None:
            process.stdout.close()
        if self._lifecycle.release(process):
            if kind == "run" and before is not None:
                self._inspect_finished_run(before, process.pid, token)
            self._safe_process_callback(
                self._process_finished, kind, process.returncode, snapshot, error,
            )

    def _process_cleanup_incomplete(self, result: StopResult) -> None:
        detail = message('PID {0} 仍未确认退出：{1}；可再次停止。', result.pid, result.error)
        self._latest_snapshot = replace(self._latest_snapshot, stage="清理未完成", detail=detail)
        self._render_snapshot(self._latest_snapshot)
        self._write_log(detail)
        self._set_busy(True)
        self.notify(detail, severity="error", timeout=10)

    def _process_started(self, pid: int, kind: str) -> None:
        self._set_busy(True)
        self.query_one("#run-log", SelectableLog).write(
            self.tr(message('[TUI] {0} 进程已启动，PID={1}', kind, pid))
        )

    def _show_suppressed_count(self, count: int) -> None:
        self.query_one("#run-log", SelectableLog).write(
            self.tr(message('[TUI] 为降低测量干扰，本窗口隐藏了 {0} 行常规输出。', count))
        )

    def _show_deferred_lines(self, lines: tuple[str, ...]) -> None:
        log = self.query_one("#run-log", SelectableLog)
        log.write(self.tr("[TUI] 测量窗口结束，显示期间延迟刷新的重要消息："))
        for line in lines:
            log.write(line)

    def _write_log(self, line: str) -> None:
        self.query_one("#run-log", SelectableLog).write(self.tr(line))

    def _consume_process_line(
        self,
        line: str,
        snapshot: ProgressSnapshot | None,
        state_changed: bool,
    ) -> None:
        if self._process_kind == "stats" and line.startswith("ACPROF_STATS "):
            receipt = json.loads(line.removeprefix("ACPROF_STATS "))
            if (not isinstance(receipt, dict) or not isinstance(receipt.get("report_path"), str)
                    or not receipt["report_path"] or type(receipt.get("reused")) is not bool):
                raise ValueError("统计进程返回的报告信息无效")
            self._stats_report_path = Path(receipt["report_path"])
            self._stats_report_reused = receipt["reused"]
            line = self.tr(message(
                "已有相同报告：{0}" if self._stats_report_reused else "统计报告已保存：{0}",
                self._stats_report_path,
            ))
        if line:
            self.query_one("#run-log", SelectableLog).write(line)
        if snapshot is not None:
            if self._process_kind == "run" and snapshot.stage == "已完成":
                snapshot = replace(snapshot, stage="核验产物", detail="进程已报告完成，等待本次产物核验。")
            was_measuring = self._latest_snapshot.measurement_active
            self._latest_snapshot = snapshot
            self.set_input_cursor_blink_enabled(not snapshot.measurement_active)
            self._sync_image_refresh_timer()
            if self._elapsed_timer is not None:
                if snapshot.measurement_active:
                    self._elapsed_timer.pause()
                elif was_measuring:
                    self._elapsed_timer.resume()
            if state_changed:
                self._render_snapshot(snapshot)

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _render_snapshot(self, snapshot: ProgressSnapshot) -> None:
        from acprof.tui.preparation import phase_summary
        self._set_text(self.query_one("#status-preparation", Static), phase_summary(snapshot))
        elapsed = (
            self._format_elapsed(time.monotonic() - self._started_monotonic)
            if self._started_monotonic
            else NOT_APPLICABLE
        )
        # Stage text with visual category coloring.
        stage_widget = self.query_one("#status-stage", Static)
        self._set_text(stage_widget, snapshot.stage)
        stage_widget.set_classes(
            self._STAGE_CSS_CLASS.get(snapshot.stage, "stage-running")
        )
        self._set_text(self.query_one('#status-elapsed', Static), elapsed)
        self._set_text(self.query_one("#status-case", Static), message(
            "当前 {0} · 已完成 {1}/{2}", snapshot.current_case or NOT_APPLICABLE,
            snapshot.completed_cases, snapshot.total_cases,
        ))
        missing = UNKNOWN if snapshot.current_case else NOT_APPLICABLE
        cpu = snapshot.cpu if snapshot.cpu != "-" else missing
        mem = f"{snapshot.mem}GB" if snapshot.mem != "-" else missing
        gpu = snapshot.gpu if snapshot.gpu != "-" else missing
        self._set_text(self.query_one("#status-resource", Static), message("CPU={0}  MEM={1}  GPU={2}", cpu, mem, gpu))
        self._set_text(self.query_one('#status-errors', Static), f'{snapshot.warnings} / {snapshot.errors}')
        self._set_text(self.query_one('#status-detail', Static), snapshot.detail)

    def _process_finished(
        self,
        kind: str,
        returncode: int,
        snapshot: ProgressSnapshot | None,
        launch_error: str,
    ) -> None:
        if kind == "stats" and returncode == 0 and not launch_error and self._stats_report_path is None:
            launch_error = "统计进程未返回报告路径"
        comparison_ready = (kind == "compare" and returncode in {0, 1} and not launch_error
                            and not self._stop_requested and self._comparison_report_path is not None
                            and self._comparison_report_path.is_file())
        if kind == "compare" and returncode == 0 and not launch_error and not self._stop_requested and not comparison_ready:
            launch_error = "比较进程未返回报告路径"
        self._close_preparation()
        if snapshot is not None and snapshot.measurement_status == "running":
            snapshot = replace(snapshot, measurement_status=(
                "cancelled" if self._stop_requested else "failed" if returncode else "passed"))
        # Stop the elapsed-time ticker.
        if self._elapsed_timer is not None:
            self._elapsed_timer.stop()
            self._elapsed_timer = None
        # Show the final elapsed time.
        if self._started_monotonic:
            final_elapsed = self._format_elapsed(
                time.monotonic() - self._started_monotonic
            )
            self._set_text(self.query_one('#status-elapsed', Static), final_elapsed)
        if kind in {"stats", "compare"}:
            # Move focus before disabling the monitor page's focused Stop button;
            # its queued focus event could otherwise reactivate that old page.
            self._activate_tab("reports-tab")
        log = self.query_one("#run-log", SelectableLog)
        unsupported_task = (
            snapshot is not None and snapshot.stage == "任务不支持"
            and returncode != 0 and not self._stop_requested and not launch_error
        )
        if launch_error:
            log.write(self.tr(message('[TUI][ERROR] 无法运行命令：{0}', launch_error)))
            self.notify(launch_error, title="任务启动失败", severity="error", timeout=8)
        elif comparison_ready:
            notice = message("比较报告已生成；请查看可比性、质量和证据不足的原因。")
            log.write(self.tr(notice))
            self.notify(notice, severity="information", timeout=8)
        elif returncode == 0 and kind != "run" and not self._stop_requested:
            log.write(self.tr(message('[TUI] {0} 任务完成，退出码 0', kind)))
            notice = (message("已有相同报告：{0}", self._stats_report_path)
                      if kind == "stats" and self._stats_report_reused else "任务已完成")
            self.notify(notice, severity="information", timeout=8 if self._stats_report_reused else 5)
        elif self._stop_requested:
            log.write(self.tr(message('[TUI] 任务已由用户终止，退出码 {0}', returncode)))
            self.notify("任务已终止；部分 case 结果可能仍可续跑", severity="warning", timeout=7)
        elif unsupported_task:
            log.write(self.tr(snapshot.detail))
            log.write(self.tr(message(
                '处理办法：换用已适配任务的模型；识别有误时修正配置；需要该任务时等待或开发适配。'
            )))
            self.notify(snapshot.detail, title="任务不支持", severity="error", timeout=12)
        elif kind != "run":
            log.write(self.tr(message('[TUI][ERROR] {0} 任务失败，退出码 {1}', kind, returncode)))
            self.notify(message('任务失败，退出码 {0}', returncode), severity="error", timeout=8)

        current_csv = ""
        if unsupported_task:
            self._latest_snapshot = replace(snapshot, measurement_active=False, measurement_status="failed")
            self._render_snapshot(self._latest_snapshot)
        elif kind == "run":
            result = self._run_result
            stage = result.stage(returncode, self._stop_requested, launch_error)
            final_state = snapshot or self._latest_snapshot
            detail = message("已完成 {0}/{1} 个资源 case；{2}", result.completed_cases,
                             result.total_cases or final_state.total_cases,
                             message("可查看结果，或检查恢复与重试选项") if stage != "已完成" else message("可查看结果与计算统计"))
            self.query_one("#run-recovery-actions").display = stage != "已完成"
            if result.detail:
                detail = join_messages("\n", (detail, result.detail))
            if result.new_cases and result.retained_dir:
                detail = join_messages("\n", (detail, message("已保留的实验目录：{0}", result.retained_dir)))
            if result.complete and not result.result_csv and not result.new_cases:
                detail = join_messages("\n", (detail, message("已有实验已完成；本次未重新采集。")))
            self._latest_snapshot = replace(final_state, stage=stage, detail=detail,
                completed_cases=result.completed_cases, total_cases=result.total_cases or final_state.total_cases,
                measurement_active=False, measurement_status=("passed" if stage == "已完成" else
                    "cancelled" if self._stop_requested else "failed"))
            self._render_snapshot(self._latest_snapshot)
            current_csv = result.result_csv
            if current_csv:
                self._remember_last_used(result_csv=current_csv, result_dir=str(Path(current_csv).parent))
            else:
                summary_text = (message("无法确认本次结果：{0}；历史结果可通过原路径查看。", result.detail)
                                if result.detail and not result.belongs_to_attempt else
                                message("本次未产生结果 CSV；已保留 {0} 个本次完成的 case。历史结果可通过原路径查看。", result.new_cases))
                self._set_text(self.query_one("#result-summary", Static), summary_text)
            self.notify(message("实验{0} · {1}", message(stage), detail),
                        severity="information" if stage == "已完成" else "warning" if stage in {"部分完成", "已停止"} else "error")
        elif kind == "probe":
            final_state = snapshot or self._latest_snapshot
            if launch_error or (returncode != 0 and not self._stop_requested):
                detail = launch_error or (
                    final_state.detail
                    if final_state.stage == "探测失败"
                    else message('探测进程退出码 {0}', returncode)
                )
                final_state = replace(
                    final_state,
                    stage="探测失败",
                    detail=detail,
                    measurement_active=False,
                )
            elif self._stop_requested:
                final_state = replace(
                    final_state,
                    stage="已终止",
                    detail="用户终止了最大尺度探测",
                    measurement_active=False,
                )
            elif final_state.stage != "探测完成":
                final_state = replace(
                    final_state,
                    stage="探测完成",
                    detail="最大尺度单次请求已完成",
                    measurement_active=False,
                )
            self._latest_snapshot = final_state
            self._render_snapshot(final_state)
        else:
            if launch_error or (returncode != 0 and not self._stop_requested and not comparison_ready):
                stage = "失败"
                detail = launch_error or message('{0} 进程退出码 {1}', kind, returncode)
            elif self._stop_requested:
                stage = "已终止"
                detail = message('用户终止了 {0} 任务', kind)
            else:
                stage = "已完成"
                detail = message('{0} 任务已完成', kind)
            self._latest_snapshot = ProgressSnapshot(stage=stage, detail=detail)
            self._render_snapshot(self._latest_snapshot)

        self._active_command = ()
        self._process_kind = ""
        self._stop_requested = False
        self._set_busy(self._is_busy())
        if self._preparation_cancelled and not self._is_busy():
            self._activate_tab("run-tab")
            self._preparation_cancelled = False
            if self._pending_source_change is not None:
                from uuid import uuid4
                change, self._pending_source_change = self._pending_source_change, None
                config = self._active_run_config or self.initial_config
                self._apply_config(replace(config, model=change["model_id"], revision=change["revision"],
                    model_source="modelscope", download_mode="auto", task="", task_family="", backend="",
                    model_spec="", resume=False, skip_build=False,
                    output_dir=str(Path(config.output_dir) / ("modelscope-" + uuid4().hex[:8]))))
        if current_csv:
            self._update_result_summary(current_csv, notify=False)
        self._sync_image_refresh_timer()
        if kind == "stats":
            report_path, self._stats_report_path = self._stats_report_path, None
            self._stats_report_reused = False
            if returncode == 0 and not launch_error and report_path is not None:
                self._open_report(str(report_path))
            else:
                self._activate_tab("reports-tab")
                self._clear_report(message("统计未完成，请查看“运行监控”中的错误或终止日志。"))
        elif kind == "compare":
            report_path, self._comparison_report_path = self._comparison_report_path, None
            # The comparison CLI uses exit 1 for a valid incompatible/unknown
            # report too; its reasons are the primary result to display.
            if comparison_ready:
                self._open_report(str(report_path))
            else:
                self._activate_tab("reports-tab")
                self._clear_report(message("比较未完成，请查看“运行监控”中的错误或终止日志。"))

    @on(Button.Pressed, "#stop-run")
    def stop_button(self) -> None:
        self.action_request_stop()

    def action_request_stop(self) -> None:
        if not self._operation_state().allows("stop"):
            self.notify("当前没有运行中的任务", severity="warning")
            return
        token = self._process_token
        self.push_screen(
            ConfirmActionScreen(
                "终止当前任务？",
                "将先向整个采集进程组发送 SIGINT，允许容器和监控器清理；"
                "超时后才会升级为 SIGTERM。已写入的 case 结果不会删除。",
                "终止任务",
                variant="error",
            ),
            lambda confirmed: self._confirmed_stop(confirmed) if token is self._process_token else None,
        )

    def _confirmed_stop(self, confirmed: bool | None) -> None:
        if not confirmed or not self._operation_state().allows("stop"):
            return
        self._stop_requested = True
        self._close_preparation()
        self._stop_process_gracefully(self._process_token, self._lifecycle.process)

    @work(thread=True, group="stop", exclusive=True, exit_on_error=False)
    def _stop_process_gracefully(self, token: object, process) -> None:
        self._safe_process_callback(self._deliver_process_callback, token, self._write_log,
                                    "[TUI] 正在请求采集进程安全停止……")
        if process is None:
            return  # The child may exit between confirmation and this worker.
        result = self._lifecycle.stop(expected_process=process)
        if not result.complete:
            self._safe_process_callback(self._deliver_process_callback, token, self._process_cleanup_incomplete, result)




    @on(Button.Pressed, "#quit-app")
    def action_request_quit(self) -> None:
        if self._image_operation:
            self.notify("镜像操作尚未完成，请稍候", severity="warning")
            return
        if not self._operation_state().allows("quit"):
            self.notify("任务仍在运行，请先使用 /stop 安全终止", severity="warning", timeout=6)
            return
        if self._summary_request is not None:
            self._summary_request.set()
        self._ui_closing = True
        self.exit()

    async def on_unmount(self) -> None:
        """Use the same bounded cleanup policy even after widgets are gone."""
        self._form_ready = False
        self._ui_closing = True
        if self._recovery_request is not None:
            self._recovery_request.set()
        self._recovery_after_check = None
        if self._summary_request is not None:
            self._summary_request.set()
        if self._report_request is not None:
            self._report_request.set()
        self._summary_request = self._report_request = self._check_request = None
        self._cancel_preview_timer()
        if self._elapsed_timer is not None:
            self._elapsed_timer.stop()
            self._elapsed_timer = None
        if self._image_refresh_timer is not None:
            self._image_refresh_timer.stop()
        result = await asyncio.to_thread(self._lifecycle.stop, closing=True)
        if not result.complete:
            print(f"[TUI] cleanup incomplete: PID={result.pid}; {result.error}", file=sys.stderr)
