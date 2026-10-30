"""AC-Prof TUI 事件处理与子进程生命周期。"""

from __future__ import annotations

import asyncio
import sys
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

from acprof.experiments.configuration import RunConfig
from acprof.tui.commands import (
    OperationState,
    PendingLaunch,
)
from acprof.tui.experiments.controller import CatalogController
from acprof.tui.images.controller import ImageController
from acprof.tui.logging.controller import LogController
from acprof.tui.logging.widget import SelectableLog
from acprof.tui.models.controller import ModelController
from acprof.tui.results.controller import ResultController
from acprof.tui.results.profile_controller import ProfileController
from acprof.tui.results.views import compose_results_tab
from acprof.tui.run.execution import ExecutionCoordinator
from acprof.tui.run.experiment_controller import ExperimentController
from acprof.tui.run.preflight_controller import PreflightController
from acprof.tui.run.presenter import RunPresenter
from acprof.tui.run.process import StopResult
from acprof.tui.run.process_controller import ProcessController
from acprof.tui.run.progress import ProgressSnapshot
from acprof.tui.run.recovery_controller import RecoveryController
from acprof.tui.run.stages import Stage
from acprof.tui.settings.controller import ConfigurationController
from acprof.tui.settings.localization_controller import LocalizationController
from acprof.tui.ui.dialogs import ConfirmActionScreen
from acprof.tui.ui.input import BarCursorApp, BarCursorInput as Input
from acprof.tui.ui.pages.images import compose_images_tab
from acprof.tui.ui.pages.monitor import compose_monitor_tab
from acprof.tui.ui.pages.run import compose_run_tab
from acprof.tui.ui.pages.settings import compose_settings_tab
from acprof.tui.ui.rendering import CjkScreen
from acprof.tui.ui.slash_controller import SlashController
from acprof.tui.ui.table import ResizableDataTable
from acprof.tui.ui.themes import THEME_CATALOG

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
        self.execution = ExecutionCoordinator()
        self.run_ui = RunPresenter(self)
        self.ui_closing: bool = False
        self.read_jobs: set[object] = set()
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
            yield from compose_results_tab(self)
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
        self.localization.set_text(self.query_one('#settings-location', Static), message_id('settings.load.file_path', self.configuration.settings_path))
        self.update_responsive_layout()
        self.configuration.selected_preset = self.configuration.select("run-preset")
        self.form_ready = True
        self.results.restore_recent_result()
        self.results.update_report_actions()
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
        process = self.execution.lifecycle.process
        return OperationState(
            process=process is not None or bool(self.execution.session.kind),
            stoppable=process is not None and process.poll() is None,
            checking=self.preflight.check_running, reading=bool(self.read_jobs),
            maintenance=bool(self.images.operation) or self.images.storage_loading,
            configuring=self.configuration.environment_open or self.catalog.picker_open,
            measuring=self.execution.session.snapshot.measurement_active, closing=self.ui_closing,
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
            table.resize_enabled = not ((busy and not refreshing_image) or self.execution.session.snapshot.measurement_active)
        for widget in self.query(
            ".config-control, .experiment-picker, #model-candidates, #run-preset, .ui-preference, .profile-tool, .report-control, "
            "#save-run-default, #restore-ui-defaults, #save-ui-settings, #open-environment-settings"
        ):
            if widget.id in {"result-csv", "report-open"} and state.allows("report"):
                widget.disabled = False
                continue
            if busy and widget.has_focus:
                self.screen.set_focus(self.query_one("#main-tabs", TabbedContent).query_one(Tabs), scroll_visible=False)
            widget.disabled = busy or (state.checking and widget.id in {
                "open-environment-settings", "open-model-store",
            })
        for widget in self.query(".image-control"):
            widget.disabled = (busy and self.images.operation != "refresh") or self.execution.session.snapshot.measurement_active
        for selector, operation in {
            "#start-run": "run", "#probe-largest": "probe",
            "#summarize-results": "summary", "#plot-results": "plot",
            "#report-open": "report",
            "#report-calculate": "stats", "#stop-run": "stop",
            "#html-report-generate": "html-report", "#html-report-open": "html-open",
            "#html-report-export": "html-export",
        }.items():
            self.query_one(selector, Button).disabled = not state.allows(operation)
        self.query_one("#html-report-cancel", Button).disabled = not (
            self.execution.session.kind == "html-report" and state.allows("stop"))
        for selector in ("#result-csv",):
            self.query_one(selector, Input).disabled = not state.allows("summary")
        self.refresh_bindings()
        if not busy:
            self.set_input_cursor_blink_enabled(True)
        self.images.update_image_controls()
        self.images.sync_image_refresh_timer()
        self.preflight.refresh_preflight_state()
        self.profiles.update_action_availability()
        self.results.update_report_actions()

    def activate_tab(self, tab_id: str) -> None:
        if self.screen.maximized is not None:
            self.screen.minimize()
        tabs = self.query_one("#main-tabs", TabbedContent)
        # Move focus before hiding the outgoing pane. Widget.focus() defers
        # this change, allowing an old pane's focus event to undo the switch.
        self.screen.set_focus(tabs.query_one(Tabs), scroll_visible=False)
        tabs.active = tab_id


    def launch(self, pending: PendingLaunch) -> None:
        self.run_ui.launch(pending)

    def tick_elapsed(self) -> None:
        self.run_ui.tick_elapsed()

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
        self.run_ui.process_cleanup_incomplete(result)

    def process_started(self, pid: int, kind: str) -> None:
        self.run_ui.process_started(pid, kind)

    def show_suppressed_count(self, count: int) -> None:
        self.run_ui.show_suppressed_count(count)

    def show_deferred_lines(self, lines: tuple[str, ...], discarded_entries: int = 0,
                            oversized_entries: int = 0) -> None:
        self.run_ui.show_deferred_lines(lines, discarded_entries, oversized_entries)

    def write_log(self, line: str) -> None:
        self.run_ui.write_log(line)

    def consume_process_line(
        self, line: str, snapshot: ProgressSnapshot | None, state_changed: bool,
    ) -> None:
        self.run_ui.consume_process_line(line, snapshot, state_changed)

    @staticmethod
    def format_elapsed(seconds: float) -> str:
        return RunPresenter.format_elapsed(seconds)

    def render_snapshot(self, snapshot: ProgressSnapshot) -> None:
        self.run_ui.render_snapshot(snapshot)

    def process_finished(
        self, kind: str, returncode: int, snapshot: ProgressSnapshot | None, launch_error: str,
    ) -> None:
        self.run_ui.process_finished(kind, returncode, snapshot, launch_error)

    @on(Button.Pressed, "#stop-run, #results-cancel")
    def stop_button(self) -> None:
        self.action_request_stop()

    def action_request_stop(self) -> None:
        if not self.operation_state().allows("stop"):
            self.notify("当前没有运行中的任务", severity="warning")
            return
        token = self.execution.session.token
        html_report = self.execution.session.kind == "html-report"
        self.push_screen(
            ConfirmActionScreen(
                message_id('results.html.cancel_confirm_title') if html_report else "终止当前任务？",
                message_id('results.html.cancel_confirm_details') if html_report else "将先向整个采集进程组发送 SIGINT，允许容器和监控器清理；"
                "超时后才会升级为 SIGTERM。已写入的 case 结果不会删除。",
                "终止任务",
                variant="error",
            ),
            lambda confirmed: self.confirmed_stop(confirmed) if token is self.execution.session.token else None,
        )

    def confirmed_stop(self, confirmed: bool | None) -> None:
        if not confirmed or not self.operation_state().allows("stop"):
            return
        if not self.execution.request_stop(self.execution.session.token):
            return
        self.preflight.close_preparation()
        self.stop_process_gracefully(self.execution.session.token, self.execution.lifecycle.process)

    @work(thread=True, group="stop", exclusive=True, exit_on_error=False)
    def stop_process_gracefully(self, token: object, process) -> None:
        self.process.safe_process_callback(self.process.deliver_process_callback, token, self.write_log,
                                    "[TUI] 正在请求采集进程安全停止……")
        if process is None:
            return  # The child may exit between confirmation and this worker.
        result = self.execution.stop(expected_process=process)
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
        result = await asyncio.to_thread(self.execution.close)
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

    @on(SelectableLog.Updated)
    def on_log_updated(self) -> None:
        self.logs.refresh_status()

    @on(Input.Changed, '#log-search')
    @on(Select.Changed, '#log-level')
    def on_log_view_changed(self) -> None:
        self.logs.view_changed()

    @on(Input.Submitted, '#log-search')
    def on_log_next(self) -> None:
        self.logs.find()

    @on(Button.Pressed, '#copy-log-all')
    def on_log_copy_all(self) -> None:
        self.logs.copy_all()

    @on(Button.Pressed, '#log-tools')
    def on_log_tools(self) -> None:
        self.logs.toggle_tools()

    @on(Button.Pressed, '#export-log')
    def on_log_export(self) -> None:
        self.logs.export()

    @on(Button.Pressed, '#copy-log-export-path')
    def on_log_export_copy_path(self) -> None:
        self.logs.copy_export_path()

    @on(Button.Pressed, '#view-log-export-path')
    def on_log_export_view_path(self) -> None:
        self.logs.view_export_path()

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

    @on(Input.Changed, '#report-json-path')
    def on_results_report_json_changed(self) -> None:
        self.results.report_json_changed()

    @on(Input.Changed, '#html-report-path')
    def on_results_html_path_changed(self) -> None:
        self.results.update_report_actions()

    @on(Button.Pressed, '#analysis-export-csv, #analysis-export-json')
    def on_analysis_export(self, event: Button.Pressed) -> None:
        self.results.export_analysis(event.button.name or "csv")

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
        if self.execution.session.kind == "html-report":
            self.action_request_stop()

    @on(Button.Pressed, '#environment-status')
    def on_preflight_show_environment_status(self) -> None:
        self.preflight.show_environment_status()

    @on(Button.Pressed, '#open-model-store')
    def on_preflight_open_model_store(self) -> None:
        self.preflight.open_model_store()

    @on(Button.Pressed, '#open-result-profiling')
    def on_profiles_open_selection(self) -> None:
        self.profiles.open_selection_dialog()

    @on(Input.Submitted, '#slash-command')
    def on_slash_slash_command_submitted(self, event: TextualInput.Submitted) -> None:
        self.slash.slash_command_submitted(event)

    @on(Button.Pressed, '#start-run')
    def on_experiments_start_run_button(self) -> None:
        self.experiments.start_run_button()

    @on(Button.Pressed, '#process-inherited-resources')
    def on_process_inherited_resources(self) -> None:
        self.configuration.use_process_inherited_resources()

    @on(Button.Pressed, '#probe-largest')
    def on_experiments_probe_largest_button(self) -> None:
        self.experiments.probe_largest_button()
