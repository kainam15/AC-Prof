"""AC-Prof TUI 事件处理与子进程生命周期。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Literal

from acprof import __version__, platform
from acprof.tui.i18n import message_id

try:
    from rich.console import Console
    from textual import events, on, work
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.content import Content
    from textual.geometry import Size
    from textual.screen import ModalScreen
    from textual.theme import Theme
    from textual.timer import Timer
    from textual.widgets import (
        Button,
        Header,
        Static,
        TabbedContent,
        TextArea,
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
from acprof.tui.busy_state import BusyControlCache, apply_busy_state
from acprof.tui.commands import (
    OperationState,
    PendingLaunch,
)
from acprof.tui.event_routes import AppEventRoutes
from acprof.tui.experiments.controller import CatalogController
from acprof.tui.images.controller import ImageController
from acprof.tui.images.presentation import ImageWorkspace
from acprof.tui.logging.controller import LogController
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
from acprof.tui.settings.controller import ConfigurationController
from acprof.tui.settings.localization_controller import LocalizationController
from acprof.tui.ui.command_palette import CommandPaletteScreen
from acprof.tui.ui.dialogs import ConfirmActionScreen
from acprof.tui.ui.environment import (
    EnvironmentHostScreen,
    EnvironmentIndicators,
    EnvironmentPopover,
)
from acprof.tui.ui.input import BarCursorInput as Input
from acprof.tui.ui.navigation import PageNavigation
from acprof.tui.ui.pages.images import compose_images_tab
from acprof.tui.ui.pages.monitor import compose_monitor_tab
from acprof.tui.ui.pages.run import compose_run_tab
from acprof.tui.ui.pages.settings import compose_settings_tab
from acprof.tui.ui.rendering import CjkScreen
from acprof.tui.ui.shortcuts import ShortcutsScreen
from acprof.tui.ui.slash_controller import SlashController
from acprof.tui.ui.themes import THEME_CATALOG
from acprof.tui.updates import UpdateController

PROJECT_DIR = Path.cwd()
# Keep the virtual-environment path. Resolving this symlink would turn
# ``.venv/bin/python`` into the system interpreter and lose the venv.
PYTHON_EXECUTABLE = Path(sys.executable).absolute()


class AcprofTui(AppEventRoutes):
    """Full-screen controller for AC-Prof collection and diagnostics."""

    TITLE = f"AC-Prof {__version__}"
    ENABLE_COMMAND_PALETTE = False
    ALLOW_IN_MAXIMIZED_VIEW = "#app-header, #environment-popover"

    BINDINGS = [
        ("f5", "request_run", "开始采集"),
        Binding("f6", "quick_check", "环境检查", priority=True),
        ("f8", "toggle_log_view", "放大日志"),
        ("f2", "show_settings", "应用设置"),
        ("alt+left", "return_from_issue", "返回原页"),
        Binding("alt+1", "select_page('run-tab')", "实验配置", show=False),
        Binding("alt+2", "select_page('monitor-tab')", "运行监控", show=False),
        Binding("alt+3", "select_page('results-tab')", "实验结果", show=False),
        Binding("alt+4", "select_page('images-tab')", "镜像管理", show=False),
        Binding("alt+5", "select_page('settings-tab')", "应用设置", show=False),
        Binding("alt+e", "toggle_status_center", "环境状态", show=False),
        Binding("ctrl+k", "open_command_palette", "快捷命令", priority=True, show=False),
        Binding("slash", "open_command_palette_key", "快捷命令", show=False),
        Binding("f1", "show_shortcuts", "快捷键", priority=True),
        Binding("question_mark", "show_shortcuts_key", "快捷键", show=False),
        Binding("f3", "toggle_run_settings", "基本/高级配置", show=False),
        Binding("ctrl+x", "request_stop", "终止任务", priority=True),
        ("ctrl+l", "clear_log", "清空日志"),
        ("ctrl+q", "request_quit", "退出"),
    ]

    CSS_PATH = Path(__file__).with_name("tui.tcss")
    IMAGE_REFRESH_INTERVAL = 5.0
    PERMISSION_REFRESH_INTERVAL = 30.0

    @property
    def project_dir(self) -> Path:
        return PROJECT_DIR

    @property
    def python_executable(self) -> Path:
        return PYTHON_EXECUTABLE


    def format_title(self, title: str, sub_title: str) -> Content:
        content = super().format_title(title, sub_title)
        platform_name = self.environment.label.split(" / ")[0]
        if title == self.title and title.endswith(f" · {platform_name}"):
            return content.stylize("dim", len(title) - len(platform_name), len(title))
        return content

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
        self.environment = platform.detect_environment()
        self.title = f"{self.TITLE} · {self.environment.label.split(' / ')[0]}"
        self.elapsed_timer: Timer | None = None
        self.configuration = ConfigurationController(self, initial_config, settings_path=settings_path)
        self.localization = LocalizationController(self)
        self.experiments = ExperimentController(self)
        self.preflight = PreflightController(self)
        self.updates = UpdateController(self, state_root=settings_path.parent / "versions" if settings_path else None)
        self.results = ResultController(self)
        self.profiles = ProfileController(self)
        self.recovery = RecoveryController(self)
        self.images = ImageController(self)
        self.models = ModelController(self)
        self.catalog = CatalogController(self)
        self.process = ProcessController(self)
        self.logs = LogController(self)
        self.slash = SlashController(self)
        self._busy_controls = BusyControlCache()
        self._manual_navigation_collapsed: bool | None = None
        for palette in THEME_CATALOG:
            self.register_theme(Theme(**palette.theme_kwargs()))
        self.theme = self.configuration.preferences.theme

    def get_default_screen(self) -> CjkScreen:
        return EnvironmentHostScreen(id="_default")

    def compose(self) -> ComposeResult:
        # A ticking clock would force periodic redraws during RAPL windows.
        with Vertical(id="app-header"):
            with Header(show_clock=False):
                with Horizontal(id="header-actions"):
                    back = self.localization.localized_widget(Button(
                        message_id('run.status.return'), id='status-return', compact=True,
                        tooltip=message_id('run.status.return_hint'),
                    ))
                    back.display = False
                    yield back
                    yield EnvironmentIndicators(self.preflight)
                    yield self.localization.localized_widget(Button(
                        "?", id="show-shortcuts", name="快捷键",
                        tooltip=message_id("common.shortcuts.header_tooltip"), compact=True,
                    ))
                    yield self.localization.localized_widget(Button(
                        "×", id="quit-app", name="退出", tooltip="退出（Ctrl+Q）", compact=True,
                    ))
        yield EnvironmentPopover(self.preflight)
        with Horizontal(id="workspace"):
            yield PageNavigation(id="page-navigation")
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
        self.localization.set_text(
            self.query_one("#main-tabs", TabbedContent).get_tab("settings-tab"),
            message_id("settings.app.tooltip"), "tooltip",
        )
        self.configuration.apply_ui_preferences()
        result_views = self.query_one("#result-views", TabbedContent)
        # Keep backend-bound widgets mounted, but prevent programmatic focus
        # from reactivating the hidden pane.
        for pane in ("comparison-view", "exports-view"):
            result_views.disable_tab(pane)
            result_views.hide_tab(pane)
        self.localization.set_text(self.query_one('#settings-location', Static), message_id('settings.load.file_path', self.configuration.settings_path))
        self.configuration.refresh_connection_badges()
        self.update_responsive_layout()
        self._busy_controls.capture(self)
        self.form_ready = True
        self.models.preview_arguments = self.models.current_arguments()
        self.results.restore_recent_result()
        self.results.update_report_actions()
        self.results.sync_result_actions_for_view(result_views.active)
        self.configuration.refresh_command_preview(notify=False)
        if not self.read_jobs:
            self.call_after_refresh(self.preflight.quick_check)
        self.set_interval(self.PERMISSION_REFRESH_INTERVAL, self.preflight.check_permission_changes)
        self.query_one("#model", Input).focus()
        self.images.refresh_timer = self.set_interval(
            self.IMAGE_REFRESH_INTERVAL, self.images.refresh_images, pause=True,
        )
        if self.configuration.settings_warning:
            self.notify(self.configuration.settings_warning, title="设置读取提示", severity="warning", timeout=8)

    async def on_event(self, event: events.Event) -> None:
        # Close the overlay before Escape reaches an editor or a maximized log.
        if (self.form_ready and not self.ui_closing and isinstance(event, events.Key)
                and event.key == "escape" and not event.is_forwarded):
            panel = self.preflight.panel
            if panel.display and panel.screen is self.screen:
                panel.close()
                return
        await super().on_event(event)

    def on_app_focus(self, _event: events.AppFocus) -> None:
        if self.form_ready:
            self.preflight.check_permission_changes()

    def on_resize(self, event: events.Resize) -> None:
        # App.size can still refer to the previous frame while Resize is
        # dispatched. Use the event's new dimensions for responsive classes.
        self.update_responsive_layout(event.size)
        if self.form_ready:
            self.preflight.panel.fit(event.size)

    @on(ImageWorkspace.WidthChanged)
    def refresh_resized_images(self) -> None:
        """Use the settled page width after the sidebar and terminal resize."""
        if (self.form_ready and self.images.inventory is not None and not self.images.images_unavailable(allow_refresh=True)
                and self.query_one("#main-tabs", TabbedContent).active == "images-tab"):
            self.images.render_images()

    def update_responsive_layout(self, size: Size | None = None) -> None:
        size = self.size if size is None else size
        collapsed = (
            self._manual_navigation_collapsed
            if self._manual_navigation_collapsed is not None
            else size.width < 92
        )
        # Reserve sidebar cells before choosing the form's column layout.
        navigation_width = 5 if collapsed else (17 if size.width < 100 else 22)
        # Textual updates the entire stylesheet after each class change.
        # Make the responsive switch atomic to avoid two redundant restyles.
        changed_collapse = self.has_class("collapsed-navigation") != collapsed
        self.update_classes({
            "collapsed-navigation": collapsed,
            "narrow": size.width - navigation_width < 110,
            "compact-navigation": size.width < 100,
            "short": size.height < 35,
        })
        if changed_collapse and self.screen_stack and self.query("#page-navigation"):
            self.query_one(PageNavigation).sync(self.query_one("#main-tabs", TabbedContent))


    def is_busy(self) -> bool:
        return self.operation_state().busy

    def operation_state(self) -> OperationState:
        process = self.execution.lifecycle.process
        return OperationState(
            process=process is not None or bool(self.execution.session.kind),
            stoppable=self.execution.session.launch_pending or (process is not None and process.poll() is None),
            checking=self.preflight.check_running or self.updates.checking or self.models.preview_worker is not None,
            reading=bool(self.read_jobs),
            maintenance=bool(self.images.operation) or self.images.storage_loading or self.updates.installing,
            restarting=self.updates.needs_restart,
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
        apply_busy_state(self, self._busy_controls, busy)
        if not busy:
            self.models.resume_preview()

    def activate_tab(self, tab_id: str) -> None:
        if tab_id == "backfill-tab":
            self.results.show_result_section("backfill-view")
            return
        if self.screen.maximized is not None:
            self.screen.minimize()
        tabs = self.query_one("#main-tabs", TabbedContent)
        # Move focus before hiding the outgoing pane. Widget.focus() defers
        # this change, allowing an old pane's focus event to undo the switch.
        self.screen.set_focus(self.query_one(f"#nav-{tab_id}"), scroll_visible=False)
        tabs.active = tab_id

    @on(PageNavigation.Selected)
    def on_page_selected(self, event: PageNavigation.Selected) -> None:
        tabs = self.query_one("#main-tabs", TabbedContent)
        # Only reactivating the selected item toggles sidebar visibility.
        # Switching pages preserves the current presentation preference.
        if tabs.active == event.pane_id:
            self._toggle_navigation_collapsed()
        self.activate_tab(event.pane_id)
        self.query_one(PageNavigation).sync(tabs)

    @on(PageNavigation.CollapseToggled)
    def on_page_collapse_toggled(self) -> None:
        self._toggle_navigation_collapsed()

    def _toggle_navigation_collapsed(self) -> None:
        self._manual_navigation_collapsed = not self.has_class("collapsed-navigation")
        self.update_responsive_layout()
        self.query_one(PageNavigation).sync(self.query_one("#main-tabs", TabbedContent))

    @on(Button.Pressed, "#show-shortcuts")
    def on_show_shortcuts_button(self) -> None:
        self.action_show_shortcuts()


    def launch(self, pending: PendingLaunch) -> None:
        self.run_ui.launch(pending)

    def tick_elapsed(self) -> None:
        self.run_ui.tick_elapsed()

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

    @on(Button.Pressed, "#stop-run")
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
        self.results.invalidate_reads()
        self.ui_closing = True
        self.exit()

    async def on_unmount(self) -> None:
        """Use the same bounded cleanup policy even after widgets are gone."""
        self.form_ready = False
        self.ui_closing = True
        self._busy_controls.clear()
        await self.models.close_preview()
        if self.recovery.recovery_request is not None:
            self.recovery.recovery_request.set()
        self.recovery.recovery_after_check = None
        self.results.invalidate_reads()
        if self.results.report_request is not None:
            self.results.report_request.set()
        self.results.report_request = self.preflight.check_request = None
        self.configuration.cancel_preview_timer()
        self.preflight.cancel_auto_check()
        self.preflight.check_running = False
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

    def action_show_shortcuts(self) -> None:
        if self.ui_closing:
            return
        if isinstance(self.screen, ShortcutsScreen):
            self.screen.dismiss()
            return
        if isinstance(self.screen, ModalScreen):
            return
        self.push_screen(ShortcutsScreen())

    def action_show_shortcuts_key(self) -> None:
        if self.ui_closing:
            return
        if isinstance(self.screen, ShortcutsScreen):
            self.screen.dismiss()
            return
        if isinstance(self.screen, ModalScreen) or isinstance(self.focused, (Input, TextArea)):
            return
        self.push_screen(ShortcutsScreen())

    def action_toggle_run_settings(self) -> None:
        if not self.ui_closing and not isinstance(self.screen, ModalScreen):
            self.configuration.open_run_settings()

    def action_select_page(self, pane_id: str) -> None:
        if not self.ui_closing and not isinstance(self.screen, ModalScreen):
            self.activate_tab(pane_id)
            self.query_one(PageNavigation).sync(self.query_one("#main-tabs", TabbedContent))

    def action_toggle_status_center(self) -> None:
        if self.ui_closing or isinstance(self.screen, ModalScreen) or not self.form_ready:
            return
        panel = self.preflight.panel
        if panel.display:
            panel.close()
            return
        counts = self.preflight.status.counts
        group = next(
            (name for name in ("error", "warning", "pending", "success") if counts.get(name)),
            "success",
        )
        panel.toggle(group=group)

    def action_open_command_palette(self) -> None:
        if self.ui_closing or not self.form_ready:
            return
        if isinstance(self.screen, CommandPaletteScreen):
            self.screen.dismiss(None)
            return
        if isinstance(self.screen, ModalScreen):
            return
        self.push_screen(
            CommandPaletteScreen(),
            lambda command: self.slash.execute_command(command, from_palette=True) if command else None,
        )

    def action_open_command_palette_key(self) -> None:
        if self.ui_closing or isinstance(self.screen, ModalScreen) or isinstance(self.focused, (Input, TextArea)):
            return
        self.action_open_command_palette()
