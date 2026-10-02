"""AC-Prof TUI 事件处理与子进程生命周期。"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Literal, Sequence

try:
    from rich.console import Console
    from rich.text import Text
    from textual import events, on, work
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.geometry import Size
    from textual.screen import ModalScreen
    from textual.theme import Theme
    from textual.widget import Widget
    from textual.widgets import (
        Button,
        Checkbox,
        Collapsible,
        ContentSwitcher,
        DataTable,
        Header,
        Select,
        Static,
        TabbedContent,
        TabPane,
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

from acprof.experiment import RunConfig, RunConfigError, build_run_command
from acprof.host.image_management import (
    ImageInventory,
    ImageLayer,
    ManagedImage,
)
from acprof.messages import join_messages, message
from acprof.platform import collection_policy_error, detect_environment
from acprof.tui import run_form
from acprof.tui.commands import (
    PendingLaunch,
    build_probe_command,
    format_command,
    parse_slash_command,
    prepare_plot,
    prepare_profile,
    prepare_stats,
    resolve_result_path,
)
from acprof.tui.diagnostics import (
    PreflightCheck,
    collection_preview,
    quick_preflight,
    summarize_result_csv,
)
from acprof.tui.i18n import error_message, translate
from acprof.tui.image_actions import ImageActions
from acprof.tui.input import BarCursorApp, BarCursorInput as Input
from acprof.tui.log import SelectableLog
from acprof.tui.presentation import CALCULATING, NOT_APPLICABLE, UNKNOWN
from acprof.tui.process import ProcessLifecycle, StopResult
from acprof.tui.progress import ProgressSnapshot, RunProgressTracker
from acprof.tui.rendering import CjkScreen
from acprof.tui.reports import ReportView, read_report
from acprof.tui.scrollbar import SolidScrollBarRender
from acprof.tui.settings import (
    UiPreferences,
    default_settings_path,
    load_settings,
    save_settings,
)
from acprof.tui.table import ResizableDataTable
from acprof.tui.themes import THEME_CATALOG
from acprof.tui.views import (
    ConfirmActionScreen,
    LogPanel,
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


class AcprofTui(ImageActions, BarCursorApp):
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
        config = self._saved_settings.run_defaults or RunConfig()
        if self._saved_settings.last_model:
            config = replace(config, model=self._saved_settings.last_model)
        self.initial_config = initial_config if initial_config is not None else config
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
        self._resolution_open = False
        self._environment_open = False
        self._last_resolution = None
        self._preparation_screen = None
        self._preparation_request: tuple[subprocess.Popen[str], int] | None = None
        self._form_ready = False
        self._applying_config = False
        self._preview_timer = None
        self._ignored_preset_event: str | None = None
        self._initial_preset = run_form.infer_preset(self.initial_config)
        self._elapsed_timer = None
        self._report_view: ReportView | None = None
        self._report_loading = False
        self._stats_report_path: Path | None = None
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
        self._update_saved_settings_summary()
        self._update_responsive_layout()
        self._form_ready = True
        self._refresh_command_preview(notify=False)
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

    def tr(self, source: str) -> str:
        return translate(source, self.ui_preferences.language)

    def notify(self, message: str, *, title: str = "", **kwargs) -> None:
        super().notify(self.tr(message), title=self.tr(title), **kwargs)

    def _localized_widget(self, widget: Widget) -> Widget:
        """Register only our own widgets, before Textual creates their children."""
        if isinstance(widget, (Button, Checkbox)):
            self._localized_text[widget, "label"] = widget.label.plain
        elif isinstance(widget, Static):
            self._localized_text[widget, "content"] = widget.content
        elif isinstance(widget, Input):
            self._localized_text[widget, "placeholder"] = widget.placeholder
        elif isinstance(widget, Collapsible):
            self._localized_text[widget, "title"] = widget.title
        if isinstance(widget.tooltip, str):
            self._localized_text[widget, "tooltip"] = widget.tooltip
        return widget

    def _localized_select(self, options, **kwargs) -> Select:
        sources = tuple(options)
        widget = Select(sources, **kwargs)
        self._localized_selects[widget] = sources
        return widget

    def _set_text(self, widget: Widget, source: str, attribute: str = "content") -> None:
        self._localized_text[widget, attribute] = source
        self._render_text(widget, attribute, source)

    def _render_text(self, widget: Widget, attribute: str, source: str) -> None:
        rendered = self.tr(source)
        if attribute == "content":
            assert isinstance(widget, Static)
            widget.update(rendered)
        else:
            setattr(widget, attribute, rendered)
            # Button labels do not invalidate cached content widths themselves.
            # Recompute geometry when switching a visible page back and forth.
            widget.refresh(layout=True)

    def _capture_language_text(self) -> None:
        tabs = self.query_one("#main-tabs", TabbedContent)
        for pane in tabs.query(TabPane):
            tab = tabs.get_tab(pane)
            self._localized_text[tab, "label"] = tab.label.plain
        self._source_bindings = {
            key: list(bindings) for key, bindings in self._bindings.key_to_bindings.items()
        }

    def _apply_language(self) -> None:
        if self._applied_language == self.ui_preferences.language:
            return
        self._applied_language = self.ui_preferences.language
        # Keep mounted widgets, drafts, selected values, log text/selection,
        # scroll positions and progress state. This runs only on a UI change.
        with self.prevent(Select.Changed), self.batch_update():
            self.sub_title = f"{detect_environment().label} · {self.tr(self.SUB_TITLE)}"
            for (widget, attribute), source in self._localized_text.items():
                self._render_text(widget, attribute, source)
            for widget, sources in self._localized_selects.items():
                value = widget.value
                widget.set_options((self.tr(label), key) for label, key in sources)
                widget.value = value
                # If the selection is the first option, its value did not
                # change. Still refresh the displayed prompt from the catalog.
                widget.mutate_reactive(Select.value)
            self._bindings.key_to_bindings = {
                key: [replace(binding, description=self.tr(binding.description)) for binding in bindings]
                for key, bindings in self._source_bindings.items()
            }
            self.refresh_bindings()
            if self._report_view is not None:
                self._render_report_view()
            self._render_images()

    def _apply_ui_preferences(self) -> None:
        self._apply_language()
        self.theme = self.ui_preferences.theme
        log = self.query_one("#run-log", SelectableLog)
        log.wrap = self.ui_preferences.log_wrap
        log.max_lines = self.ui_preferences.log_max_lines
        self.query_one("#bottom-panel").set_class(
            not self.ui_preferences.show_command_bar, "command-hidden",
        )

    def _update_saved_settings_summary(self) -> None:
        config = self._saved_settings.run_defaults
        summary = (
            message('已记住：{0} · CPU {1} · 内存 {2} GB', config.model or message('模型待填写'), config.cpus, config.mems)
            if config else message("尚未保存实验默认参数。")
        )
        if self._saved_settings.last_model:
            summary = join_messages("", (
                summary, message('\n下次启动自动填入模型：{0}', self._saved_settings.last_model),
            ))
        self._set_text(self.query_one('#saved-run-summary', Static), summary)
        self._set_text(self.query_one('#settings-location', Static), message('保存位置：{0}', self.settings_path))

    @on(Select.Changed, ".ui-preference")
    @on(Checkbox.Changed, ".ui-preference")
    def _ui_preference_changed(self) -> None:
        if not self._form_ready or self._is_busy():
            return
        preferences = UiPreferences(
            language=self._select("ui-language"),
            theme=self._select("ui-theme"),
            log_max_lines=int(self.query_one("#ui-log-lines", Select).value),
            log_wrap=self._checked("ui-log-wrap"),
            show_command_bar=self._checked("ui-command-bar"),
        )
        if preferences == self.ui_preferences:
            return
        self.ui_preferences = preferences
        self._apply_ui_preferences()
        self._set_text(self.query_one('#settings-status', Static), '已应用 · 点击保存设置可在下次启动时沿用')
        self.query_one('#settings-status').set_classes('page-summary stage-running')

    @on(Button.Pressed, "#restore-ui-defaults")
    def restore_ui_defaults(self) -> None:
        if self._is_busy():
            return
        defaults = UiPreferences()
        with self.prevent(Select.Changed, Checkbox.Changed):
            self.query_one("#ui-language", Select).value = defaults.language
            self.query_one("#ui-theme", Select).value = defaults.theme
            self.query_one("#ui-log-lines", Select).value = defaults.log_max_lines
            self.query_one("#ui-log-wrap", Checkbox).value = defaults.log_wrap
            self.query_one("#ui-command-bar", Checkbox).value = defaults.show_command_bar
        self.ui_preferences = defaults
        self._apply_ui_preferences()
        self._set_text(self.query_one('#settings-status', Static), '界面已恢复默认 · 点击保存设置可保留')
        self.query_one('#settings-status').set_classes('page-summary stage-running')

    def _save_settings(self, *, remember_run: bool) -> None:
        if self._is_busy():
            self.notify("任务完成后可保存设置", severity="warning")
            return
        try:
            config = (
                self._collect_config(allow_empty_model=True)
                if remember_run else self._saved_settings.run_defaults
            )
            settings = replace(
                self._saved_settings,
                ui=self._saved_settings.ui if remember_run else self.ui_preferences,
                run_defaults=config,
            )
            save_settings(self.settings_path, settings, PROJECT_DIR)
        except (OSError, ValueError, RunConfigError) as exc:
            self.notify(error_message(exc), title="设置未保存", severity="error")
            self._set_text(self.query_one('#settings-status', Static), '保存失败 · 请检查配置或文件权限')
            self.query_one('#settings-status').set_classes('page-summary stage-error')
            return
        self._saved_settings = settings
        self._settings_warning = ""
        self._update_saved_settings_summary()
        message = "已记住当前实验配置" if remember_run else "界面设置已保存"
        if not remember_run:
            self._set_text(self.query_one('#settings-status', Static), message)
            self.query_one('#settings-status').set_classes('page-summary stage-success')
        self.notify(message, timeout=3)

    @on(Button.Pressed, "#save-ui-settings")
    def save_ui_settings(self) -> None:
        self._save_settings(remember_run=False)

    @on(Button.Pressed, "#save-run-default")
    def save_run_default(self) -> None:
        self._save_settings(remember_run=True)

    def action_show_settings(self) -> None:
        self._activate_tab("settings-tab")

    @on(Button.Pressed, "#open-environment-settings")
    def open_environment_settings(self) -> None:
        if self._is_busy() or self._check_running:
            return
        from acprof.tui.environment import EnvironmentSettingsScreen
        try:
            screen = EnvironmentSettingsScreen(
                PROJECT_DIR, self.settings_path.parent, sniff_iface=self._input("sniff-iface"),
            )
        except (OSError, ValueError) as read_error:
            self.notify(message('无法读取连接配置：{0}', type(read_error).__name__), severity="error")
            return
        self._environment_open = True
        self._set_busy(True)
        self.push_screen(screen, self._environment_closed)

    def _environment_closed(self, _saved: bool | None) -> None:
        self._environment_open = False
        self._set_busy(False)

    @on(Button.Pressed, "#open-run-settings")
    def open_run_settings(self) -> None:
        self._activate_tab("run-tab")
        pages = self.query_one("#experiment-pages", ContentSwitcher)
        show_advanced = pages.current != "advanced-form"
        pages.current = "advanced-form" if show_advanced else "run-form"
        self._set_text(self.query_one("#run-title", Static), "采集参数" if show_advanced else "配置实验")
        self._set_text(
            self.query_one("#open-run-settings", Button),
            "返回基本配置" if show_advanced else "高级参数", "label",
        )


    def _gpu_options(self) -> list[tuple[str, str]]:
        options = [("仅 CPU", "off"), ("仅 GPU", "on"), ("CPU + GPU", "off,on")]
        if self.initial_config.gpus not in {value for _, value in options}:
            options.append((message('自定义：{0}', self.initial_config.gpus), self.initial_config.gpus))
        return options


    def _input(self, widget_id: str) -> str:
        return self.query_one(f"#{widget_id}", Input).value.strip()

    def _select(self, widget_id: str) -> str:
        value = self.query_one(f"#{widget_id}", Select).value
        return "" if value is Select.NULL else str(value)

    def _checked(self, widget_id: str) -> bool:
        return bool(self.query_one(f"#{widget_id}", Checkbox).value)

    def _configure_interaction(self) -> None:
        # Textual ignores another click while a button's active effect lasts
        # (200 ms by default). Use focus/hover styling for immediate feedback.
        for button in self.query(Button):
            button.active_effect_duration = 0
        for field in self.query(Input):
            # The app toggles the native caret without Input's repaint timer.
            field.cursor_blink = False

    def _configure_scrollbars(self) -> None:
        # Instance-level renderers keep this behavior local to this TUI.
        for widget in self.query(Widget):
            if widget.is_scrollable:
                widget.vertical_scrollbar.renderer = SolidScrollBarRender
                widget.horizontal_scrollbar.renderer = SolidScrollBarRender

    @on(Button.Pressed, "#copy-log")
    def copy_log_selection(self) -> None:
        log = self.query_one("#run-log", SelectableLog)
        if log.copy_selection():
            self.notify("已复制日志选区", timeout=2)
        else:
            self.notify("先在日志中拖动选择文字，Ctrl+A 可全选", timeout=3)

    @on(Button.Pressed, "#follow-log")
    def follow_latest_log(self) -> None:
        self.query_one("#run-log", SelectableLog).follow_tail()

    @on(Button.Pressed, "#expand-log")
    @on(Button.Pressed, "#restore-log")
    def toggle_log_view_button(self) -> None:
        self.action_toggle_log_view()

    def action_toggle_log_view(self) -> None:
        if isinstance(self.screen, ModalScreen):
            return
        panel = self.query_one("#log-panel", LogPanel)
        if self.screen.maximized is panel:
            self.screen.minimize()
        else:
            self._activate_tab("monitor-tab")
            self.screen.maximize(panel, container=False)
        self.query_one("#run-log", SelectableLog).focus()

    def _cancel_preview_timer(self) -> None:
        if self._preview_timer is not None:
            self._preview_timer.stop()
            self._preview_timer = None

    @on(Input.Changed, ".config-control")
    @on(Select.Changed, ".config-control")
    @on(Checkbox.Changed, ".config-control")
    def _configuration_changed(self) -> None:
        if not self.is_running or not self._form_ready or self._applying_config or self._is_busy():
            return
        self._cancel_preview_timer()
        # Coalesce typing and preset field updates into one validation/render.
        self._preview_timer = self.set_timer(0.05, self._sync_form_state)

    def _sync_form_state(self) -> None:
        self._preview_timer = None
        # Timer callbacks already queued before shutdown can run after the
        # form has been removed; do not query widgets during that phase.
        if not self.is_running or not self._form_ready or self._is_busy() or self._applying_config:
            return
        self._refresh_command_preview(notify=False, sync_preset=True)

    @on(Select.Changed, "#run-preset")
    def _run_preset_changed(self, event: Select.Changed) -> None:
        if not self._form_ready or self._applying_config or self._is_busy():
            return
        preset = str(event.value)
        if self._ignored_preset_event == preset:
            self._ignored_preset_event = None
            return
        if preset == "smoke":
            self.preset_smoke()
        elif preset == "main":
            self.preset_main()
        elif preset == "default":
            self.preset_default()

    def _collect_config(self, *, allow_empty_model: bool = False) -> RunConfig:
        return run_form.collect_config(
            {key: self._input(key) for key in run_form.INPUT_FIELDS},
            {key: self._select(key) for key in run_form.SELECT_FIELDS},
            {key: self._checked(key) for key in run_form.CHECKED_FIELDS},
            project_dir=PROJECT_DIR, allow_empty_model=allow_empty_model,
        )

    def _apply_config(self, config: RunConfig, *, preset: str = "custom") -> None:
        self._cancel_preview_timer()
        self._applying_config = True
        values, selects, checks = run_form.config_values(config)
        # Value watchers post Changed messages asynchronously. Suppressing
        # them here avoids dozens of queued debounce timers after a preset.
        try:
            with self.prevent(Input.Changed, Select.Changed, Checkbox.Changed):
                with self.batch_update():
                    for widget_id, value in values.items():
                        self.query_one(f"#{widget_id}", Input).value = value
                    for widget_id, value in selects.items():
                        self.query_one(f"#{widget_id}", Select).value = value
                    for widget_id, value in checks.items():
                        self.query_one(f"#{widget_id}", Checkbox).value = value
                    self.query_one("#run-preset", Select).value = preset
                    self._ignored_preset_event = None
        finally:
            self._applying_config = False
        if self._form_ready:
            self._refresh_command_preview(notify=False, sync_preset=True)

    def _show_config_error(self, exc: RunConfigError) -> None:
        text = join_messages("\n", (message("• {0}", error) for error in exc.errors))
        self.notify(text, title="配置有误", severity="error", timeout=8)

    def _refresh_command_preview(
        self, *, notify: bool = True, sync_preset: bool = False
    ) -> bool:
        config = None
        try:
            config = self._collect_config()
            command = build_run_command(
                config,
                project_dir=PROJECT_DIR,
                python_executable=PYTHON_EXECUTABLE,
            )
        except RunConfigError as exc:
            self._set_text(
                self.query_one("#config-summary", Static),
                message("配置待完善 · {0}", join_messages("; ", exc.errors[:2])),
            )
            self._set_text(
                self.query_one("#command-preview", Static),
                message("配置尚未完成：{0}", join_messages("; ", exc.errors)),
            )
            if notify:
                self._show_config_error(exc)
            return False
        finally:
            if sync_preset and config is not None:
                selected_preset = self._select("run-preset")
                if selected_preset != "custom":
                    adjusted = not run_form.matches_preset(config, selected_preset)
                    options = tuple((message("{0} · 已调整", message(label)) if key == selected_preset and adjusted else label, key)
                        for label, key in run_form.PRESET_OPTIONS)
                    widget = self.query_one("#run-preset", Select)
                    self._localized_selects[widget] = options
                    with self.prevent(Select.Changed):
                        widget.set_options((self.tr(label), key) for label, key in options)
                        widget.value = selected_preset
        case_count = (
            len(config.cpus.split(","))
            * len(config.mems.split(","))
            * len(config.gpus.split(","))
        )
        scale_summary = (
            message('{0} 档', len(config.input_scales.split(',')))
            if config.input_scales
            else message("自动规划")
        )
        profiler_summary = (
            message("分析器关闭")
            if config.compute_profile_tool == "none"
            and config.execution_profile_tool == "none"
            else (
                message('计算={0} · 执行={1}', config.compute_profile_tool, config.execution_profile_tool)
            )
        )
        self._set_text(self.query_one("#config-summary", Static), message(
            "{0} 个资源 case · 输入规模 {1} · 单请求超时 {2:g}s · {3} · 输出 {4}",
            case_count, scale_summary, config.request_timeout_seconds,
            profiler_summary, config.output_dir,
        ))
        self._set_text(self.query_one('#command-preview', Static), format_command(command, project_dir=PROJECT_DIR))
        if notify:
            self.notify("命令预览已更新", timeout=2)
        return True

    def _is_busy(self) -> bool:
        return (self._lifecycle.process is not None or bool(self._process_kind) or self._report_loading
                or bool(self._image_operation) or self._resolution_open or self._environment_open)

    def _set_busy(self, busy: bool) -> None:
        # Configuration changes during a run can queue preview redraws and
        # make the visible settings differ from the running subprocess.
        if busy:
            self._cancel_preview_timer()
        for table in self.query(ResizableDataTable):
            refreshing_image = self._image_operation == "refresh" and table.has_class("image-control")
            table.resize_enabled = not ((busy and not refreshing_image) or self._latest_snapshot.measurement_active)
        for widget in self.query(
            ".config-control, #run-preset, .ui-preference, .profile-tool, .report-control, "
            "#save-run-default, #restore-ui-defaults, #save-ui-settings, #open-environment-settings"
        ):
            widget.disabled = busy
        for widget in self.query(".image-control"):
            widget.disabled = (busy and self._image_operation != "refresh") or self._latest_snapshot.measurement_active
        for selector in (
            "#start-run",
            "#probe-largest",
            "#quick-check",
            "#inspect-model",
            "#summarize-results",
            "#plot-results",
            "#profile-dry-run",
            "#profile-run",
        ):
            self.query_one(selector, Button).disabled = busy
        self.query_one("#stop-run", Button).disabled = not busy or self._report_loading or bool(self._image_operation)
        if not busy:
            self.set_input_cursor_blink_enabled(True)
        self._update_image_controls()
        self._sync_image_refresh_timer()

    def _activate_tab(self, tab_id: str) -> None:
        if self.screen.maximized is not None:
            self.screen.minimize()
        tabs = self.query_one("#main-tabs", TabbedContent)
        # Move focus before hiding the outgoing pane. Widget.focus() defers
        # this change, allowing an old pane's focus event to undo the switch.
        self.screen.set_focus(tabs.query_one(Tabs), scroll_visible=False)
        tabs.active = tab_id

    def preset_smoke(self) -> None:
        self._apply_config(RunConfig.smoke(self._input("model")), preset="smoke")
        self.notify("已应用基础 CPU Smoke 预设", timeout=3)

    def preset_main(self) -> None:
        self._apply_config(
            RunConfig.main_matrix(self._input("model")),
            preset="main",
        )
        self.notify("已应用主矩阵预设（分析器关闭）", timeout=3)

    def preset_default(self) -> None:
        self._apply_config(
            RunConfig(model=self._input("model")),
            preset="default",
        )
        self.notify("已恢复完整默认配置", timeout=3)

    @on(Button.Pressed, "#start-run")
    def start_run_button(self) -> None:
        self.action_request_run()

    @on(Button.Pressed, "#probe-largest")
    def probe_largest_button(self) -> None:
        self.action_request_probe()

    def action_request_probe(self) -> None:
        if self._is_busy():
            self.notify("已有任务正在运行", severity="warning")
            return
        try:
            config = self._collect_config()
            command = build_probe_command(
                config,
                project_dir=PROJECT_DIR,
                python_executable=PYTHON_EXECUTABLE,
            )
        except RunConfigError as exc:
            self._show_config_error(exc)
            return

        cpu = min(int(value) for value in config.cpus.split(","))
        memory_candidates = sorted(
            set(int(value) for value in config.mems.split(","))
        )
        gpu_modes = config.gpus.split(",")
        gpu = "off" if "off" in gpu_modes else "on"
        largest_scale = (
            max(float(value) for value in config.input_scales.split(","))
            if config.input_scales
            else None
        )
        scale_text = f"{largest_scale:g}" if largest_scale is not None else message("自动规划后的最大值")
        memory_text = ",".join(
            f"{value}GB" for value in memory_candidates
        )
        preview = format_command(command, project_dir=PROJECT_DIR)
        self._pending_launch = PendingLaunch(tuple(command), "probe", config)
        self.push_screen(
            ConfirmActionScreen(
                "探测最低配置的最大输入？",
                join_messages("", (
                    message(
                        "资源：CPU={0}、GPU={1}\n内存候选：{2}（从小到大）\n输入规模：{3}\n\n"
                        "每档使用全新容器并最多执行一次最大输入请求；OOM 时自动尝试下一档，"
                        "第一个成功值就是最低可用内存。结果单独写入 "
                        "probes/，不会写入或修改正式实验 CSV。最大输入请求不设超时，"
                        "可用 /stop 手动终止。\n\n",
                        cpu, gpu, memory_text, scale_text,
                    ),
                    preview,
                )),
                "开始探测",
            ),
            self._confirmed_launch,
        )

    def action_request_run(self) -> None:
        if self._is_busy():
            self.notify("已有任务正在运行", severity="warning")
            return
        try:
            config = self._collect_config()
            environment = detect_environment()
            error = collection_policy_error(environment, profiling_mode=config.profiling_mode,
                                            compute_tool=config.compute_profile_tool,
                                            execution_tool=config.execution_profile_tool)
            if environment.environment == "wsl2" and error:
                self.notify(error, severity="error", timeout=10)
                return
            command = build_run_command(
                config,
                project_dir=PROJECT_DIR,
                python_executable=PYTHON_EXECUTABLE,
            )
        except RunConfigError as exc:
            self._show_config_error(exc)
            return
        preview = format_command(command, project_dir=PROJECT_DIR)
        self._set_text(self.query_one('#command-preview', Static), preview)
        self._pending_launch = PendingLaunch(tuple(command), "run", config)
        self.push_screen(
            ConfirmActionScreen(
                "开始 AC-Prof 采集？",
                join_messages("", (
                    message("将启动独立采集进程。正式测量窗口内 TUI 会停止常规日志刷新。\n\n"),
                    collection_preview(config),
                    preview,
                )),
                "开始采集",
            ),
            self._confirmed_launch,
        )

    def _confirmed_launch(self, confirmed: bool | None) -> None:
        pending = self._pending_launch
        self._pending_launch = None
        self._sync_image_refresh_timer()
        if not confirmed or pending is None:
            return
        self._launch(pending)

    def _remember_last_used(
        self, *, model: str = "", result_dir: str = "", result_csv: str = "",
    ) -> None:
        """Remember confirmed inputs, preserving explicitly saved preferences."""
        updates = {}
        report_source = self.query_one("#report-source", Input)
        if result_csv and report_source.value in {"", self._saved_settings.last_result_csv}:
            report_source.value = result_csv
        if model.strip():
            updates["last_model"] = model.strip()
        for widget_id, value in (("result-dir", result_dir), ("result-csv", result_csv)):
            if value:
                self.query_one(f"#{widget_id}", Input).value = value
                updates[f"last_{widget_id.replace('-', '_')}"] = value
        settings = replace(self._saved_settings, **updates)
        if settings == self._saved_settings:
            return
        if self._settings_warning:
            self.notify(
                "本地设置无法读取，已保留原文件。可在设置页主动保存后恢复自动记忆。",
                title="自动记忆未保存", severity="warning",
            )
            return
        try:
            save_settings(self.settings_path, settings, PROJECT_DIR)
        except (OSError, ValueError, RunConfigError) as exc:
            self.notify(error_message(exc), title="自动记忆未保存", severity="warning")
            return
        self._saved_settings = settings
        self._update_saved_settings_summary()

    def _launch(self, pending: PendingLaunch) -> None:
        if self._is_busy():
            self.notify("已有任务正在运行", severity="warning")
            return
        model = ""
        result_dir, result_csv = pending.result_dir, pending.result_csv
        if pending.kind in {"run", "probe"} and pending.config is not None:
            model = pending.config.model
            if pending.kind == "run":
                result_dir = str(pending.config.result_dir(PROJECT_DIR))
                result_csv = str(pending.config.result_csv(PROJECT_DIR))
        # One atomic save before the subprocess exists, outside measurement
        # windows. Failed or interrupted attempts retain their intended paths.
        self._remember_last_used(model=model, result_dir=result_dir, result_csv=result_csv)
        self._active_run_config = pending.config if pending.kind == "run" else None
        self._active_command = pending.command
        self._process_kind = pending.kind
        self._started_monotonic = time.monotonic()
        self._stop_requested = False
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
        log.write(f"$ {format_command(pending.command, project_dir=PROJECT_DIR)}")
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
        "正式测量": "stage-measuring",
        "最大尺度探测": "stage-measuring",
    }
    _STAGE_CLASSES = frozenset({
        "stage-idle", "stage-running", "stage-measuring",
        "stage-success", "stage-error",
    })

    @work(thread=True, group="process", exclusive=True, exit_on_error=False)
    def _execute_command(self, command: list[str], kind: str) -> None:
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
            # run.py otherwise captures the entire full-screen pane, including
            # ANSI redraws, when the TUI itself is launched inside tmux.
            child_env.pop("TMUX", None)
            child_env.pop("TMUX_PANE", None)
            process = self._lifecycle.start(command, cwd=PROJECT_DIR, env=child_env)
            if self._stop_requested:
                self._lifecycle.stop()
            self.call_from_thread(self._process_started, process.pid, kind)
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
                        self.call_from_thread(
                            self._show_deferred_lines,
                            tuple(deferred_important_lines),
                        )
                        deferred_important_lines.clear()
                    if suppressed_lines:
                        self.call_from_thread(
                            self._show_suppressed_count,
                            suppressed_lines,
                        )
                        suppressed_lines = 0
                self.call_from_thread(
                    self._consume_process_line,
                    "" if line.startswith(("ACPROF_EVENT ", PREFIX)) else line,
                    snapshot,
                    state_changed,
                )
                preparation = parse_preparation_event(line)
                if line.startswith("[network-preflight] "):
                    self.call_from_thread(self._network_preflight_report, line)
                elif line.startswith("[network-download] "):
                    self.call_from_thread(self._network_download_report, line)
                if preparation is not None:
                    self.call_from_thread(self._preparation_event, preparation)
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
                self._watch_failed_process(process, kind, final_snapshot, launch_error)
            else:
                if process is not None:
                    if process.stdout is not None:
                        process.stdout.close()
                    self._lifecycle.release(process)
                    returncode = process.returncode
                if suppressed_lines:
                    self._safe_process_callback(self._show_suppressed_count, suppressed_lines)
                if deferred_important_lines:
                    self._safe_process_callback(self._show_deferred_lines, tuple(deferred_important_lines))
                self._safe_process_callback(
                    self._process_finished, kind, returncode, final_snapshot, launch_error,
                )

    def _safe_process_callback(self, callback, *args) -> None:
        try:
            self.call_from_thread(callback, *args)
        except Exception as exc:
            # Widgets may already have been pruned during shutdown. Process
            # cleanup and ownership must not depend on their availability.
            print(f"[TUI] callback failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    @work(thread=True, group="process-reap", exit_on_error=False)
    def _watch_failed_process(self, process, kind, snapshot, error) -> None:
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
        self._preparation_request = None
        if self._preparation_screen is not None:
            self._preparation_screen.dismiss(None)
            self._preparation_screen = None
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
        if kind == "stats":
            # Move focus before disabling the monitor page's focused Stop button;
            # its queued focus event could otherwise reactivate that old page.
            self._activate_tab("reports-tab")
        self._set_busy(False)
        log = self.query_one("#run-log", SelectableLog)
        unsupported_task = (
            snapshot is not None and snapshot.stage == "任务不支持"
            and returncode != 0 and not self._stop_requested and not launch_error
        )
        if launch_error:
            log.write(self.tr(message('[TUI][ERROR] 无法运行命令：{0}', launch_error)))
            self.notify(launch_error, title="任务启动失败", severity="error", timeout=8)
        elif returncode == 0:
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
        else:
            log.write(self.tr(message('[TUI][ERROR] {0} 任务失败，退出码 {1}', kind, returncode)))
            self.notify(message('任务失败，退出码 {0}', returncode), severity="error", timeout=8)

        if unsupported_task:
            # No measurement ran. Do not show an older CSV as this run's result.
            self._latest_snapshot = snapshot
            self._render_snapshot(snapshot)
        elif kind == "run":
            if snapshot is not None:
                self._latest_snapshot = snapshot
            final_csv = snapshot.final_csv if snapshot is not None else ""
            if not final_csv and self._active_run_config is not None:
                final_csv = str(self._active_run_config.result_csv(PROJECT_DIR))
            if final_csv:
                final_path = Path(final_csv).expanduser()
                if not final_path.is_absolute():
                    final_path = PROJECT_DIR / final_path
                self._remember_last_used(
                    result_csv=str(final_path), result_dir=str(final_path.parent),
                )
                if final_path.is_file():
                    self._update_result_summary(str(final_path), notify=False)
            final_state = self._latest_snapshot
            if launch_error or (returncode != 0 and not self._stop_requested):
                final_state = replace(
                    final_state,
                    stage="失败",
                    detail=launch_error or message('采集进程退出码 {0}', returncode),
                    measurement_active=False,
                )
            elif self._stop_requested:
                final_state = replace(
                    final_state,
                    stage="已终止",
                    detail="用户请求终止；可使用相同输出目录续跑",
                    measurement_active=False,
                )
            self._latest_snapshot = final_state
            self._render_snapshot(final_state)
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
            if launch_error or (returncode != 0 and not self._stop_requested):
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
        self._sync_image_refresh_timer()
        if kind == "stats":
            report_path, self._stats_report_path = self._stats_report_path, None
            self._stats_report_reused = False
            if returncode == 0 and not launch_error and report_path is not None:
                self._open_report(str(report_path))
            else:
                self._activate_tab("reports-tab")
                self._clear_report(message("统计未完成，请查看“运行监控”中的错误或终止日志。"))

    @on(Button.Pressed, "#stop-run")
    def stop_button(self) -> None:
        self.action_request_stop()

    def action_request_stop(self) -> None:
        if self._image_operation:
            self.notify("镜像操作尚未完成，请稍候", severity="warning")
            return
        if not self._is_busy():
            self.notify("当前没有运行中的任务", severity="warning")
            return
        self.push_screen(
            ConfirmActionScreen(
                "终止当前任务？",
                "将先向整个采集进程组发送 SIGINT，允许容器和监控器清理；"
                "超时后才会升级为 SIGTERM。已写入的 case 结果不会删除。",
                "终止任务",
                variant="error",
            ),
            self._confirmed_stop,
        )

    def _confirmed_stop(self, confirmed: bool | None) -> None:
        if not confirmed:
            return
        self._stop_requested = True
        self._stop_process_gracefully()

    @work(thread=True, group="stop", exclusive=True, exit_on_error=False)
    def _stop_process_gracefully(self) -> None:
        self._safe_process_callback(self._write_log, "[TUI] 正在请求采集进程安全停止……")
        result = self._lifecycle.stop()
        if not result.complete:
            self._safe_process_callback(self._process_cleanup_incomplete, result)

    @on(Button.Pressed, "#quick-check")
    def quick_check_button(self) -> None:
        self.action_quick_check()

    def _network_preflight_report(self, line: str) -> None:
        import json
        report = json.loads(line.split(" ", 1)[1])
        from acprof.host.network_preflight import format_summary
        self.query_one("#network-download-summary", Static).update(format_summary(report))

    def _network_download_report(self, line: str) -> None:
        import json
        report = json.loads(line.split(" ", 1)[1])
        summary = self.query_one("#network-download-summary", Static)
        summary.update(f"{summary.content}\n{report['category']}: "
            f"verified_new_payload_bytes={report['verified_new_payload_bytes']:,}; "
            f"cache_savings_bytes={report['cache_savings_bytes']:,}; wire_bytes=unknown")

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

    def _preparation_event(self, event: dict) -> None:
        from acprof.tui.preparation import PreparationScreen, phase_summary
        request = event.get("request")
        if not isinstance(request, dict) or not request or self._stop_requested:
            return
        if self._latest_snapshot.measurement_active or self._preparation_request is not None:
            raise RuntimeError("unexpected preparation request")
        process = self._lifecycle.process
        if process is None or process.poll() is not None:
            return
        if request["kind"] == "error":
            self._write_log(f"[preparation][ERROR] {event['stage']}: {request.get('detail', '')}")
        self._preparation_request = (process, request["id"])
        self._preparation_screen = PreparationScreen(event, phase_summary(self._latest_snapshot))
        self.push_screen(self._preparation_screen, self._preparation_answered)

    def _preparation_answered(self, result) -> None:
        from acprof.preparation_events import encode_reply
        pending = self._preparation_request
        self._preparation_request = None
        self._preparation_screen = None
        if pending is None:
            return
        process, request_id = pending
        result = result or {"action": "cancel"}
        if result["action"] == "cancel":
            self._stop_requested = True
        try:
            self._lifecycle.reply(process, encode_reply(request_id, **result))
        except (OSError, ValueError, RuntimeError) as exc:
            self.notify(str(exc), severity="error")
            self._stop_requested = True
            self._stop_process_gracefully()

    @on(Button.Pressed, "#inspect-model")
    def inspect_model(self) -> None:
        if self._is_busy() or self._check_running or self._latest_snapshot.measurement_active:
            return
        config = RunConfig(model=self._input("model"), model_spec=self._input("model-spec"),
                           task=self._input("task"), backend=self._input("backend"),
                           output_dir=self._input("output-dir") or "results")
        if not config.model:
            self.notify("请先填写模型 ID", severity="warning")
            return
        from acprof.tui.model_resolution import ModelResolutionScreen
        task = None
        previous = self._last_resolution
        if (previous and previous["task"].model_id == config.model and str(previous["spec"]) == config.model_spec
                and previous.get("selection") == {"task": config.task, "backend": config.backend}):
            import json
            task = previous["task"]
            report_path = previous["output"] / "model_resolution.json"
            try:
                resolution = json.loads(report_path.read_text())
                if resolution.get("contract", {}).get("revision") == task.model_revision:
                    task.model_resolution = resolution
            except (OSError, ValueError):
                pass
        self._resolution_open = True
        self._set_busy(True)
        self.push_screen(ModelResolutionScreen(config, task=task), self._resolution_closed)

    def _resolution_closed(self, result) -> None:
        self._resolution_open = False
        self._set_busy(False)
        if not result:
            return
        self._last_resolution = result
        self.query_one("#model-spec", Input).value = str(result["spec"])
        if result["probe"]:
            from acprof.installation import cli_command
            command = [*cli_command("inspect", python_executable=PYTHON_EXECUTABLE), result["task"].model_id,
                       "--model-spec", str(result["spec"]), "--expected-revision", result["task"].model_revision,
                       "--probe", result["probe"], "--output-dir", str(result["output"]), "--skip-build"]
            self._launch(PendingLaunch(tuple(command), "inspect"))

    def action_quick_check(self) -> None:
        if self._is_busy() or self._check_running:
            self.notify("请等待当前任务完成", severity="warning")
            return
        # Host diagnostics do not require a model ID or a complete resource
        # matrix, so they remain usable on a freshly configured machine.
        config = RunConfig(
            model="preflight-only",
            profiling_mode=self._select("profiling-mode"),
            gpus=self._select("gpus"),
            sniff_iface=self._input("sniff-iface"),
        )
        self._check_running = True
        self._sync_image_refresh_timer()
        # Disabling a focused button first moves focus to another control in
        # the old pane, which queues a request to reactivate that pane.
        self._activate_tab("monitor-tab")
        self.query_one("#quick-check", Button).disabled = True
        self.query_one("#run-log", SelectableLog).write(self.tr("[TUI] 开始只读快速环境检查……"))
        self._execute_quick_check(config)

    @work(thread=True, group="preflight", exclusive=True, exit_on_error=False)
    def _execute_quick_check(self, config: RunConfig) -> None:
        try:
            checks = quick_preflight(config, project_dir=PROJECT_DIR)
            error = ""
        except Exception as exc:
            checks = []
            error = f"{type(exc).__name__}: {exc}"
        self.call_from_thread(self._show_quick_check, checks, error)

    def _show_quick_check(
        self,
        checks: Sequence[PreflightCheck],
        error: str,
    ) -> None:
        self._check_running = False
        self._sync_image_refresh_timer()
        if not self._is_busy():
            self.query_one("#quick-check", Button).disabled = False
        log = self.query_one("#run-log", SelectableLog)
        if error:
            log.write(self.tr(message('[TUI][ERROR] 环境检查失败：{0}', error)))
            self.notify(error, severity="error")
            return
        # SelectableLog retains plain text for wrapping, selection, and copying;
        # Rich renderables cannot be written to its TextArea document.
        log.write(self.tr("[TUI] 环境检查结果："))
        status_label = {"ok": "通过", "warn": "警告", "fail": "失败"}
        for check in checks:
            log.write(self.tr(message(
                "[{0}] {1}: {2}",
                message(status_label.get(check.status, check.status)), check.label, check.detail,
            )))
        failures = sum(check.status == "fail" for check in checks)
        warnings = sum(check.status == "warn" for check in checks)
        log.write(self.tr(message(
            "[TUI] 快速检查完成：{0} 通过，{1} 警告，{2} 失败。"
            "run.py 启动时仍会执行权威预检。",
            len(checks) - failures - warnings, warnings, failures,
        )))
        severity = "error" if failures else ("warning" if warnings else "information")
        self.notify(
            message('环境检查：{0} 失败，{1} 警告', failures, warnings),
            severity=severity,
            timeout=6,
        )

    @on(Button.Pressed, "#clear-log")
    def clear_log_button(self) -> None:
        self.action_clear_log()

    def action_clear_log(self) -> None:
        self.query_one("#run-log", SelectableLog).clear()

    @on(Button.Pressed, "#summarize-results")
    def summarize_results_button(self) -> None:
        if self._is_busy():
            self.notify("请等待当前任务完成", severity="warning")
            return
        self._update_result_summary(self._input("result-csv"))

    def _update_result_summary(self, result_csv: str, *, notify: bool = True) -> None:
        if not result_csv:
            if notify:
                self.notify("请填写结果 CSV 路径", severity="warning")
            return
        csv_path = resolve_result_path(result_csv, PROJECT_DIR)
        try:
            summary = summarize_result_csv(csv_path)
        except (OSError, csv.Error, UnicodeError, ValueError) as exc:
            self._set_text(self.query_one('#result-summary', Static), message('无法读取结果：{0}', exc))
            if notify:
                self.notify(str(exc), severity="error")
            return
        self._remember_last_used(result_csv=str(csv_path))
        from acprof.tui.diagnostics import result_summary_text
        self._set_text(self.query_one("#result-summary", Static), result_summary_text(summary, csv_path))
        if notify:
            self.notify("结果摘要已更新", timeout=3)

    @on(Button.Pressed, "#plot-results")
    def plot_results_button(self) -> None:
        self._launch_plot()

    def _launch_plot(self, path: str | None = None) -> None:
        if self._is_busy():
            self.notify("已有任务正在运行", severity="warning")
            return
        result_csv = path or self._input("result-csv")
        if not result_csv:
            self.notify("请填写结果 CSV 路径", severity="warning")
            return
        try:
            pending = prepare_plot(result_csv, project_dir=PROJECT_DIR, python_executable=PYTHON_EXECUTABLE)
        except RunConfigError as exc:
            self.notify(str(exc), severity="error")
            return
        self._launch(pending)

    def _clear_report(self, status: str) -> None:
        self._report_view = None
        self.query_one("#report-table", DataTable).clear(columns=True)
        self._set_text(self.query_one("#report-status", Static), status)
        self._set_text(self.query_one("#report-detail", Static), "表格可滚动；选择一行查看口径与数据来源。")

    @on(Input.Changed, "#report-source")
    def report_source_changed(self) -> None:
        if self._form_ready and not self._is_busy():
            self._clear_report("CSV / 目录：计算统计；JSON：查看报告。采集结束后操作。")

    @on(Button.Pressed, "#report-open")
    def open_report_button(self) -> None:
        self._open_report()

    def _open_report(self, path: str | None = None) -> None:
        if self._is_busy() or self._check_running or self._latest_snapshot.measurement_active:
            self.notify("请等待当前任务完成", severity="warning")
            return
        self._activate_tab("reports-tab")
        source = path if path is not None else self._input("report-source")
        if not source.strip():
            self._clear_report("请填写报告 JSON 路径，或使用当前结果计算统计。")
            return
        report_path = resolve_result_path(source, PROJECT_DIR)
        if report_path.suffix.lower() != ".json":
            self._clear_report("请选择 JSON 报告；实验目录或 CSV 请点击“计算统计”。")
            return
        with self.prevent(Input.Changed):
            self.query_one("#report-source", Input).value = str(report_path)
        self._clear_report(message("{0} 正在读取报告", CALCULATING))
        self.screen.set_focus(self.query_one("#report-table"), scroll_visible=False)
        self._report_loading = True
        self._set_busy(True)
        self._execute_report_read(report_path)

    @work(thread=True, group="report", exclusive=True, exit_on_error=False)
    def _execute_report_read(self, path: Path) -> None:
        try:
            view, error = read_report(path), ""
        except Exception as exc:
            view, error = None, error_message(exc)
        self.call_from_thread(self._show_report, view, error)

    def _show_report(self, view: ReportView | None, error: str) -> None:
        self._report_loading = False
        self._set_busy(self._is_busy())
        if view is None:
            self._clear_report(message("报告读取失败：{0}", error))
            return
        self._report_view = view
        self._render_report_view()

    def _render_report_view(self) -> None:
        view = self._report_view
        if view is None:
            return
        table = self.query_one("#report-table", DataTable)
        cursor, scroll = table.cursor_coordinate, table.scroll_offset
        table.clear(columns=True)
        for column in view.columns:
            table.add_column(Text(self.tr(column)), key=str(column))
        for index, row in enumerate(view.rows):
            table.add_row(*(Text(self.tr(cell)) for cell in row.cells), key=str(index))
        table.move_cursor(row=min(cursor.row, max(0, len(view.rows) - 1)), column=cursor.column, scroll=False)
        table.scroll_to(scroll.x, scroll.y, animate=False, force=True)
        self._set_text(self.query_one("#report-status", Static), view.title)
        self._show_report_row(table.cursor_row)

    @on(DataTable.RowHighlighted, "#report-table")
    def report_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._show_report_row(event.cursor_row)

    def _show_report_row(self, index: int) -> None:
        view = self._report_view
        if view is None:
            return
        detail = view.rows[index].detail if 0 <= index < len(view.rows) else message("没有可统计的正式成功窗口。")
        self._set_text(self.query_one("#report-detail", Static), join_messages("\n", (detail, view.note)))

    @on(Button.Pressed, "#report-calculate")
    def calculate_report_button(self) -> None:
        self._launch_stats()

    def _launch_stats(self, path: str | None = None) -> None:
        if self._is_busy() or self._check_running or self._latest_snapshot.measurement_active:
            self.notify("请等待当前任务完成", severity="warning")
            return
        self._activate_tab("reports-tab")
        source = path if path is not None else self._input("report-source")
        if not source.strip():
            self._clear_report("请填写实验目录或结果 CSV 路径。")
            return
        try:
            pending = prepare_stats(source, project_dir=PROJECT_DIR, python_executable=PYTHON_EXECUTABLE)
        except (OSError, ValueError) as exc:
            self._clear_report(str(exc))
            return
        self._stats_report_path = None
        self._stats_report_reused = False
        self._clear_report(message("{0} 正在计算窗口统计，完成后自动显示报告。", CALCULATING))
        self._launch(pending)

    @on(Button.Pressed, "#profile-dry-run")
    def profile_dry_run_button(self) -> None:
        self._launch_profile(dry_run=True)

    @on(Button.Pressed, "#profile-run")
    def profile_run_button(self) -> None:
        self._request_profile_run()

    def _profile_command(
        self,
        *,
        dry_run: bool,
        result_dir: str | None = None,
        tools: str | None = None,
    ) -> tuple[list[str], Path] | None:
        directory = result_dir or self._input("result-dir")
        selected_tools = tools if tools is not None else ",".join(
            checkbox.name
            for checkbox in self.query("#profile-tools Checkbox")
            if checkbox.value and checkbox.name is not None
        )
        if not directory:
            self.notify("请填写结果目录", severity="warning")
            return None
        if not selected_tools:
            self.notify("请至少勾选一个补采工具", severity="warning")
            return None
        try:
            pending = prepare_profile(directory, tools=selected_tools, dry_run=dry_run,
                                      project_dir=PROJECT_DIR, python_executable=PYTHON_EXECUTABLE)
            return list(pending.command), Path(pending.result_dir)
        except FileNotFoundError as exc:
            self.notify(str(exc), severity="error")
            return None
        except RunConfigError as exc:
            self._show_config_error(exc)
            return None

    def _launch_profile(
        self,
        *,
        dry_run: bool,
        result_dir: str | None = None,
        tools: str | None = None,
    ) -> None:
        if self._is_busy():
            self.notify("已有任务正在运行", severity="warning")
            return
        prepared = self._profile_command(
            dry_run=dry_run,
            result_dir=result_dir,
            tools=tools,
        )
        if prepared is not None:
            command, result_path = prepared
            self._launch(PendingLaunch(
                tuple(command), "profile-dry-run" if dry_run else "profile", result_dir=str(result_path),
            ))

    def _request_profile_run(
        self,
        result_dir: str | None = None,
        tools: str | None = None,
    ) -> None:
        if self._is_busy():
            self.notify("已有任务正在运行", severity="warning")
            return
        prepared = self._profile_command(
            dry_run=False,
            result_dir=result_dir,
            tools=tools,
        )
        if prepared is None:
            return
        command, result_path = prepared
        self._pending_launch = PendingLaunch(tuple(command), "profile", result_dir=str(result_path))
        self.push_screen(
            ConfirmActionScreen(
                "执行 profiler 补采？",
                join_messages("", (
                    message(
                        "该操作会启动隔离 profiler，并在成功后原子回填现有结果。"
                        "原文件会按项目规则备份。\n\n"
                    ),
                    format_command(command, project_dir=PROJECT_DIR),
                )),
                "执行补采",
                variant="warning",
            ),
            self._confirmed_launch,
        )

    @on(Input.Submitted, "#slash-command")
    def slash_command_submitted(self, event: Input.Submitted) -> None:
        value = event.value
        event.input.value = ""
        try:
            command, args = parse_slash_command(value)
        except RunConfigError as exc:
            self._show_config_error(exc)
            return

        if command == "run":
            self.action_request_run()
        elif command == "probe":
            self.action_request_probe()
        elif command == "check":
            self.action_quick_check()
        elif command in {"stop", "cancel"}:
            self.action_request_stop()
        elif command == "status":
            snapshot = self._latest_snapshot
            self.query_one("#run-log", SelectableLog).write(
                f"[TUI] status={self.tr(snapshot.stage)}; "
                f"case={snapshot.completed_cases}/{snapshot.total_cases}; "
                f"resource=CPU {snapshot.cpu}, MEM {snapshot.mem}GB, GPU {snapshot.gpu}; "
                f"warnings={snapshot.warnings}; errors={snapshot.errors}"
            )
            self._activate_tab("monitor-tab")
        elif command == "smoke":
            self.preset_smoke()
        elif command == "main":
            self.preset_main()
        elif command in {"defaults", "default"}:
            self.preset_default()
        elif command == "preview":
            self._refresh_command_preview()
            self._activate_tab("run-tab")
        elif command == "plot":
            self._launch_plot(args[0] if args else None)
        elif command == "stats":
            self._launch_stats(args[0] if args else None)
        elif command == "report":
            self._open_report(args[0] if args else None)
        elif command == "images":
            self.action_show_images()
        elif command == "profile":
            self._launch_profile(
                dry_run=True,
                result_dir=args[0] if args else None,
                tools=args[1] if len(args) > 1 else None,
            )
        elif command in {"profile-run", "profile!"}:
            self._request_profile_run(
                result_dir=args[0] if args else None,
                tools=args[1] if len(args) > 1 else None,
            )
        elif command in {"results", "summary"}:
            if self._is_busy():
                self.notify("请等待当前任务完成", severity="warning")
                return
            path = args[0] if args else self._input("result-csv")
            if args:
                self.query_one("#result-csv", Input).value = path
            self._update_result_summary(path)
            self._activate_tab("plot-tab")
        elif command in {"log", "logs"}:
            self.action_toggle_log_view()
        elif command == "clear":
            self.action_clear_log()
        elif command == "settings":
            self.action_show_settings()
        elif command == "help":
            self.query_one("#run-log", SelectableLog).write(
                self.tr("[TUI] /run 采集 · /probe 最大输入探测 · /check 环境检查 · "
                "/status 状态 · /stop 终止 · "
                "/smoke 最小预设 · /main 主矩阵 · /defaults 默认 · /preview 命令预览 · "
                "/plot [csv] 绘图 · /profile [dir] [tools] 补采计划 · "
                "/profile-run [dir] [tools] 执行补采 · /results [csv] 摘要 · "
                "/stats [csv/dir] 统计 · /report [json] 报告 · /images 镜像管理 · "
                "/settings 应用设置 · /log 放大日志 · /clear 清日志 · /quit 退出")
            )
            self._activate_tab("monitor-tab")
        elif command in {"quit", "exit"}:
            self.action_request_quit()
        else:
            self.notify(message('未知快捷命令：/{0}', command), severity="error")

    @on(Button.Pressed, "#quit-app")
    def action_request_quit(self) -> None:
        if self._image_operation:
            self.notify("镜像操作尚未完成，请稍候", severity="warning")
            return
        if self._is_busy():
            self.notify("任务仍在运行，请先使用 /stop 安全终止", severity="warning", timeout=6)
            return
        self.exit()

    async def on_unmount(self) -> None:
        """Use the same bounded cleanup policy even after widgets are gone."""
        self._form_ready = False
        self._cancel_preview_timer()
        if self._elapsed_timer is not None:
            self._elapsed_timer.stop()
            self._elapsed_timer = None
        if self._image_refresh_timer is not None:
            self._image_refresh_timer.stop()
        result = await asyncio.to_thread(self._lifecycle.stop, closing=True)
        if not result.complete:
            print(f"[TUI] cleanup incomplete: PID={result.pid}; {result.error}", file=sys.stderr)
