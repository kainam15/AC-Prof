"""AC-Prof TUI ui/pages/monitor.py components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.geometry import Size
from textual.widgets import Button, Static, TabPane

from acprof.tui.i18n import message_id
from acprof.tui.logging.widget import SelectableLog
from acprof.tui.run.monitor_panel import MonitorHeader
from acprof.tui.ui.input import BarCursorInput as Input
from acprof.tui.ui.progress_widgets import DownloadProgressPanel, PhaseTimes

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui
from acprof.tui.logging.panel import LogPanel


class LogToolbarButton(Button):
    """Measure the label's styled padding instead of Button's fixed allowance."""

    def on_mount(self) -> None:
        # Textual 8.2.8 rejects zero line-pad in TCSS, but its Python style
        # property supports it. Keep the border while reclaiming label padding.
        self.styles.line_pad = 0

    def get_content_width(self, container: Size, viewport: Size) -> int:
        return Content.from_text(self.label).get_optimal_width(self.styles.get_rules(), container.width)


def compose_monitor_tab(app: AcprofTui) -> ComposeResult:
    with TabPane("运行监控", id="monitor-tab"):
        with Vertical(id="monitor-panel"):
            yield MonitorHeader(id="status-grid", classes="page-header")
            yield PhaseTimes("", id="status-phase-times", markup=False)
            yield DownloadProgressPanel(id="download-progress")
            with LogPanel(id="log-panel"):
                with Horizontal(id="log-controls"):
                    yield app.localization.localized_select(
                        [("ALL", "ALL"), ("INFO", "INFO"), ("WARN", "WARN"), ("ERROR", "ERROR"), ("DEBUG", "DEBUG")],
                        value="ALL", allow_blank=False, id="log-level", classes="log-control",
                    )
                    yield app.localization.localized_widget(Input(placeholder="搜索保留日志", id="log-search", classes="log-control"))
                    yield app.localization.localized_widget(Button("导出", id="export-log", classes="log-control", compact=True))
                yield Static("", id="log-status", markup=False)
                with Horizontal(id="log-export-result"):
                    yield app.localization.localized_widget(Static(message_id('logs.export.completed')))
                    yield app.localization.localized_widget(Button(message_id('logs.export.copy_path'),
                        id="copy-log-export-path", classes="log-control", compact=True))
                    yield app.localization.localized_widget(Button(message_id('logs.export.view_path'),
                        id="view-log-export-path", classes="log-control", compact=True))
                with Horizontal(id="log-toolbar", classes="action-bar"):
                    with Horizontal(classes="action-secondary"):
                        yield app.localization.localized_widget(LogToolbarButton("复制全部", id="copy-log-all", classes="log-tool log-control"))
                        yield app.localization.localized_widget(LogToolbarButton("回到底部", id="follow-log", classes="log-tool"))
                        yield app.localization.localized_widget(LogToolbarButton("放大日志", id="expand-log", classes="log-tool"))
                        yield app.localization.localized_widget(LogToolbarButton("返回监控", id="restore-log", classes="log-tool"))
                        yield app.localization.localized_widget(LogToolbarButton("清空日志", id="clear-log", classes="log-tool"))
                        yield app.localization.localized_widget(LogToolbarButton("工具", id="log-tools", classes="log-tool log-control", tooltip="筛选 / 搜索"))
                    with Horizontal(classes="action-primary"):
                        yield app.localization.localized_widget(LogToolbarButton(
                            "终止任务", id="stop-run", classes="log-tool",
                            variant="error", disabled=True,
                        ))
                yield SelectableLog(
                    id="run-log",
                    max_lines=app.configuration.preferences.log_max_lines,
                    wrap=app.configuration.preferences.log_wrap,
                    details=False,
                )
