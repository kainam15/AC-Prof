"""AC-Prof TUI ui/pages/monitor.py components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical
from textual.widgets import Button, Static, TabPane

from acprof.tui.i18n import message_id
from acprof.tui.logging.widget import SelectableLog
from acprof.tui.ui.input import BarCursorInput as Input
from acprof.tui.ui.progress_widgets import DownloadProgressPanel, PreparationStatus

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui
from acprof.tui.logging.panel import LogPanel


def compose_monitor_tab(app: AcprofTui) -> ComposeResult:
    with TabPane("运行监控", id="monitor-tab"):
        with Vertical(id="monitor-panel"):
            with Grid(id="status-grid", classes="page-header"):
                yield app.localization.localized_widget(Static("阶段", classes="status-label"))
                yield app.localization.localized_widget(Static("等待", id="status-stage", markup=False))
                yield app.localization.localized_widget(Static("运行时间", classes="status-label elapsed-status"))
                yield app.localization.localized_widget(Static("", id="status-elapsed", classes="elapsed-status", markup=False))

                yield app.localization.localized_widget(Static("Case", classes="status-label case-status"))
                yield app.localization.localized_widget(Static("", id="status-case", classes="case-status", markup=False))
                yield app.localization.localized_widget(Static("资源", classes="status-label resource-status"))
                yield app.localization.localized_widget(Static("", id="status-resource", classes="resource-status", markup=False))

                yield app.localization.localized_widget(Static("警告 / 错误", classes="status-label error-count"))
                yield app.localization.localized_widget(Static("", id="status-errors", classes="error-count", markup=False))
                yield app.localization.localized_widget(Static("详情", classes="status-label detail-status"))
                yield app.localization.localized_widget(Static("尚未启动", id="status-detail", classes="detail-status", markup=False))

            yield PreparationStatus(id="status-preparation")
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
                        yield app.localization.localized_widget(Button("复制全部", id="copy-log-all", classes="log-tool log-control"))
                        yield app.localization.localized_widget(Button("回到底部", id="follow-log", classes="log-tool"))
                        yield app.localization.localized_widget(Button("放大日志", id="expand-log", classes="log-tool"))
                        yield app.localization.localized_widget(Button("返回监控", id="restore-log", classes="log-tool"))
                        yield app.localization.localized_widget(Button("清空日志", id="clear-log", classes="log-tool"))
                        yield app.localization.localized_widget(Button("工具", id="log-tools", classes="log-tool log-control", tooltip="筛选 / 搜索"))
                    with Horizontal(classes="action-primary"):
                        yield app.localization.localized_widget(Button(
                            "终止任务", id="stop-run", classes="log-tool",
                            variant="error", disabled=True,
                        ))
                yield SelectableLog(
                    id="run-log",
                    max_lines=app.configuration.preferences.log_max_lines,
                    wrap=app.configuration.preferences.log_wrap,
                    details=False,
                )
