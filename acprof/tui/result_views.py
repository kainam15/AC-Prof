"""One experiment selection with overview, statistics, comparison and exports."""
from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Collapsible,
    ContentSwitcher,
    Label,
    Static,
    TabbedContent,
    TabPane,
)

from acprof.tui.i18n import message_id
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.table import ResizableDataTable

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


def compose_results_tab(app: AcprofTui) -> ComposeResult:
    local = app.localization.localized_widget
    with TabPane("实验结果", id="results-tab"):
        with Vertical(id="result-source-header", classes="page-header"):
            yield local(Static("当前实验：未选择", id="report-experiment", markup=False))
            with Horizontal(classes="path-picker-row"):
                yield local(Input(app.configuration.saved_settings.last_result_csv, id="result-csv",
                                  placeholder=message_id("results.input.source_hint")))
                yield local(Button("选择实验", name="result-csv", id="choose-result-csv", classes="experiment-picker"))
        with TabbedContent(initial="overview-view", id="result-views"):
            with TabPane(message_id("results.tabs.overview"), id="overview-view"):
                with VerticalScroll(id="plot-body", classes="pane-scroll"):
                    yield local(Static("选择或完成一次实验后，这里会显示结果摘要。", id="result-summary", markup=False))
                    with local(Collapsible(title="性能深度分析", collapsed=True, id="profile-section")):
                        yield local(Static(message_id("profiling.overview.select_experiment"), id="profile-status", markup=False))
                        yield local(Button(message_id("profiling.overview.open"), id="open-result-profiling", disabled=True))
            with TabPane(message_id("results.tabs.statistics"), id="statistics-view"):
                yield from compose_statistics(app)
            with TabPane(message_id("results.tabs.compare"), id="comparison-view"):
                yield from compose_comparison(app)
            with TabPane(message_id("results.tabs.import_export"), id="exports-view"):
                yield from compose_exports(app)
        with Horizontal(id="result-actions", classes="action-bar"):
            with Horizontal(classes="action-secondary"):
                yield local(Button("读取摘要", id="summarize-results"))
                yield local(Button(message_id("results.statistics.cancel"), id="results-cancel", disabled=True))
            with Horizontal(classes="action-primary"):
                yield local(Button("计算统计", id="report-calculate", classes="report-control", variant="primary"))
                yield local(Button("生成 PNG 图表", id="plot-results"))


def compose_statistics(app: AcprofTui) -> ComposeResult:
    local = app.localization.localized_widget
    with VerticalScroll(id="report-panel"):
        yield local(Static(message_id("results.statistics.empty_hint"), id="report-status", classes="page-summary", markup=False))
        with ContentSwitcher(initial="report-empty", id="report-content"):
            yield local(Static(message_id("results.statistics.empty_hint"), id="report-empty", markup=False))
            yield local(Static("正在读取统计报告…", id="report-loading", markup=False))
            yield local(Static("统计报告不可用", id="report-error", markup=False))
            with Vertical(id="report-results"):
                yield local(ResizableDataTable(id="report-table", cursor_type="row", zebra_stripes=True, fixed_columns=1))
                yield local(Static("表格可滚动；选择一行查看口径与数据来源。", id="report-detail", markup=False))


def compose_comparison(app: AcprofTui) -> ComposeResult:
    local = app.localization.localized_widget
    with VerticalScroll(classes="pane-scroll", id="comparison-options"):
        with Grid(classes="form-grid tool-form-grid"):
            for side, label in (("left", "左侧实验"), ("right", "右侧实验")):
                yield local(Label(label))
                with Horizontal(classes="path-picker-row"):
                    yield local(Input(id=f"comparison-{side}", classes="report-control",
                                      placeholder="实验目录；同组多个目录用分号分隔"))
                    yield local(Button("选择实验", name=f"comparison-{side}", id=f"choose-comparison-{side}", classes="experiment-picker"))
            yield local(Label("基线"))
            yield app.localization.localized_select((("左侧实验", "left"), ("右侧实验", "right")),
                value="left", allow_blank=False, id="comparison-baseline", classes="report-control")
            yield local(Label("比较用途"))
            yield app.localization.localized_select((("相同硬件", "same-hardware"), ("跨硬件", "cross-hardware"), ("资源配置扩展", "resource-scaling")),
                value="same-hardware", allow_blank=False, id="comparison-purpose", classes="report-control")
        yield local(Button("比较实验", id="report-compare", classes="report-control"))


def compose_exports(app: AcprofTui) -> ComposeResult:
    local = app.localization.localized_widget
    with VerticalScroll(classes="pane-scroll", id="result-export-panel"):
        yield local(Label(message_id("results.export.path_label")))
        yield local(Input(id="analysis-export-path", classes="report-control",
                          placeholder=message_id("results.export.path_hint")))
        with Horizontal(classes="html-actions"):
            yield local(Button(message_id("results.export.csv"), id="analysis-export-csv", name="csv", classes="report-control"))
            yield local(Button(message_id("results.export.json"), id="analysis-export-json", name="json", classes="report-control"))
        yield Static("", id="analysis-export-status", markup=False)
        with local(Collapsible(title="已有 JSON 报告", collapsed=True, id="report-file-options")):
            with Horizontal(classes="path-picker-row"):
                yield local(Input(id="report-json-path", classes="report-control", placeholder="JSON 文件路径"))
                yield local(Button("查看 JSON", id="report-open", classes="report-control"))
        with local(Collapsible(title="交互式 HTML 报告", collapsed=False, id="html-report-panel")):
            with Horizontal(classes="html-actions"):
                yield local(Button("生成 HTML", id="html-report-generate", classes="report-control"))
                yield local(Button("打开 HTML", id="html-report-open", classes="report-control"))
                yield local(Button("取消生成", id="html-report-cancel", disabled=True))
            yield local(Input(app.configuration.saved_settings.last_html_report, id="html-report-path",
                              classes="report-control", placeholder="已有 HTML 路径；生成成功后自动填入"))
            yield local(Static("每次生成新文件，保留旧报告。HTML 包含 quality 限制、统计口径与来源。",
                               id="html-report-status", markup=False))
            with local(Collapsible(title=message_id("results.html.additional_inputs_label"), collapsed=True)):
                yield local(Input(id="html-extra-sources", classes="report-control",
                                  placeholder=message_id("results.html.additional_inputs_hint")))
            with local(Collapsible(title="导出 / 远程获取", collapsed=True, id="html-export-options")):
                with Horizontal(classes="path-picker-row"):
                    yield local(Input(id="html-export-path", classes="report-control",
                                      placeholder="导出到新 .html 文件（目录须已存在）"))
                    yield local(Button("导出 HTML", id="html-report-export", classes="report-control"))
                yield local(Static("SSH / 无桌面：用 SFTP 或远程 IDE 下载上方 HTML 到本机，再用浏览器打开；无需 Web 服务。", markup=False))
