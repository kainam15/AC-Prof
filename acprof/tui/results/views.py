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
from acprof.tui.ui.input import BarCursorInput as Input
from acprof.tui.ui.table import ResizableDataTable

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


def compose_results_tab(app: AcprofTui) -> ComposeResult:
    local = app.localization.localized_widget
    with TabPane("实验结果", id="results-tab"):
        with Vertical(id="result-source-header", classes="page-header"):
            with Horizontal(classes="path-picker-row"):
                yield local(Input(app.configuration.saved_settings.last_result_csv, id="result-csv",
                                  placeholder=message_id("results.input.source_hint")))
                yield local(Button("选择实验", name="result-csv", id="choose-result-csv", classes="experiment-picker"))
        with TabbedContent(initial="overview-view", id="result-views"):
            with TabPane(message_id("results.tabs.overview"), id="overview-view"):
                with VerticalScroll(id="plot-body", classes="pane-scroll"):
                    yield local(Static(message_id('results.summary.empty'), id='result-summary', markup=False))
                    with local(Collapsible(title="性能深度分析", collapsed=True, id="profile-section")):
                        yield local(Static(message_id("profiling.overview.select_experiment"), id="profile-status", markup=False))
                        yield local(Button(message_id("profiling.overview.open"), id="open-result-profiling", disabled=True))
            with TabPane(message_id("results.tabs.statistics"), id="statistics-view"):
                yield from compose_statistics(app)
            with TabPane(message_id("results.tabs.compare"), id="comparison-view"):
                yield from compose_comparison(app)
            with TabPane(message_id("results.tabs.import_export"), id="exports-view"):
                yield Static("", id="legacy-exports-placeholder")
        with Horizontal(id="result-actions", classes="action-bar"):
            with Horizontal(classes="action-secondary"):
                yield local(Button("读取摘要", id="summarize-results"))
                yield local(Button(message_id("results.statistics.cancel"), id="results-cancel", disabled=True))
            with Horizontal(classes="action-primary"):
                yield local(Button(message_id("results.analysis.generate"), id="report-calculate", classes="report-control", variant="primary"))
                yield local(Button("生成 PNG 图表", id="plot-results"))


def compose_statistics(app: AcprofTui) -> ComposeResult:
    local = app.localization.localized_widget
    with VerticalScroll(id="analysis-workflow", classes="pane-scroll"):
        yield from compose_html(app)
        with local(Collapsible(title=message_id("results.analysis.parameters"), collapsed=True, id="analysis-parameters")):
            with Grid(classes="form-grid tool-form-grid"):
                for name, label, hint in (
                    ("metrics", message_id("results.analysis.metrics"), "latency_app_s,throughput_samples_per_s"),
                    ("confidence", message_id("results.analysis.confidence"), "0.95"),
                    ("resamples", message_id("results.analysis.resamples"), "5000"),
                    ("seed", message_id("results.analysis.seed"), "0"),
                    ("block-size", message_id("results.analysis.block_size"), "1"),
                ):
                    yield local(Label(label))
                    yield Input(id="analysis-" + name, placeholder=hint, classes="report-control")
        with local(Collapsible(title=message_id("results.analysis.advanced"), collapsed=True, id="analysis-details")):
            yield local(Button(message_id("results.analysis.window_json"), id="report-window-stats", classes="report-control"))
            with Vertical(id="report-panel"):
                yield local(Static('', id='report-status', classes='page-summary', markup=False))
                with ContentSwitcher(initial="report-empty", id="report-content"):
                    yield local(Static(message_id('results.statistics.no_selection'), id='report-empty', markup=False))
                    yield local(Static("正在读取统计报告…", id="report-loading", markup=False))
                    yield local(Static("统计报告不可用", id="report-error", markup=False))
                    with Vertical(id="report-results"):
                        yield local(ResizableDataTable(id="report-table", cursor_type="row", zebra_stripes=True, fixed_columns=1))
                        yield local(Static("表格可滚动；选择一行查看口径与数据来源。", id="report-detail", markup=False))
            yield from compose_exports(app)


def compose_comparison(app: AcprofTui) -> ComposeResult:
    local = app.localization.localized_widget
    with VerticalScroll(classes="pane-scroll", id="comparison-options"):
        yield local(Static("", id="comparison-status", markup=False))
        yield local(Static(message_id("results.analysis.shared_parameters"), markup=False))
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
        yield local(Button(message_id("results.analysis.compare"), id="report-compare", classes="report-control"))


def compose_exports(app: AcprofTui) -> ComposeResult:
    local = app.localization.localized_widget
    with Vertical(id="result-export-panel"):
        with Vertical(id="analysis-export-controls"):
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


def compose_html(app: AcprofTui) -> ComposeResult:
    local = app.localization.localized_widget
    with Vertical(id="html-report-panel"):
        yield local(Label(message_id("results.html.title"), classes="html-section-title"))
        with Horizontal(id="html-source-row", classes="html-actions"):
            yield local(Static(message_id("results.html.current_source"), id="html-current-source"))
            yield local(Button(message_id("results.html.add_inputs"), id="html-extra-toggle", classes="report-control"))
        yield local(Input(id="html-extra-sources", classes="report-control",
                          placeholder=message_id("results.html.additional_inputs_hint")))
        with Horizontal(classes="html-actions"):
            yield local(Button(message_id("results.html.generate"), id="html-report-generate",
                               classes="report-control", variant="primary"))
            yield local(Button(message_id("results.html.cancel"), id="html-report-cancel", disabled=True))
        yield local(Static(message_id("results.html.generation_note"), id="html-report-status", markup=False))
        with Vertical(id="html-report-file"):
            yield local(Static(message_id("results.html.selected_report"), classes="html-file-heading"))
            yield Static("", id="html-report-filename", markup=False)
            yield Static("", id="html-report-location", markup=False)
            with Horizontal(classes="html-actions"):
                yield local(Button(message_id("results.html.open"), id="html-report-open", classes="report-control"))
                yield local(Button(message_id("results.html.copy_path"), id="html-report-copy", classes="report-control"))
                yield local(Button(message_id("results.html.save_as"), id="html-export-toggle", classes="report-control"))
        with Horizontal(id="html-export-fields", classes="path-picker-row"):
            yield local(Input(id="html-export-path", classes="report-control",
                              placeholder=message_id("results.html.export_path_hint")))
            yield local(Button(message_id("results.html.export"), id="html-report-export", classes="report-control"))
        yield local(Button(message_id("results.html.use_existing"), id="html-existing-toggle", classes="report-control"))
        yield local(Input(app.configuration.saved_settings.last_html_report, id="html-report-path",
                          classes="report-control", placeholder=message_id("results.html.path_hint")))
        yield local(Static(message_id("results.html.remote_download_hint"), id="html-remote-hint", markup=False))
