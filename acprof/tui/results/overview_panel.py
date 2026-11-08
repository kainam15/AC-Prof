"""Mounted overview widgets; source reads and action gates stay in ResultController."""
from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Collapsible, ProgressBar, Static

from acprof.tui.i18n import message_id
from acprof.tui.results.overview import OverviewView
from acprof.tui.ui.presentation import COMPACT_COLLAPSED_SYMBOL, COMPACT_EXPANDED_SYMBOL
from acprof.tui.ui.structured import Callout, FactGrid, StatusBanner, StructuredWidget
from acprof.tui.ui.view_models import CalloutContent, StatusSummary, ViewSection


class ExperimentOverviewPanel(StructuredWidget):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.view: OverviewView | None = None

    def compose(self) -> ComposeResult:
        yield StatusBanner(StatusSummary("neutral", message_id("results.summary.empty")),
                           id="overview-banner", headline_id="result-summary")
        with Horizontal(id="overview-case-progress"):
            yield Static("", id="overview-case-label", markup=False)
            # A known initial total and no ETA keep the native Bar out of indeterminate mode.
            yield ProgressBar(total=1, show_eta=False, show_percentage=False, id="overview-case-bar")
        yield Callout(CalloutContent("neutral", "", "", ""), id="overview-callout")
        with Vertical(id="overview-facts"):
            for index in range(4):
                yield FactGrid(ViewSection("", ()), id=f"overview-section-{index}")
        with Collapsible(title="", collapsed=True, id="overview-check-details",
                         collapsed_symbol=COMPACT_COLLAPSED_SYMBOL,
                         expanded_symbol=COMPACT_EXPANDED_SYMBOL):
            yield Static("", id="result-data-status", markup=False)
            yield Static("", id="result-profile-status", markup=False)
            yield Button("", id="overview-profiling", disabled=True)
            yield Static("", id="result-html-status", markup=False)

    def on_mount(self) -> None:
        self.show_hint(message_id("results.summary.empty"))
        self.refresh_language()

    def show_hint(self, source: str, *, level: str = "neutral") -> None:
        self.view = None
        self.query_one(StatusBanner).update_summary(StatusSummary(level, source))
        for selector in ("#overview-case-progress", "#overview-facts", "#overview-callout"):
            self.query_one(selector).display = False

    def update_view(self, view: OverviewView) -> None:
        self.view = view
        self.query_one(StatusBanner).update_summary(view.summary)
        progress = self.query_one("#overview-case-progress")
        progress.display = view.total_cases is not None
        if view.total_cases is not None:
            self.query_one(ProgressBar).update(total=view.total_cases, progress=view.completed_cases)
        callout = self.query_one(Callout)
        callout.display = view.callout is not None
        if view.callout is not None:
            callout.update_content(view.callout)
        self.query_one("#overview-facts").display = True
        for section, grid in zip(view.sections, self.query(FactGrid), strict=True):
            grid.update_section(section)
        self.refresh_language()

    def refresh_language(self) -> None:
        if self.view is not None and self.view.total_cases is not None:
            self.query_one("#overview-case-label", Static).update(Text(self.tr(message_id(
                "results.view.case_count", self.view.completed_cases, self.view.total_cases))))
        self.query_one("#overview-check-details", Collapsible).title = self.tr(
            message_id("results.view.check_details"))
        self.query_one("#overview-profiling", Button).label = Text(self.tr(
            message_id("profiling.overview.open")))

    def show_reason(self) -> None:
        callout = self.query_one(Callout)
        if callout.display:
            callout.scroll_visible(animate=False, top=True)
            callout.show_details()
        else:
            details = self.query_one("#overview-check-details", Collapsible)
            details.collapsed = False
            details.scroll_visible(animate=False)
