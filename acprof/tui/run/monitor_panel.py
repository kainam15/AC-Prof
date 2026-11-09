"""Event-driven monitor facts; the existing presenter owns the elapsed clock."""
from __future__ import annotations

from rich.console import RenderableType
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from acprof.tui.i18n import message_id
from acprof.tui.run.monitor_view import MonitorHeaderView, monitor_header_view
from acprof.tui.run.progress import ProgressSnapshot
from acprof.tui.ui.progress_widgets import ProgressSummary
from acprof.tui.ui.structured import FactGrid, StatusBanner, StructuredWidget
from acprof.tui.ui.view_models import ViewSection


class CaseProgress(ProgressSummary):
    """Render known processed-case counts without a timer, pulse or ETA."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.total: int | None = None
        self.completed = 0
        self.display = False

    def update_progress(self, *, total: int | None, completed: int) -> None:
        if (total, completed) == (self.total, self.completed):
            return
        self.total, self.completed = total, completed
        self.display = total is not None and total > 0
        self.refresh()

    def render(self) -> RenderableType:
        if self.total is None or self.total <= 0:
            return Text()
        # Rich's low-level bar is a renderable, with no Progress/Live scheduler.
        # A full bar remains active-colored: processed does not mean successful.
        return self.byte_bar(self.completed, self.total)


class MonitorHeader(StructuredWidget):
    """Apply pure view data only when an existing progress event changes it."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.view = monitor_header_view(ProgressSnapshot())

    @staticmethod
    def _grid(section: ViewSection | None, widget_id: str) -> FactGrid:
        grid = FactGrid(section or ViewSection("", ()), id=widget_id)
        grid.display = section is not None
        return grid

    def compose(self) -> ComposeResult:
        with Horizontal(id="monitor-status-line"):
            yield StatusBanner(self.view.summary, id="monitor-summary", headline_id="status-stage")
            with Horizontal(id="monitor-clock", classes="elapsed-status"):
                yield Static("", id="monitor-clock-label", markup=False)
                yield Static("", id="status-elapsed", markup=False)
        facts = Horizontal(id="monitor-facts")
        facts.display = self.view.cases is not None or self.view.resources is not None
        with facts:
            cases = Vertical(id="monitor-case-block")
            cases.display = self.view.cases is not None
            with cases:
                yield self._grid(self.view.cases, "monitor-cases")
                yield CaseProgress(id="case-progress")
            yield self._grid(self.view.resources, "monitor-resources")
        yield self._grid(self.view.checks, "status-preparation")
        yield self._grid(self.view.details, "monitor-details")

    def on_mount(self) -> None:
        self.refresh_language()

    def on_resize(self) -> None:
        self.set_class(self.size.width < 48, "compact-facts")

    def refresh_language(self) -> None:
        self.query_one("#monitor-clock-label", Static).update(self.tr(message_id("run.monitor.elapsed_label")))

    def update_view(self, view: MonitorHeaderView) -> None:
        if view == self.view:
            return
        previous, self.view = self.view, view
        if view.summary != previous.summary:
            self.query_one(StatusBanner).update_summary(view.summary)
        for widget_id, section in (
            ("monitor-cases", view.cases), ("monitor-resources", view.resources),
            ("status-preparation", view.checks), ("monitor-details", view.details),
        ):
            grid = self.query_one("#" + widget_id, FactGrid)
            grid.display = section is not None
            if section is not None and section != grid.section:
                grid.update_section(section)
        self.query_one("#monitor-facts").display = view.cases is not None or view.resources is not None
        self.query_one("#monitor-case-block").display = view.cases is not None
        self.query_one(CaseProgress).update_progress(total=view.progress_total, completed=view.progress_completed)
