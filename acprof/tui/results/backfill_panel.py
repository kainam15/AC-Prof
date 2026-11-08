"""Compose the backfill page from shared structured widgets; never read results."""
from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import Static

from acprof.tui.i18n import message_id
from acprof.tui.results.backfill_view import BackfillView, backfill_view
from acprof.tui.ui.structured import Callout, FactGrid, StatusBanner, StructuredWidget
from acprof.tui.ui.view_models import CalloutContent, ViewSection


class BackfillPanel(StructuredWidget):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.view = backfill_view(None)

    def compose(self) -> ComposeResult:
        yield StatusBanner(self.view.summary, id="backfill-banner", headline_id="profile-status")
        yield Callout(CalloutContent("neutral", "", "", ""), id="backfill-callout")
        for index in range(2):
            yield FactGrid(ViewSection("", ()), id=f"backfill-section-{index}")
        yield Static("", id="backfill-isolation-note", classes="page-hint", markup=False)

    def on_mount(self) -> None:
        self._render_view()

    def update_view(self, view: BackfillView) -> None:
        if view == self.view:
            return
        self.view = view
        if self.is_mounted:
            self._render_view()

    def _render_view(self) -> None:
        self.query_one(StatusBanner).update_summary(self.view.summary)
        callout = self.query_one(Callout)
        callout.display = self.view.callout is not None
        if self.view.callout is not None:
            callout.update_content(self.view.callout)
        for index, grid in enumerate(self.query(FactGrid)):
            grid.display = index < len(self.view.sections)
            if grid.display:
                grid.update_section(self.view.sections[index])
        self.query_one("#backfill-isolation-note").display = bool(self.view.sections)
        self.refresh_language()

    def refresh_language(self) -> None:
        self.query_one("#backfill-isolation-note", Static).update(
            Text(self.tr(message_id("profiling.backfill.isolation_note"))))
