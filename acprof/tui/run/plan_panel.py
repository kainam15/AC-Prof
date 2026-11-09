"""Present the collection plan without owning probes, validation or scheduling."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll

from acprof.tui.run.planning import PlanView
from acprof.tui.ui.structured import Callout, FactGrid, StatusBanner, StructuredWidget
from acprof.tui.ui.view_models import CalloutContent, StatusSummary, ViewSection


class RunPlanPanel(StructuredWidget):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.view: PlanView | None = None

    def compose(self) -> ComposeResult:
        yield StatusBanner(StatusSummary("neutral", ""), id="run-plan-status")
        with VerticalScroll(id="run-plan-body"):
            yield FactGrid(ViewSection("", ()), id="run-plan-facts")
            callout = Callout(CalloutContent("neutral", "", "", ""), id="run-plan-callout")
            callout.display = False
            yield callout

    def show_plan(self, view: PlanView) -> None:
        if view == self.view:
            return
        self.view = view
        self.query_one(StatusBanner).update_summary(view.summary)
        facts = self.query_one(FactGrid)
        facts.update_section(view.section)
        facts.display = bool(view.section.rows or view.section.detail_rows)
        callout = self.query_one(Callout)
        callout.display = view.callout is not None
        if view.callout is not None:
            callout.update_content(view.callout)
