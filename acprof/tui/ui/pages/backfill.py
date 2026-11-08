"""Posthoc profiling of the experiment selected by the shared result controller."""
from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, TabPane

from acprof.tui.i18n import message_id
from acprof.tui.results.backfill_panel import BackfillPanel
from acprof.tui.ui.input import BarCursorInput as Input

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


def compose_backfill_tab(app: AcprofTui) -> ComposeResult:
    local = app.localization.localized_widget
    with TabPane(message_id("profiling.backfill.title"), id="backfill-tab"):
        with Vertical(classes="page-header"):
            with Horizontal(classes="path-picker-row"):
                yield local(Input(app.configuration.saved_settings.last_result_csv, id="backfill-source",
                                  placeholder=message_id("results.input.source_hint")))
                yield local(Button("选择实验", name="experiment-source", id="choose-backfill-result", classes="experiment-picker"))
        with VerticalScroll(id="backfill-body", classes="pane-scroll"):
            yield BackfillPanel(id="backfill-panel")
        with Horizontal(classes="action-bar"):
            with Horizontal(classes="action-secondary"):
                yield local(Button(message_id("results.overview.refresh"), id="refresh-backfill-state"))
                yield local(Button(message_id("results.statistics.cancel"), id="backfill-cancel", disabled=True))
            with Horizontal(classes="action-primary"):
                yield local(Button(message_id("profiling.overview.open"), id="open-result-profiling", disabled=True, variant="primary"))
