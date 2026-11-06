"""Posthoc profiling of the experiment selected by the shared result controller."""
from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Label, Static, TabPane

from acprof.tui.i18n import message_id

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


def compose_backfill_tab(app: AcprofTui) -> ComposeResult:
    local = app.localization.localized_widget
    with TabPane(message_id("profiling.backfill.title"), id="backfill-tab"):
        with Vertical(classes="page-header"):
            with Horizontal(classes="path-picker-row"):
                yield local(Static(message_id("profiling.backfill.source_hint"), id="backfill-source", markup=False))
                yield local(Button("选择实验", name="result-csv", id="choose-backfill-result", classes="experiment-picker"))
        with VerticalScroll(id="backfill-body", classes="pane-scroll"):
            yield local(Label(message_id("profiling.overview.title"), classes="section-title"))
            yield local(Static(message_id("profiling.overview.select_experiment"), id="profile-status", markup=False))
            yield local(Static(message_id("profiling.backfill.isolation_note"), classes="page-hint", markup=False))
        with Horizontal(classes="action-bar"):
            with Horizontal(classes="action-secondary"):
                yield local(Button("读取摘要", id="summarize-backfill"))
                yield local(Button(message_id("results.statistics.cancel"), id="backfill-cancel", disabled=True))
            with Horizontal(classes="action-primary"):
                yield local(Button(message_id("profiling.overview.open"), id="open-result-profiling", disabled=True, variant="primary"))
