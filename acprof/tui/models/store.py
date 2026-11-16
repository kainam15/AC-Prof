"""On-demand storage review; never polls or prunes during measurement."""
from pathlib import Path
from threading import Event

from textual import on, work
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from acprof.host.model_store.store import ModelStoreCancelled, disk_report, prune_store
from acprof.network_policy import parse_bytes
from acprof.tui.i18n import message_id
from acprof.tui.models.store_summary import (
    model_store_facts,
    model_store_prune_summary,
    model_store_rows,
)
from acprof.tui.ui.contracts import translated_app
from acprof.tui.ui.dialogs import ScrollConfirmActionScreen, focus_horizontal_action
from acprof.tui.ui.rendering import CjkCompositor
from acprof.tui.ui.structured import FactGrid, StatusBanner
from acprof.tui.ui.table import ResizableDataTable
from acprof.tui.ui.view_models import StatusSummary, ViewSection


class ModelStoreScreen(ModalScreen):
    BINDINGS = [("escape", "close", "关闭")]
    DEFAULT_CSS = """
    ModelStoreScreen { align: center middle; }
    #store-dialog { width: 94%; height: 85%; border: round $accent; background: $surface; padding: 0 1; }
    #store-scroll { height: 1fr; }
    #store-banner, #store-facts, #store-status { height: auto; }
    #store-facts { margin: 0 0 1 0; }
    #store-table { height: auto; max-height: 16; margin: 0 0 1 0; }
    #store-note { height: auto; color: $text-muted; margin-bottom: 1; }
    #store-actions { height: 3; }
    """

    def __init__(self, root: Path, target: str = ""):
        super().__init__()
        self._compositor = CjkCompositor()
        self.root = root
        self.target = target
        self.preview = None
        self._report: dict = {}
        self._working = True
        self._close_requested = False
        self._cancel = Event()

    def compose(self):
        with Vertical(id="store-dialog"):
            yield Static(translated_app(self.app).tr(message_id('downloads.model_store.usage_title')), markup=False)
            with VerticalScroll(id="store-scroll"):
                yield StatusBanner(
                    StatusSummary("info", message_id("images.storage.loading")),
                    id="store-banner",
                    headline_id="store-status",
                    include_actions=False,
                )
                yield FactGrid(ViewSection("", ()), id="store-facts", collapsible=False)
                yield ResizableDataTable(id="store-table", zebra_stripes=True, cursor_type="row")
                yield Static("", id="store-note", classes="page-hint", markup=False)
            with Horizontal(id="store-actions", classes="action-bar"):
                yield Button(translated_app(self.app).tr(message_id('common.action.close_dialog')), id="store-close")
                yield Button(translated_app(self.app).tr(message_id('common.action.refresh')), id="store-refresh", disabled=True)
                yield Button(translated_app(self.app).tr(message_id('downloads.prune.unused_models')), id="store-prune", disabled=True)

    def on_mount(self):
        for button in self.query(Button):
            button.active_effect_duration = 0
        self.query_one("#store-facts", FactGrid).display = False
        self.query_one("#store-table", ResizableDataTable).display = False
        self.refresh_report()

    def _set_status(self, status, *, level="info"):
        self.query_one("#store-banner", StatusBanner).update_summary(StatusSummary(level, status))

    def on_unmount(self):
        self._cancel.set()

    def _post(self, callback, *args):
        if self.is_attached and not self._pruning:
            app = self.app
            try:
                app.call_from_thread(callback, *args)
            except RuntimeError:
                if app.is_running:
                    raise

    @work(thread=True, exclusive=True, group="model-store", exit_on_error=False)
    def refresh_report(self):
        try:
            report = disk_report(self.root, cancel=self._cancel)
            if self.target:
                report = {**report, "capacity_bytes": parse_bytes(self.target)}
            self._post(self._show_capacity, report)
            preview = prune_store(root=self.root, target_bytes=self.target, cancel=self._cancel,
                                  on_wait=lambda: self._post(self._show_waiting))
        except ModelStoreCancelled:
            self._post(self._complete, translated_app(self.app).tr(message_id('run.status.cancelled')))
        except (ValueError, OSError) as exc:
            self._post(self._complete, str(exc))
        else:
            self._post(self._complete, "", preview)

    def _show_capacity(self, report):
        if not self.is_attached or self._pruning:
            return
        self._report = report
        self._render_report()
        if not self._close_requested:
            self._set_status(message_id("downloads.prune.preparing_preview"), level="info")

    def _show_waiting(self):
        if self.is_attached and not self._pruning and not self._close_requested:
            self._set_status(message_id("downloads.model_store.busy"), level="warning")

    def _complete(self, error="", preview=None):
        if not self.is_attached or self._pruning:
            return
        self._working = False
        if self._close_requested:
            self.dismiss()
            return
        self.preview = preview
        self.query_one("#store-refresh", Button).disabled = False
        if error:
            self._set_status(error, level="error")
        else:
            self._set_status(message_id("downloads.store.status_ready"), level="success")
        self._render_report()
        self.query_one("#store-prune", Button).disabled = not (
            preview and (preview["entries"] or preview["reclaimable_bytes"]))

    def _render_report(self):
        tr = translated_app(self.app).tr
        facts = self.query_one("#store-facts", FactGrid)
        facts.display = bool(self._report)
        facts.update_section(model_store_facts(self._report, self.preview))
        table = self.query_one("#store-table", ResizableDataTable)
        rows = model_store_rows(self._report, self.preview, tr)
        table.display = bool(rows)
        table.clear(columns=True)
        if rows:
            for column_label in (
                message_id("models.input.id_label"),
                message_id("downloads.store.col_revision"),
                message_id("downloads.store.col_size"),
                message_id("downloads.store.col_last_used"),
                message_id("downloads.store.col_preview"),
            ):
                table.add_column(tr(column_label))
            for row in rows:
                table.add_row(*row)
        note = self.query_one("#store-note", Static)
        if not self._report:
            note.update("")
            note.display = False
        else:
            notes = [tr(message_id("downloads.store.shared_weights_note"
                                   if self._report["models"] else "downloads.store.no_registered_models"))]
            if self.preview is not None:
                if self.preview["entries"] or self.preview["reclaimable_bytes"]:
                    notes.append(tr(message_id("downloads.store.prune_revalidation_note")))
                else:
                    notes.append(tr(message_id("downloads.store.nothing_to_prune")))
            note.update("\n".join(notes))
            note.display = True

    @on(Button.Pressed, "#store-refresh")
    def refresh_clicked(self):
        if self._working:
            return
        self._begin_work()
        self.refresh_report()

    def _begin_work(self, status=None):
        self._working = True
        self._cancel = Event()
        self.preview = None
        self._report = {}
        self._render_report()
        for key in ("store-refresh", "store-prune"):
            self.query_one(f"#{key}", Button).disabled = True
        self._set_status(status if status is not None else message_id("downloads.prune.preparing_preview"), level="info")

    @on(Button.Pressed, "#store-prune")
    def prune_clicked(self):
        if self._working or not self.preview:
            return
        tr = translated_app(self.app).tr
        self.app.push_screen(ScrollConfirmActionScreen(
            tr(message_id('downloads.prune.unused_models')), model_store_prune_summary(self._report, self.preview, tr),
            tr(message_id('downloads.prune.unused_models')), variant="warning"), self._confirmed)

    def _confirmed(self, confirmed):
        if not self.is_attached or self._pruning or self._close_requested:
            return
        if confirmed and self.preview is not None:
            approved = set(self.preview["entries"])
            self._begin_work(message_id("downloads.prune.running"))
            self.apply_prune(approved)

    @work(thread=True, exclusive=True, group="model-store", exit_on_error=False)
    def apply_prune(self, approved):
        try:
            prune_store(root=self.root, apply=True, target_bytes=self.target,
                        approved_entries=approved, cancel=self._cancel,
                        on_wait=lambda: self._post(self._show_waiting))
        except ModelStoreCancelled:
            self._post(self._complete, translated_app(self.app).tr(message_id('run.status.cancelled')))
        except (ValueError, OSError) as exc:
            self._post(self._complete, str(exc))
        else:
            self._post(self._pruned)

    def _pruned(self):
        if not self.is_attached or self._pruning:
            return
        if self._close_requested:
            self._complete()
        else:
            self._begin_work()
            self.refresh_report()

    @on(Button.Pressed, "#store-close")
    def action_close(self):
        if self._working:
            self._close_requested = True
            self._cancel.set()
            self._set_status(message_id("downloads.model_store.cancelling"), level="warning")
        else:
            self.dismiss()

    def on_key(self, event) -> None:
        if focus_horizontal_action(self, "#store-actions", event.key):
            event.prevent_default()
            event.stop()
