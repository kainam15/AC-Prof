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
from acprof.tui.models.store_summary import model_store_prune_summary, model_store_summary
from acprof.tui.ui.contracts import translated_app
from acprof.tui.ui.dialogs import ScrollConfirmActionScreen
from acprof.tui.ui.rendering import CjkCompositor


class ModelStoreScreen(ModalScreen):
    BINDINGS = [("escape", "close", "关闭")]
    DEFAULT_CSS = """
    ModelStoreScreen { align: center middle; }
    #store-dialog { width: 94%; height: 85%; border: round $accent; background: $surface; padding: 0 1; }
    #store-scroll { height: 1fr; }
    #store-report, #store-status { height: auto; }
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
                yield Static(translated_app(self.app).tr(message_id('images.storage.loading')), id="store-status", markup=False)
                yield Static("", id="store-report", markup=False)
            with Horizontal(id="store-actions", classes="action-bar"):
                yield Button(translated_app(self.app).tr(message_id('common.action.close_dialog')), id="store-close")
                yield Button(translated_app(self.app).tr(message_id('common.action.refresh')), id="store-refresh", disabled=True)
                yield Button(translated_app(self.app).tr(message_id('downloads.prune.unused_models')), id="store-prune", disabled=True)

    def on_mount(self):
        for button in self.query(Button):
            button.active_effect_duration = 0
        self.refresh_report()

    def on_unmount(self):
        self._cancel.set()

    def _post(self, callback, *args):
        if self.is_mounted:
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
        if not self.is_mounted:
            return
        self._report = report
        self._render_report()
        if not self._close_requested:
            self.query_one("#store-status", Static).update(translated_app(self.app).tr(message_id('downloads.prune.preparing_preview')))

    def _show_waiting(self):
        if self.is_mounted and not self._close_requested:
            self.query_one("#store-status", Static).update(translated_app(self.app).tr(message_id('downloads.model_store.busy')))

    def _complete(self, error="", preview=None):
        if not self.is_mounted:
            return
        self._working = False
        if self._close_requested:
            self.dismiss()
            return
        self.preview = preview
        self.query_one("#store-refresh", Button).disabled = False
        self.query_one("#store-status", Static).update(error)
        self._render_report()
        self.query_one("#store-prune", Button).disabled = not (
            preview and (preview["entries"] or preview["reclaimable_bytes"]))

    def _render_report(self):
        self.query_one("#store-report", Static).update(
            model_store_summary(self._report, self.preview, translated_app(self.app).tr))

    @on(Button.Pressed, "#store-refresh")
    def refresh_clicked(self):
        if self._working:
            return
        self._begin_work()
        self.refresh_report()

    def _begin_work(self, status="正在计算清理预览……"):
        self._working = True
        self._cancel = Event()
        self.preview = None
        self._report = {}
        self._render_report()
        for key in ("store-refresh", "store-prune"):
            self.query_one(f"#{key}", Button).disabled = True
        self.query_one("#store-status", Static).update(translated_app(self.app).tr(status))

    @on(Button.Pressed, "#store-prune")
    def prune_clicked(self):
        if self._working or not self.preview:
            return
        tr = translated_app(self.app).tr
        self.app.push_screen(ScrollConfirmActionScreen(
            tr(message_id('downloads.prune.unused_models')), model_store_prune_summary(self._report, self.preview, tr),
            tr(message_id('downloads.prune.unused_models')), variant="warning"), self._confirmed)

    def _confirmed(self, confirmed):
        if confirmed and self.preview is not None:
            approved = set(self.preview["entries"])
            self._begin_work("正在复核并清理未使用模型……")
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
        if not self.is_mounted:
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
            self.query_one("#store-status", Static).update(translated_app(self.app).tr(message_id('downloads.model_store.cancelling')))
        else:
            self.dismiss()
