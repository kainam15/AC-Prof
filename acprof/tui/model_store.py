"""On-demand storage review; never polls or prunes during measurement."""
from pathlib import Path

from textual import on, work
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from acprof.host.model_store import disk_report, prune_store
from acprof.tui.rendering import CjkCompositor
from acprof.tui.views import ConfirmActionScreen


class ModelStoreScreen(ModalScreen):
    BINDINGS = [("escape", "close", "关闭")]
    DEFAULT_CSS = """
    ModelStoreScreen { align: center middle; }
    #store-dialog { width: 94%; height: 85%; border: round $accent; background: $surface; padding: 0 1; }
    #store-scroll { height: 1fr; }
    #store-report { height: auto; }
    #store-actions { height: 3; }
    """

    def __init__(self, root: Path, target: str = ""):
        super().__init__()
        self._compositor = CjkCompositor()
        self.root = root
        self.target = target
        self.preview = None
        self._working = True

    def compose(self):
        with Vertical(id="store-dialog"):
            yield Static("Model Store", markup=False)
            with VerticalScroll(id="store-scroll"):
                yield Static("", id="store-report", markup=False)
            with Horizontal(id="store-actions", classes="action-bar"):
                yield Button(self.app.tr("关闭"), id="store-close", disabled=True)
                yield Button(self.app.tr("刷新"), id="store-refresh", disabled=True)
                yield Button(self.app.tr("清理未使用模型"), id="store-prune", disabled=True)

    def on_mount(self):
        self.refresh_report()

    @work(thread=True, exclusive=True)
    def refresh_report(self):
        import json
        try:
            report = disk_report(self.root)
            self.preview = prune_store(root=self.root, target_bytes=self.target)
            text = json.dumps({**report, "prune_preview": self.preview}, ensure_ascii=False, indent=2)
        except (ValueError, OSError) as exc:
            self.preview = None
            text = str(exc)
        self.app.call_from_thread(self._show_report, text)

    def _show_report(self, text):
        if not self.is_mounted:
            return
        self._working = False
        self.query_one("#store-close", Button).disabled = False
        self.query_one("#store-refresh", Button).disabled = False
        self.query_one("#store-report", Static).update(text)
        self.query_one("#store-prune", Button).disabled = not (self.preview and self.preview["entries"])

    @on(Button.Pressed, "#store-refresh")
    def refresh_clicked(self):
        if self._working:
            return
        self._begin_work()
        self.refresh_report()

    def _begin_work(self):
        self._working = True
        for key in ("store-close", "store-refresh", "store-prune"):
            self.query_one(f"#{key}", Button).disabled = True

    @on(Button.Pressed, "#store-prune")
    def prune_clicked(self):
        if self._working or not self.preview:
            return
        import json
        self.app.push_screen(ConfirmActionScreen(self.app.tr("清理未使用模型"),
            json.dumps(self.preview, ensure_ascii=False, indent=2), self.app.tr("清理")), self._confirmed)

    def _confirmed(self, confirmed):
        if confirmed:
            self._begin_work()
            self.apply_prune()

    @work(thread=True, exclusive=True)
    def apply_prune(self):
        # Keep all newly added entries; approval only covered the displayed list.
        approved = set(self.preview["entries"])
        current = disk_report(self.root)
        keep = {item["entry_id"] for item in current["models"]} - approved
        try:
            prune_store(root=self.root, apply=True, keep=keep)
        except (ValueError, OSError) as exc:
            self.preview = None
            self.app.call_from_thread(self._show_report, str(exc))
            return
        self.app.call_from_thread(self.refresh_report)

    @on(Button.Pressed, "#store-close")
    def action_close(self):
        if not self._working:
            self.dismiss()
