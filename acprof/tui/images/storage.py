"""Storage space modal; Docker work is scheduled by ImageActions."""

from __future__ import annotations

from collections.abc import Callable

from rich import box
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from acprof.host.images.inventory import STORAGE_KINDS, DockerStorage, ImageManagementError
from acprof.messages import join_messages, message
from acprof.tui.i18n import message_id
from acprof.tui.ui.contracts import translated_app
from acprof.tui.ui.dialogs import focus_horizontal_action
from acprof.tui.ui.presentation import UNKNOWN, format_bytes
from acprof.tui.ui.rendering import CjkCompositor
from acprof.tui.ui.structured import StatusBanner
from acprof.tui.ui.view_models import StatusSummary, format_local_timestamp


class StorageSpaceScreen(ModalScreen[None]):
    AUTO_FOCUS = ""
    BINDINGS = [("escape", "close", "关闭 (Esc)")]
    CSS = """
    StorageSpaceScreen { align: center middle; background: $background 75%; }
    #storage-dialog {
        width: 76;
        max-width: 94%;
        height: 38;
        max-height: 92%;
        border: round $accent;
        border-title-align: center;
        background: $surface;
        padding: 0 2;
    }
    #storage-content { height: 1fr; scrollbar-size: 1 1; }
    #storage-content Static { height: auto; }
    #storage-content .storage-heading { color: $text; text-style: bold; margin-top: 1; }
    #storage-disk-bar-row { height: 1; margin: 1 0; }
    #storage-content #storage-disk-bar { width: 1fr; height: 1; }
    #storage-disk-percent { width: 9; text-align: right; }
    #storage-banner { height: auto; padding: 0 0 1 0; }
    #storage-note, .storage-state { color: $text-muted; }
    #storage-note, #storage-details { margin-top: 1; }
    #storage-buttons { height: 3; align-horizontal: right; margin-top: 1; }
    #storage-buttons Button { width: auto; min-width: 12; margin-left: 1; }
    """

    def __init__(self, refresh: Callable[[], None], selected_count: int):
        super().__init__()
        self._compositor = CjkCompositor()
        self.refresh_storage = refresh
        self.selected_count = selected_count
        self._storage: DockerStorage | None = None
        self._disk_reading: DockerStorage | None = None
        self._usage_reading: DockerStorage | None = None
        self._estimate: int | None = None
        self._error: ImageManagementError | None = None
        self._loading = True
        self._estimate_pending = False

    def compose(self) -> ComposeResult:
        tr = translated_app(self.app).tr
        with Vertical(id="storage-dialog") as dialog:
            dialog.border_title = tr("存储空间")
            yield StatusBanner(StatusSummary("info", message_id("images.storage.loading")),
                               id="storage-banner", headline_id="storage-status")
            with VerticalScroll(id="storage-content"):
                yield Static(tr("Docker 数据目录"), classes="storage-heading")
                yield Static("", id="storage-root", markup=False)
                yield Static("", id="storage-environment-state", classes="storage-state", markup=False)
                yield Static(tr("磁盘空间"), classes="storage-heading")
                yield Static("", id="storage-disk-state", classes="storage-state", markup=False)
                with Horizontal(id="storage-disk-bar-row"):
                    yield Static("", id="storage-disk-bar")
                    yield Static(tr(UNKNOWN), id="storage-disk-percent")
                yield Static("", id="storage-disk-stats")
                yield Static(tr("Docker 空间"), classes="storage-heading")
                yield Static("", id="storage-usage-state", classes="storage-state", markup=False)
                yield Static("", id="storage-usage")
                yield Static("", id="storage-totals")
                yield Static(tr("当前所选镜像"), classes="storage-heading")
                yield Static("", id="storage-selection-state", classes="storage-state", markup=False)
                yield Static("", id="storage-selection")
                yield Static(tr("合计可能含共享缓存；释放量为镜像层估算上限，实际可能更少。"),
                             id="storage-note", markup=False)
                yield Static("", id="storage-details", markup=False)
            with Horizontal(id="storage-buttons"):
                yield Button(tr("刷新"), id="storage-refresh")
                yield Button(tr("关闭 (Esc)"), id="storage-close", variant="primary")

    def on_mount(self) -> None:
        self.show_loading()
        # Even a synchronous scheduling failure needs a fully mounted screen
        # to receive its result, and the first frame should show loading.
        self.call_after_refresh(self.refresh_storage)

    def _pairs(self, selector: str, rows: tuple[tuple[str, str], ...]) -> None:
        table = Table.grid(padding=(0, 3))
        table.add_column(min_width=18)
        table.add_column()
        for label, value in rows:
            table.add_row(Text(translated_app(self.app).tr(label)), Text(translated_app(self.app).tr(value)))
        self.query_one(selector, Static).update(table)

    def show_storage(self, storage: DockerStorage | None, estimate: int | None,
                     error: ImageManagementError | None = None, *, finished: bool = True) -> None:
        previous = self._storage
        if (storage is None or previous is None
                or (storage.connection, storage.daemon_id, storage.endpoint) !=
                   (previous.connection, previous.daemon_id, previous.endpoint)):
            self._disk_reading = self._usage_reading = None
        if storage is not None:
            if storage.disk is not None:
                self._disk_reading = storage
            elif (storage.disk_error is None or self._disk_reading is None
                  or self._disk_reading.root_dir != storage.root_dir):
                self._disk_reading = None
            if any(row.size_bytes is not None or row.reclaimable_bytes is not None for row in storage.usage):
                self._usage_reading = storage
        self._storage, self._estimate, self._error = storage, estimate, error
        self._loading = False
        self._estimate_pending = not finished
        self._render_storage()
        self.query_one("#storage-refresh", Button).disabled = not finished

    def _state(self, reading: DockerStorage | None, *, failed: bool = False,
               unavailable: str = "images.storage.unverifiable") -> str:
        if self._loading:
            return message_id("images.storage.item_loading")
        if reading is None:
            return message_id("images.storage.failed" if failed else unavailable)
        timestamp = format_local_timestamp(reading.observed_at.isoformat())
        if reading is self._storage:
            return message_id("images.storage.partial_read" if failed else "images.storage.current_read", timestamp)
        return message_id("images.storage.last_read", timestamp)

    def _render_storage(self) -> None:
        tr = translated_app(self.app).tr
        storage = None if self._loading else self._storage
        disk_reading = None if self._loading else self._disk_reading
        usage_reading = None if self._loading else self._usage_reading
        disk = disk_reading.disk if disk_reading else None
        estimate = self._estimate if not self._loading and not self._estimate_pending else None
        pending = tr(message_id("images.storage.item_loading"))

        def size(value: int | None) -> str:
            return pending if self._loading else tr(format_bytes(value))

        root = storage.root_dir if storage else ""
        self.query_one("#storage-root", Static).update(pending if self._loading else root or tr(UNKNOWN))
        self.query_one("#storage-environment-state", Static).update(tr(self._state(storage, failed=bool(self._error and not storage))))
        self.query_one("#storage-disk-state", Static).update(tr(self._state(
            disk_reading, failed=bool(storage and storage.disk_error))))
        self.query_one("#storage-usage-state", Static).update(tr(self._state(
            usage_reading, failed=bool(storage and storage.usage_error))))
        percentage = disk.used / disk.total * 100 if disk and disk.total else None
        self.query_one("#storage-disk-bar", Static).update(
            ProgressBar(total=100, completed=percentage, complete_style=self.app.theme_variables["accent"],
                        finished_style=self.app.theme_variables["accent"])
            if percentage is not None else Text("—")
        )
        self.query_one("#storage-disk-percent", Static).update(
            f"{percentage:.1f}%" if percentage is not None else "—" if self._loading else tr(UNKNOWN))
        self._pairs("#storage-disk-stats", tuple(
            (label, size(value)) for label, value in (
                ("总容量", disk.total if disk else None), ("已用", disk.used if disk else None),
                ("可用", disk.available if disk else None),
            )
        ))
        table = Table(box=box.SQUARE, expand=True, padding=(0, 1))
        for label in ("类型", "占用", "可回收"):
            table.add_column(tr(label), justify="left" if label == "类型" else "right", no_wrap=True)
        usage = {row.kind: row for row in usage_reading.usage} if usage_reading else {}
        for kind, label in zip(STORAGE_KINDS, ("镜像", "Docker 容器", "Volume", "Build Cache"), strict=True):
            row = usage.get(kind)
            table.add_row(tr(label), size(row.size_bytes if row else None),
                          size(row.reclaimable_bytes if row else None))
        self.query_one("#storage-usage", Static).update(table)
        self._pairs("#storage-totals", (
            ("Docker 合计", size(usage_reading.total_bytes if usage_reading else None)),
            ("可回收", size(usage_reading.reclaimable_bytes if usage_reading else None)),
        ))
        # Never combine a historical disk reading with a new release estimate.
        available_after = min(disk.total, disk.available + estimate) if (
            disk and disk_reading is storage and estimate is not None) else None
        selection_state: str
        if not self.selected_count:
            selection_state = message_id("images.storage.not_applicable")
        elif self._loading or self._estimate_pending:
            selection_state = message_id("images.storage.item_loading")
        else:
            selection_state = self._state(storage if estimate is not None else None, failed=bool(self._error))
        self.query_one("#storage-selection-state", Static).update(tr(selection_state))

        def selected_size(value: int | None) -> str:
            if not self.selected_count:
                return tr(message_id("images.storage.not_applicable"))
            if self._loading or self._estimate_pending:
                return pending
            return ("≈" if value is not None else "") + tr(format_bytes(value))

        self._pairs("#storage-selection", (
            ("预计释放", selected_size(estimate)),
            ("删除后可用", selected_size(available_after)),
        ))
        notes = [tr(note) for note in storage.warnings] if storage else []
        if not self.selected_count:
            notes.append(tr("尚未勾选镜像。"))
        issues = []
        if self._error and not self._loading:
            issues.append(("当前所选镜像" if storage else "Docker 环境", self._error))
        if storage:
            issues.extend((label, error) for label, error in (
                ("磁盘空间", storage.disk_error), ("Docker 空间", storage.usage_error),
            ) if error is not None)
        for label, error in issues:
            notes.append(f"{tr(label)}: {tr(error.reason)}" + (f"\n{error.detail}" if error.detail else ""))
        self.query_one("#storage-details", Static).update("\n".join(dict.fromkeys(notes)))
        if issues:
            label, error = issues[0]
            reason = message_id("images.storage.environment_changed") if error.code == "identity" else message(error.reason)
            headline = message("{0}：{1}", message(label), reason)
            if len(issues) > 1:
                headline = join_messages("\n", (headline, message_id("images.storage.more_errors", len(issues) - 1)))
            self._show_status("warning" if storage else "error", headline)
        elif self._loading or self._estimate_pending:
            self._show_status("info", message_id("images.storage.loading"))
        elif storage and storage.disk is None:
            self._show_status("warning", message_id("images.storage.remote_disk_unverifiable"))
        elif self.selected_count and estimate is None:
            self._show_status("warning", message_id("images.storage.unverified_layer_sizes"))
        else:
            self._show_status("success", message_id("images.storage.complete"))

    def _show_status(self, level: str, headline: str) -> None:
        self.query_one("#storage-banner", StatusBanner).update_summary(StatusSummary(level, headline))

    def show_loading(self) -> None:
        self._loading = True
        self._render_storage()
        self.query_one("#storage-refresh", Button).disabled = True

    def show_blocked(self, reason: str) -> None:
        self._loading = False
        # No identity check ran: don't present an earlier reading as current.
        self._storage = self._disk_reading = self._usage_reading = None
        self._estimate = None
        self._error = None
        self._estimate_pending = False
        self._render_storage()
        self._show_status("warning", reason)
        self.query_one("#storage-refresh", Button).disabled = False

    @on(Button.Pressed, "#storage-refresh")
    def request_refresh(self, event: Button.Pressed) -> None:
        event.stop()
        self.refresh_storage()

    @on(Button.Pressed, "#storage-close")
    def close_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_close()

    def action_close(self) -> None:
        self.dismiss(None)

    def on_key(self, event: events.Key) -> None:
        if focus_horizontal_action(self, "#storage-buttons", event.key):
            event.prevent_default()
            event.stop()
