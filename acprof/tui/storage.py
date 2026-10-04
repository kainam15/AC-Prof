"""Storage space modal; Docker work is scheduled by ImageActions."""

from __future__ import annotations

from collections.abc import Callable

from rich import box
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from acprof.host.image_management import STORAGE_KINDS, DockerStorage
from acprof.messages import message
from acprof.tui.presentation import UNKNOWN, format_bytes
from acprof.tui.rendering import CjkCompositor


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
    #storage-note, #storage-status { color: $text-muted; margin-top: 1; }
    #storage-buttons { height: 3; align-horizontal: right; margin-top: 1; }
    #storage-buttons Button { width: auto; min-width: 12; margin-left: 1; }
    """

    def __init__(self, refresh: Callable[[], None], selected_count: int):
        super().__init__()
        self._compositor = CjkCompositor()
        self.refresh_storage = refresh
        self.selected_count = selected_count

    def compose(self) -> ComposeResult:
        tr = self.app.tr
        with Vertical(id="storage-dialog") as dialog:
            dialog.border_title = tr("存储空间")
            with VerticalScroll(id="storage-content"):
                yield Static(tr("Docker 数据目录"), classes="storage-heading")
                yield Static(tr(UNKNOWN), id="storage-root", markup=False)
                yield Static(tr("磁盘空间"), classes="storage-heading")
                with Horizontal(id="storage-disk-bar-row"):
                    yield Static("", id="storage-disk-bar")
                    yield Static(tr(UNKNOWN), id="storage-disk-percent")
                yield Static("", id="storage-disk-stats")
                yield Static(tr("Docker 空间"), classes="storage-heading")
                yield Static("", id="storage-usage")
                yield Static("", id="storage-totals")
                yield Static(tr("当前所选镜像"), classes="storage-heading")
                yield Static("", id="storage-selection")
                yield Static(tr("合计可能含共享缓存；释放量为镜像层估算上限，实际可能更少。"),
                             id="storage-note", markup=False)
                yield Static("", id="storage-status", markup=False)
            with Horizontal(id="storage-buttons"):
                yield Button(tr("刷新"), id="storage-refresh")
                yield Button(tr("关闭 (Esc)"), id="storage-close", variant="primary")

    def on_mount(self) -> None:
        self.show_storage(None, None)
        self.refresh_storage()

    def _pairs(self, selector: str, rows: tuple[tuple[str, str], ...]) -> None:
        table = Table.grid(padding=(0, 3))
        table.add_column(min_width=18)
        table.add_column()
        for label, value in rows:
            table.add_row(Text(self.app.tr(label)), Text(self.app.tr(value)))
        self.query_one(selector, Static).update(table)

    def show_storage(self, storage: DockerStorage | None, estimate: int | None, error: str = "") -> None:
        tr = self.app.tr
        disk = storage.disk if storage else None
        self.query_one("#storage-root", Static).update(storage.root_dir if storage else tr(UNKNOWN))
        percentage = disk.used / disk.total * 100 if disk and disk.total else None
        self.query_one("#storage-disk-bar", Static).update(
            ProgressBar(total=100, completed=percentage, complete_style=self.app.theme_variables["accent"],
                        finished_style=self.app.theme_variables["accent"])
            if percentage is not None else Text("—")
        )
        self.query_one("#storage-disk-percent", Static).update(f"{percentage:.1f}%" if percentage is not None else tr(UNKNOWN))
        self._pairs("#storage-disk-stats", tuple(
            (label, format_bytes(value)) for label, value in (
                ("总容量", disk.total if disk else None), ("已用", disk.used if disk else None),
                ("可用", disk.available if disk else None),
            )
        ))
        table = Table(box=box.SQUARE, expand=True, padding=(0, 1))
        for label in ("类型", "占用", "可回收"):
            table.add_column(tr(label), justify="left" if label == "类型" else "right", no_wrap=True)
        usage = {row.kind: row for row in storage.usage} if storage else {}
        for kind, label in zip(STORAGE_KINDS, ("镜像", "Docker 容器", "Volume", "Build Cache")):
            row = usage.get(kind)
            table.add_row(tr(label), tr(format_bytes(row.size_bytes if row else None)),
                          tr(format_bytes(row.reclaimable_bytes if row else None)))
        self.query_one("#storage-usage", Static).update(table)
        self._pairs("#storage-totals", (
            ("Docker 合计", format_bytes(storage.total_bytes if storage else None)),
            ("可回收", format_bytes(storage.reclaimable_bytes if storage else None)),
        ))
        available_after = min(disk.total, disk.available + estimate) if disk and estimate is not None else None
        self._pairs("#storage-selection", (
            ("预计释放", ("≈" if estimate else "") + format_bytes(estimate)),
            ("删除后可用", ("≈" if available_after is not None else "") + format_bytes(available_after)),
        ))
        notes = [tr(note) for note in storage.warnings] if storage else []
        if not self.selected_count:
            notes.append(tr("尚未勾选镜像。"))
        if error:
            notes.append(tr(message("存储空间读取未完成：{0}", error)))
        self.query_one("#storage-status", Static).update("\n".join(notes))
        self.query_one("#storage-refresh", Button).disabled = False

    def show_loading(self) -> None:
        self.query_one("#storage-refresh", Button).disabled = True
        self.query_one("#storage-status", Static).update(self.app.tr("正在读取存储空间……"))

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
