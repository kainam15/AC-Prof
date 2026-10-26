"""Event-driven progress presentation with theme-aware colors and no animation."""
from __future__ import annotations

from rich.console import Group, RenderableType
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from acprof.download_events import DownloadProgress
from acprof.tui.app_contracts import translated_app
from acprof.tui.i18n import message_id
from acprof.tui.monitor import visible_statuses
from acprof.tui.presentation import UNKNOWN, format_bytes
from acprof.tui.progress import ProgressSnapshot


class ProgressSummary(Static):
    COMPONENT_CLASSES = {
        "progress--label", "progress--value", "progress--active",
        "progress--success", "progress--warning", "progress--error",
    }
    DEFAULT_CSS = """
    ProgressSummary .progress--label { color: $text-muted; }
    ProgressSummary .progress--value { color: $text; text-style: bold; }
    ProgressSummary .progress--active { color: $accent; text-style: bold; }
    ProgressSummary .progress--success { color: $success; text-style: bold; }
    ProgressSummary .progress--warning { color: $warning; text-style: bold; }
    ProgressSummary .progress--error { color: $error; text-style: bold; }
    """

    def text(self, source: str, role: str = "value") -> Text:
        return Text(translated_app(self.app).tr(source), self.get_component_rich_style(f"progress--{role}"),
                    overflow="fold", no_wrap=False)

    def metric(self, label: str, value: str, role: str = "value") -> Text:
        return Text.assemble(self.text(label, "label"), "  ", self.text(value, role))


class PreparationStatus(ProgressSummary):
    snapshot: ProgressSnapshot | None = None

    def update_snapshot(self, snapshot: ProgressSnapshot) -> None:
        self.snapshot = snapshot
        self.display = bool(visible_statuses(snapshot))
        self.refresh(layout=True)

    def render(self) -> RenderableType:
        if self.snapshot is None:
            return Text()
        states = {
            "not_started": ("未开始", "label"), "running": ("进行中", "active"),
            "waiting": ("待确认", "warning"), "passed": ("通过", "success"),
            "failed": ("失败", "error"), "cancelled": ("已取消", "warning"),
        }
        snapshot = self.snapshot
        table = Table.grid(padding=(0, 3))
        cells = []
        for label, status in visible_statuses(snapshot):
            value, role = states[status]
            cells.append(self.metric(message_id(label), value, role))
        if not cells:
            return Text()
        table.add_row(*cells)
        return table


class DownloadProgressPanel(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield DownloadProgressContent()

    def update_progress(self, progress: DownloadProgress) -> None:
        self.query_one(DownloadProgressContent).update_progress(progress)


class DownloadProgressContent(ProgressSummary):
    progress: DownloadProgress | None = None

    def update_progress(self, progress: DownloadProgress) -> None:
        self.progress = progress
        self.refresh(layout=True)

    def render(self) -> RenderableType:
        progress = self.progress
        if progress is None:
            return Text()
        states = {
            "downloading": ("ui.0938", "active"), "verifying": ("ui.0939", "warning"),
            "complete": ("ui.0940", "success"), "failed": ("ui.0941", "error"),
            "cancelled": ("ui.0942", "warning"),
        }
        state, role = states[progress.state]
        heading = Table.grid(expand=True, padding=(0, 2))
        heading.add_column(ratio=1, overflow="fold")
        heading.add_column()
        heading.add_row(Text(progress.model, self.get_component_rich_style("progress--value"),
                             overflow="fold", no_wrap=False),
                        self.text(message_id(state), role) if progress.state != "complete" else Text())

        known = progress.completed_bytes is not None and progress.total_bytes > 0
        percent = (f"{progress.completed_bytes / progress.total_bytes:.0%}"
                   if progress.completed_bytes is not None and progress.total_bytes > 0 else UNKNOWN)
        amount = Text.assemble(
            self.text(message_id('ui.2147'), "label"), "  ",
            self.text(format_bytes(progress.completed_bytes)), " / ", self.text(format_bytes(progress.total_bytes)),
            "  ", self.text(percent, role if known else "label"),
        )
        progress_row = Table.grid(padding=(0, 2))
        if progress.completed_bytes is not None and progress.total_bytes > 0:
            # Rich's determinate bar renders only when an existing event updates us.
            progress_row.add_row(ProgressBar(
                total=progress.total_bytes, completed=progress.completed_bytes, width=12,
                complete_style=self.get_component_rich_style(f"progress--{role}"),
                finished_style=self.get_component_rich_style(f"progress--{role}"),
                style=self.get_component_rich_style("progress--label"),
            ), amount)
        else:
            progress_row.add_row(amount)

        cells = []
        if progress.files_total:
            cells.append(self.metric(message_id('ui.2148'), f"{progress.files_completed}/{progress.files_total}"))
        if progress.cached_bytes is not None:
            cells.append(self.metric(message_id('ui.2149'), format_bytes(progress.cached_bytes)))
        if progress.transferred_bytes is not None:
            cells.append(self.metric(message_id('ui.2150'), format_bytes(progress.transferred_bytes)))
        if progress.rate_bytes_s is not None:
            speed = format_bytes(int(progress.rate_bytes_s)) + "/s"
            cells.append(self.metric(message_id('ui.2151'), speed, "active"))
        parts: list[RenderableType] = [heading, progress_row]
        if cells:
            # Let Rich size the single row by content rather than equal-width columns.
            metrics = Table.grid(expand=True, padding=(0, 2))
            metrics.add_row(*cells)
            parts.append(metrics)
        if progress.detail:
            detail = (message_id('ui.0949') if progress.detail == 'retry_cache_baseline_unknown'
                      else message_id('ui.0946', progress.detail))
            parts.append(self.text(detail, "warning"))
        return Group(*parts)
