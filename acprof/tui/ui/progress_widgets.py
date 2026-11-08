"""Event-driven progress presentation with theme-aware colors and no animation."""
from __future__ import annotations

from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from acprof.events.download import DownloadProgress
from acprof.events.image import ImageProgress
from acprof.tui.i18n import message_id
from acprof.tui.run.monitor import visible_statuses
from acprof.tui.run.progress import ProgressSnapshot
from acprof.tui.ui.contracts import translated_app
from acprof.tui.ui.presentation import UNKNOWN, format_bytes


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

    def byte_bar(self, current: int, total: int, role: str = "active") -> ProgressBar:
        return ProgressBar(total=total, completed=current, width=12,
                           complete_style=self.get_component_rich_style(f"progress--{role}"),
                           finished_style=self.get_component_rich_style(f"progress--{role}"),
                           style=self.get_component_rich_style("progress--label"))


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


class PhaseTimes(Static):
    DEFAULT_CSS = "PhaseTimes { display: none; height: auto; padding: 0 2; color: $text-muted; }"


class DownloadProgressPanel(VerticalScroll):
    def compose(self) -> ComposeResult:
        plan = Static(id="download-plan-summary")
        plan.display = False
        yield plan
        yield DownloadProgressContent()
        yield ImageProgressContent()

    def update_plan(self, summary: str) -> None:
        plan = self.query_one("#download-plan-summary", Static)
        plan.update(summary)
        plan.display = bool(summary)

    def update_progress(self, progress: DownloadProgress | ImageProgress) -> None:
        self.set_class(isinstance(progress, ImageProgress) and progress.known_bytes is not None, "image-accounting")
        model = self.query_one(DownloadProgressContent)
        image = self.query_one(ImageProgressContent)
        image.display = isinstance(progress, ImageProgress)
        model.display = not image.display
        if isinstance(progress, ImageProgress):
            image.update_progress(progress)
        else:
            model.update_progress(progress)


class ImageProgressContent(ProgressSummary):
    progress: ImageProgress | None = None
    fallback_reason: str = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation_payloads: dict[str, int] = {}

    def reset_accounting(self) -> None:
        self.operation_payloads.clear()
        self.progress = None
        self.fallback_reason = ""

    def update_progress(self, progress: ImageProgress) -> None:
        if progress.operation_id and progress.transferred_bytes is not None:
            self.operation_payloads[progress.operation_id] = max(
                self.operation_payloads.get(progress.operation_id, 0), progress.transferred_bytes)
        if (self.progress is None or self.progress.component != progress.component
                or progress.phase == "pulling"
                or progress.phase == "checking" and self.progress.phase in {"complete", "failed", "cancelled"}):
            self.fallback_reason = ""
        if progress.phase == "fallback":
            self.fallback_reason = progress.reason_code
        self.progress = progress
        self.refresh(layout=True)

    def render(self) -> RenderableType:
        progress = self.progress
        if progress is None:
            return Text()
        phase = ("pulling_prebuilt" if progress.phase == "pulling" and progress.component in {"platform", "environment"}
                 else progress.phase)
        role = {"fallback": "warning", "failed": "error", "cancelled": "warning", "complete": "success"}.get(progress.phase, "active")
        parts: list[RenderableType] = [self.metric(message_id("run.docker.component." + progress.component),
                                                 message_id("run.docker.phase." + phase), role)]
        parts.append(Text(progress.image, overflow="fold"))
        if self.fallback_reason in {"docker_manifest_missing", "docker_platform_mismatch", "docker_network_failed"}:
            parts.append(self.text(message_id("run.docker.fallback",
                                             message_id("run.docker.fallback_reason." + self.fallback_reason)), "warning"))
        if progress.content:
            content = progress.content
            status = content["status"]
            label = self.text(message_id("run.docker.content." + status), "success" if status == "complete" else "warning")
            if content.get("present") is not None:
                label.append(f" ({content['present']}/{content.get('required') or '?'})")
            parts.append(label)
        if progress.current is not None and progress.total is not None:
            row = Table.grid(padding=(0, 2))
            row.add_row(self.byte_bar(progress.current, progress.total), self.text(message_id("run.docker.activity_progress",
                        message_id("run.docker.activity." + progress.activity), format_bytes(progress.current),
                        format_bytes(progress.total), f"{progress.current / progress.total:.0%}")))
            parts.append(row)
        elif progress.current is not None:
            parts.append(self.metric(message_id("run.docker.activity." + progress.activity), format_bytes(progress.current)))
        if progress.known_bytes is not None:
            cumulative = sum(self.operation_payloads.values()) if self.operation_payloads else progress.transferred_bytes
            cells = [self.metric(message_id("run.docker.downloaded"), format_bytes(cumulative)),
                     self.metric(message_id("run.docker.identified"), format_bytes(progress.known_bytes)),
                     self.metric(message_id("run.docker.unknown_items"), str(progress.unknown_items))]
            if progress.cached_items or progress.cached_bytes:
                cells.append(self.metric(message_id("run.docker.cached"),
                                         f"{format_bytes(progress.cached_bytes)} / {progress.cached_items}"))
            if progress.reused_steps:
                cells.append(self.metric(message_id("run.docker.reused_steps"), str(progress.reused_steps)))
            parts.append(Columns(cells, padding=(0, 2)))
            if len(self.operation_payloads) > 1:
                parts.append(self.metric(message_id("run.docker.operation_downloaded"), format_bytes(progress.transferred_bytes)))
            parts.append(self.text(message_id("run.docker.scope." + progress.transfer_status), "label"))
        elif progress.phase in {"pulling", "building"} and progress.current is None:
            parts.append(self.text(message_id("run.docker.unknown_size"), "label"))
        if progress.detail and progress.phase != "failed":
            parts.append(Text(progress.detail, overflow="fold"))
        return Group(*parts)


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
            "downloading": ("downloads.state.downloading", "active"), "verifying": ("downloads.state.verifying", "warning"),
            "complete": ("downloads.state.verified", "success"), "failed": ("downloads.state.failed", "error"),
            "cancelled": ("downloads.state.cancelled", "warning"),
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
            self.text(message_id('downloads.progress.file_bytes'), "label"), "  ",
            self.text(format_bytes(progress.completed_bytes)), " / ", self.text(format_bytes(progress.total_bytes)),
            "  ", self.text(percent, role if known else "label"),
        )
        progress_row = Table.grid(padding=(0, 2))
        if progress.completed_bytes is not None and progress.total_bytes > 0:
            # Rich's determinate bar renders only when an existing event updates us.
            progress_row.add_row(self.byte_bar(progress.completed_bytes, progress.total_bytes, role), amount)
        else:
            progress_row.add_row(amount)

        cells = []
        if progress.files_total:
            cells.append(self.metric(message_id('downloads.progress.file_count'), f"{progress.files_completed}/{progress.files_total}"))
        if progress.cached_bytes is not None:
            cells.append(self.metric(message_id('downloads.progress.cached_bytes'), format_bytes(progress.cached_bytes)))
        if progress.transferred_bytes is not None:
            cells.append(self.metric(message_id('downloads.progress.network_payload'), format_bytes(progress.transferred_bytes)))
        if progress.rate_bytes_s is not None:
            speed = format_bytes(int(progress.rate_bytes_s)) + "/s"
            cells.append(self.metric(message_id('downloads.progress.rate'), speed, "active"))
        parts: list[RenderableType] = [heading, progress_row]
        if cells:
            # Wrap whole metrics when the sidebar leaves too little page width.
            parts.append(Columns(cells, padding=(0, 2)))
        if progress.detail:
            detail = (message_id('downloads.progress.retry_baseline_pending') if progress.detail == 'retry_cache_baseline_unknown'
                      else message_id('downloads.progress.accounting_error', progress.detail))
            parts.append(self.text(detail, "warning"))
        return Group(*parts)
