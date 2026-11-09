"""Event-driven progress presentation with theme-aware colors and no animation."""
from __future__ import annotations

import time

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
        self.set_class(isinstance(progress, ImageProgress), "image-accounting")
        model = self.query_one(DownloadProgressContent)
        image = self.query_one(ImageProgressContent)
        image.display = isinstance(progress, ImageProgress)
        model.display = not image.display
        if isinstance(progress, ImageProgress):
            image.update_progress(progress)
        else:
            model.update_progress(progress)

    def refresh_timing(self) -> None:
        if self.display:
            image = self.query_one(ImageProgressContent)
            if image.display:
                image.refresh_timing()


class ImageProgressContent(ProgressSummary):
    progress: ImageProgress | None = None
    fallback_reason: str = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation_payloads: dict[str, int] = {}
        self.operation_statuses: dict[str, str] = {}
        self.uncovered_sources: set[str] = set()
        self._received_at = 0.0
        self._elapsed_seconds = 0.0
        self._idle_seconds = 0.0
        self._clock: tuple[int, int, bool] = (0, 0, False)

    def reset_accounting(self) -> None:
        self.operation_payloads.clear()
        self.operation_statuses.clear()
        self.uncovered_sources.clear()
        self.progress = None
        self.fallback_reason = ""
        self._clock = (0, 0, False)

    def _timing(self, now: float) -> tuple[int, int, bool]:
        progress = self.progress
        active = progress is not None and progress.phase in {"pulling", "building"}
        delta = max(0.0, now - self._received_at) if active else 0.0
        elapsed = int(self._elapsed_seconds + delta)
        idle = int(self._idle_seconds + delta)
        return elapsed, idle, bool(active and progress is not None and idle >= progress.stall_after_seconds)

    def refresh_timing(self) -> None:
        """Use the run's preparation ticker; this widget owns no timer."""
        if self.progress is None or not self.progress.operation_id:
            return
        clock = self._timing(time.monotonic())
        if clock != self._clock:
            layout = clock[2] != self._clock[2]
            self._clock = clock
            self.refresh(layout=layout)

    def update_progress(self, progress: ImageProgress) -> None:
        now = time.monotonic()
        same_operation = (self.progress is not None and progress.operation_id
                          and self.progress.operation_id == progress.operation_id)
        previous_elapsed, previous_idle, _ = self._timing(now)
        self._elapsed_seconds = max(progress.elapsed_seconds, previous_elapsed if same_operation else 0)
        terminal = progress.phase not in {"pulling", "building"}
        self._idle_seconds = max(progress.idle_seconds, previous_idle if same_operation and terminal else 0)
        self._received_at = now
        if progress.operation_id:
            self.operation_statuses[progress.operation_id] = progress.transfer_status
            if progress.transferred_bytes is not None:
                self.operation_payloads[progress.operation_id] = max(
                    self.operation_payloads.get(progress.operation_id, 0), progress.transferred_bytes)
        self.uncovered_sources.update(progress.uncovered_sources)
        if (self.progress is None or self.progress.component != progress.component
                or progress.phase == "pulling"
                or progress.phase == "checking" and self.progress.phase in {"complete", "failed", "cancelled"}):
            self.fallback_reason = ""
        if progress.phase == "fallback":
            self.fallback_reason = progress.reason_code
        self.progress = progress
        self._clock = self._timing(now)
        self.refresh(layout=True)

    def _cumulative(self, progress: ImageProgress) -> tuple[str, int | None]:
        statuses = tuple(self.operation_statuses.values()) or (progress.transfer_status,)
        amount = sum(self.operation_payloads.values()) if self.operation_payloads else progress.transferred_bytes
        if amount is None:
            return ("waiting" if all(status == "waiting" for status in statuses) else "unavailable"), None
        if amount == 0 and all(status == "cached" for status in statuses):
            return "cached", 0
        status = "reported" if all(item in {"reported", "cached"} for item in statuses) else "partial"
        return status, amount

    @staticmethod
    def _duration(seconds: int) -> str:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _source_parts(self, progress: ImageProgress) -> list[RenderableType]:
        parts: list[RenderableType] = []
        if progress.source_image:
            parts.append(self.metric(message_id("run.docker.source_image"), progress.source_image))
        cells = []
        for label, value in (("registry", progress.source_registry), ("domain", progress.source_domain)):
            if value:
                cells.append(self.metric(message_id("run.docker.source_" + label), value))
        if cells:
            parts.append(Columns(cells, padding=(0, 2)))
        source_labels = []
        for prefix, value in (("category", progress.source_category), ("source_kind", progress.source_kind),
                              ("network_owner", progress.network_owner)):
            if value:
                source_labels.append(self.text(message_id("run.docker." + prefix + "." + value), "label"))
        if source_labels:
            parts.append(Text(" · ").join(source_labels))
        return parts

    def _accounting_parts(self, progress: ImageProgress, status: str) -> list[RenderableType]:
        parts: list[RenderableType] = []
        cells = []
        if progress.known_bytes or progress.unknown_items:
            cells.append(self.metric(message_id("run.docker.identified"), format_bytes(progress.known_bytes)))
        if progress.unknown_items is not None:
            cells.append(self.metric(message_id("run.docker.unknown_items"), str(progress.unknown_items)))
        if progress.cached_bytes:
            cells.append(self.metric(message_id("run.docker.cached"), format_bytes(progress.cached_bytes)))
        if progress.cached_items:
            cells.append(self.metric(message_id("run.docker.cached_items"), str(progress.cached_items)))
        if progress.reused_steps:
            cells.append(self.metric(message_id("run.docker.reused_steps"), str(progress.reused_steps)))
        if cells:
            parts.append(Columns(cells, padding=(0, 2)))
        if len(self.operation_statuses) > 1:
            parts.append(self.metric(message_id("run.docker.operation_downloaded"),
                                     message_id("run.docker.download_state." + progress.transfer_status,
                                                format_bytes(progress.transferred_bytes))))
        parts.append(self.text(message_id("run.docker.scope." + status), "label"))
        if self.uncovered_sources:
            names = [translated_app(self.app).tr(message_id("run.docker.uncovered." + name))
                     for name in sorted(self.uncovered_sources)]
            parts.append(self.text(message_id("run.docker.uncovered_sources", " · ".join(names)), "label"))
        return parts

    def render(self) -> RenderableType:
        progress = self.progress
        if progress is None:
            return Text()
        phase = ("pulling_prebuilt" if progress.phase == "pulling" and progress.component in {"platform", "environment"}
                 else progress.phase)
        heading = ("run.docker.activity." + progress.activity
                   if progress.phase in {"pulling", "building"} and progress.activity in {"frontend", "metadata", "build"}
                   else "run.docker.phase." + phase)
        role = {"fallback": "warning", "failed": "error", "cancelled": "warning", "complete": "success"}.get(progress.phase, "active")
        parts: list[RenderableType] = [self.metric(message_id("run.docker.component." + progress.component),
                                                 message_id(heading), role)]
        status, cumulative = self._cumulative(progress)
        parts.append(self.metric(message_id("run.docker.downloaded"),
                                 message_id("run.docker.download_state." + status, format_bytes(cumulative))))
        if progress.operation_id:
            elapsed, idle, stalled = self._clock
            parts.append(self.text(message_id("run.docker.wait_time", self._duration(elapsed),
                                               self._duration(idle)), "label"))
            if stalled:
                parts.append(self.text(message_id("run.docker.stalled"), "warning"))
        parts.extend(self._source_parts(progress))
        if progress.activity in {"frontend", "metadata"} and progress.transferred_bytes is None:
            parts.append(self.text(message_id("run.docker.metadata_payload"), "label"))
        if progress.source_registry:
            parts.append(self.text(message_id("run.docker.source_reference"), "label"))
        if progress.image != progress.source_image:
            parts.append(self.metric(message_id("run.docker.target_image"), progress.image))
        if self._clock[2]:
            parts.append(self.text(message_id("run.docker.stalled_hint"), "warning"))
        if self.fallback_reason in {"docker_manifest_missing", "docker_platform_mismatch", "docker_network_failed"}:
            parts.append(self.text(message_id("run.docker.fallback",
                                             message_id("run.docker.fallback_reason." + self.fallback_reason)), "warning"))
        if progress.content:
            content = progress.content
            content_status = content["status"]
            label = self.text(message_id("run.docker.content." + content_status),
                              "success" if content_status == "complete" else "warning")
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
        parts.extend(self._accounting_parts(progress, status))
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
