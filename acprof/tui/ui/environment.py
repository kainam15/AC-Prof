"""Interactive, cached preflight presentation. No host probes or polling."""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.geometry import Offset
from textual.timer import Timer
from textual.widgets import Button, Static

from acprof.capabilities import measurement_requested
from acprof.host.preparation.preparation_log import redact_text
from acprof.platform import capability_matrix
from acprof.tui.i18n import message_id
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.ui.rendering import CjkScreen

if TYPE_CHECKING:
    from acprof.tui.run.preflight_controller import PreflightController


METRICS = (
    "latency", "throughput", "container_cpu", "container_memory", "process_cpu", "process_memory",
    "cpu_energy", "cpu_instructions", "capture_interface", "gpu_device", "gpu_memory", "gpu_utilization", "gpu_power",
)
LABELS = {
    "environment": "environment", "docker": "docker", "cgroup": "cgroup", "uv": "uv",
    "process_resources": "process_resources", "latency": "latency", "throughput": "throughput",
    "container_cpu": "cpu", "container_memory": "memory", "process_cpu": "cpu", "process_memory": "memory",
    "cpu_energy": "rapl", "cpu_instructions": "perf", "capture_interface": "capture",
    "capture_permissions": "capture_permissions",
    "gpu_device": "gpu", "gpu_memory": "gpu_memory", "gpu_utilization": "gpu_utilization", "gpu_power": "gpu_power",
    "profiler": "profiler", "check_error": "check_error",
}
REMEDIES = {
    "environment": "platform", "docker": "docker", "cgroup": "cgroup", "uv": "uv",
    "process_resources": "resources", "cpu_energy": "rapl", "cpu_instructions": "permissions",
    "tcpdump": "capture_tools", "tshark": "capture_tools", "capture_permissions": "permissions",
    "capture_interface": "interface", "gpu_device": "gpu", "gpu_memory": "gpu", "gpu_utilization": "gpu",
    "gpu_power": "gpu", "process_cpu": "process_metrics", "process_memory": "process_metrics",
}


def item_status(row: PreflightCheck) -> str:
    return "error" if row.capability_status == "error" and row.status != "not_requested" else row.status


def display_checks(controller: PreflightController) -> tuple[PreflightCheck, ...]:
    """Fill evidence gaps for display only; never feed these rows into launch policy."""
    config = controller.preflight_config()
    fresh = controller.check_completed and controller.config_matches()
    # Cached results cannot describe capabilities of a newly selected runtime,
    # device or collection mode.
    rows = list(controller.checks) if fresh else []
    if fresh and controller.error:
        rows.append(PreflightCheck(message_id("run.environment.label.check_error"), "error", controller.error,
                                   check_id="check_error"))
    known = {row.check_id for row in rows}
    runtime = controller.selected_runtime or config.runtime
    support = capability_matrix(controller.app.environment)
    gpu = "on" in config.gpus.split(",")
    for name in METRICS:
        if name.startswith("container_") and runtime == "process":
            continue
        if name.startswith("process_") and runtime != "process":
            continue
        metric = {"capture_interface": "packet_latency", "process_cpu": "container_cpu",
                  "process_memory": "container_memory"}.get(name, name)
        requested = (gpu if name in {"gpu_device", "gpu_memory", "gpu_utilization"} else
                     measurement_requested(config.profiling_mode, metric, gpu=gpu))
        if name in known:
            continue
        status = ("not_requested" if not requested else
                  "declared" if fresh and name in support else "pending")
        detail = message_id("run.environment.unverified")
        if status == "not_requested":
            detail = message_id("run.preflight.not_collected")
        rows.append(PreflightCheck(name, status, detail, check_id=name,
                                   source=f"capability_matrix: {support[name]}" if status == "declared" else "not_checked"))
    tools = tuple(tool for tool in (config.compute_profile_tool, config.execution_profile_tool) if tool != "none")
    rows.append(PreflightCheck("Profiler", "pending" if tools else "not_requested",
                              message_id("run.environment.profiler_detail", ", ".join(tools)) if tools else
                              message_id("run.preflight.not_collected"), check_id="profiler", source="configuration"))
    # A capability status never overrides the UI severity from quick_preflight.
    return tuple(rows)


class EnvironmentIndicator(Static, can_focus=True):
    BINDINGS = [("enter,space", "toggle_environment", "Environment")]

    def __init__(self, controller: PreflightController) -> None:
        super().__init__("", id="current-environment", markup=False)
        self.controller = controller

    def action_toggle_environment(self) -> None:
        self.controller.show_environment_status()

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.action_toggle_environment()

    def on_enter(self) -> None:
        self.controller.panel.show()

    def on_leave(self) -> None:
        self.controller.panel.schedule_close()


class EnvironmentHostScreen(CjkScreen):
    """Hide transient overlays when a modal takes over the screen."""

    def on_screen_suspend(self) -> None:
        for panel in self.query(EnvironmentPopover):
            if not panel.pinned:
                panel.close(restore_focus=False)

    def _forward_event(self, event: events.Event) -> None:
        # Textual routes mouse input directly to children (Pilot also enters
        # here). Observe it before a Button or Select can stop its propagation.
        if isinstance(event, events.MouseDown):
            for panel in self.query(EnvironmentPopover):
                panel.dismiss_outside(event.screen_offset)
        super()._forward_event(event)


class EnvironmentSignal(Static, can_focus=True):
    """A cached check with a theme-aware dot and hover/keyboard explanation."""

    COMPONENT_CLASSES = {"environment--success", "environment--warning", "environment--error", "environment--muted"}
    DEFAULT_CSS = """
    EnvironmentSignal .environment--success { color: $success; }
    EnvironmentSignal .environment--warning { color: $warning; }
    EnvironmentSignal .environment--error { color: $error; }
    EnvironmentSignal .environment--muted { color: $text-muted; }
    """

    def __init__(self, panel: EnvironmentPopover, row: PreflightCheck, identifier: str) -> None:
        super().__init__(id=f"preflight-signal-{identifier}", classes="environment-signal", markup=False)
        self.panel = panel
        self.row = row

    def render(self) -> Text:
        status = item_status(self.row)
        role = {"ok": "success", "fail": "error", "error": "error", "not_requested": "muted"}.get(status, "warning")
        dot = "○" if status == "not_requested" else "●"
        return Text.assemble(
            (dot, self.get_component_rich_style("environment--" + role)),
            f" {self.panel.label(self.row)} · {self.panel.tr('signal.' + status)}",
        )

    def on_enter(self) -> None:
        self.panel.explain(self.row)

    def on_focus(self) -> None:
        self.panel.explain(self.row)


class EnvironmentPopover(Vertical):
    CLOSE_DELAY = 0.18

    def __init__(self, controller: PreflightController) -> None:
        super().__init__(id="environment-popover")
        self.controller = controller
        self.pinned = False
        self.close_timer: Timer | None = None
        self._rows: tuple[PreflightCheck, ...] = ()
        self._rows_language = ""

    def tr(self, key: str, *values: object) -> str:
        return self.controller.app.tr(message_id(f"run.environment.{key}", *values))

    def compose(self) -> ComposeResult:
        yield Static("", id="preflight-title", markup=False)
        with VerticalScroll(id="preflight-scroll"):
            yield Static("", id="preflight-summary", markup=False)
            yield Static("", id="preflight-issues", markup=False)
            yield Static("", id="preflight-boundary", markup=False)
            yield Grid(id="preflight-capabilities")
        with VerticalScroll(id="preflight-inspection"):
            yield Static("", id="preflight-explanation", markup=False)
            yield Static("", id="preflight-evidence", markup=False)
        yield Static("", id="preflight-action-reason", markup=False)
        with Horizontal(id="preflight-buttons"):
            yield Button("", id="preflight-retry", compact=True)

    def cancel_close(self) -> None:
        if self.close_timer is not None:
            self.close_timer.stop()
            self.close_timer = None

    def show(self, *, pin: bool = False) -> None:
        if self.app.screen is not self.screen or not self.controller.app.form_ready:
            return
        self.cancel_close()
        self.pinned = self.pinned or pin
        self.display = True
        self.update_cached()

    def toggle(self) -> None:
        if self.display and self.pinned:
            self.close()
        else:
            self.show(pin=True)

    def close(self, *, restore_focus: bool = True) -> None:
        self.cancel_close()
        focused = self.app.focused
        inside = focused is not None and (focused is self or self in focused.ancestors)
        self.pinned = False
        self.display = False
        if restore_focus and inside and self.app.screen is self.screen:
            self.controller.app.query_one("#current-environment").focus(scroll_visible=False)

    def dismiss_outside(self, point: Offset) -> None:
        entry = self.controller.app.query_one("#current-environment")
        if self.display and not self.region.contains_point(point) and not entry.region.contains_point(point):
            self.close(restore_focus=False)

    def schedule_close(self) -> None:
        self.cancel_close()
        if self.display and not self.pinned:
            self.close_timer = self.set_timer(self.CLOSE_DELAY, self.close_if_outside)

    def close_if_outside(self) -> None:
        self.close_timer = None
        if not self.pinned:
            self.dismiss_outside(self.app.mouse_position)

    def on_enter(self) -> None:
        self.cancel_close()

    def on_leave(self) -> None:
        self.schedule_close()

    def on_unmount(self) -> None:
        self.cancel_close()

    def on_resize(self) -> None:
        self.fit()

    def fit(self) -> None:
        if not self.is_mounted:
            return
        height = max(1, self.app.size.height - 2)
        self.styles.max_height = height
        reason_height = 2 if self.query_one("#preflight-action-reason").display else 0
        self.query_one("#preflight-scroll").styles.max_height = max(1, height - 12 - reason_height)
        self.query_one("#preflight-capabilities").styles.grid_size_columns = 2 if self.app.size.width >= 60 else 1

    def label(self, row: PreflightCheck) -> str:
        return self.tr("label." + LABELS[row.check_id]) if row.check_id in LABELS else self.controller.app.tr(row.label)

    def status(self, row: PreflightCheck) -> str:
        return self.tr("item." + item_status(row))

    def remedy(self, row: PreflightCheck) -> str:
        return self.tr("remedy." + REMEDIES.get(row.check_id, "generic"))

    def explain(self, row: PreflightCheck) -> None:
        if not self.display or row not in self._rows:
            return
        detail = redact_text(self.controller.app.tr(row.detail), limit=8192)
        text = self.tr("capability_line", self.label(row), self.status(row)) + "\n"
        text += self.tr("explanation", detail, self.remedy(row)) if item_status(row) in {"warn", "fail", "error"} else detail
        text += "\n" + self.tr("source", row.source, row.capability_status or row.status)
        self.query_one("#preflight-explanation", Static).update(redact_text(text, limit=None))
        self.query_one("#preflight-inspection").scroll_home(animate=False)

    def update_cached(self) -> None:
        if not self.is_mounted or not self.display:
            return
        self.fit()
        controller = self.controller
        config = controller.preflight_config()
        state = controller.display_state
        runtime = controller.selected_runtime or (config.runtime if config.runtime != "auto" else self.tr("runtime_pending"))
        completed = controller.completed_at.strftime("%Y-%m-%d %H:%M:%S %Z") if controller.completed_at else self.tr("no_check")
        self.query_one("#preflight-title", Static).update(self.tr("title", self.tr("state." + state),
                                                           self.tr("pinned" if self.pinned else "hover")))
        summary = self.tr("summary", runtime.title(), config.profiling_mode.title(), config.gpus.title(), completed)
        can_start = not controller.preflight_run_reason() and controller.app.operation_state().allows("run")
        summary += "\n" + self.tr("can_start" if can_start else "cannot_start")
        self.query_one("#preflight-summary", Static).update(summary)
        rows = display_checks(controller)
        lines = [self.tr("stale_notice")] if state == "stale" else []
        if state == "error":
            lines.append(self.tr("check_error_hint"))
        self.query_one("#preflight-issues", Static).update("\n".join(lines))
        self.query_one("#preflight-issues").display = bool(lines)
        self.query_one("#preflight-boundary", Static).update(self.tr("boundary"))
        evidence = self.tr("evidence") + "\n" + self.tr("lock_cached" if controller.lock_acquired else "lock_unknown")
        self.query_one("#preflight-evidence", Static).update(evidence)
        allowed = controller.app.operation_state().allows("check")
        self.query_one("#preflight-retry", Button).disabled = not allowed
        reason = "" if allowed else self.tr("measurement_cached" if controller.app.execution.session.snapshot.measurement_active else "operation_busy")
        self.query_one("#preflight-action-reason", Static).update(reason)
        self.query_one("#preflight-action-reason").display = bool(reason)
        self.fit()
        button = self.query_one("#preflight-retry", Button)
        label = self.tr("retry")
        if str(button.label) != label:
            button.label = label
            button.refresh(layout=True)
        button.tooltip = reason if button.disabled else None
        language = controller.app.configuration.preferences.language
        if rows != self._rows or language != self._rows_language:
            self._rows, self._rows_language = rows, language
            self.query_one("#preflight-explanation", Static).update(self.tr("inspect_hint"))
            self.call_later(self.render_rows, rows)

    async def render_rows(self, rows: tuple[PreflightCheck, ...]) -> None:
        if rows != self._rows or not self.is_mounted:
            return
        summary = self.query_one("#preflight-capabilities")
        await summary.remove_children()
        priority = {"fail": 0, "error": 0, "warn": 1, "ok": 2, "declared": 3, "pending": 3, "not_requested": 4}
        ordered = sorted(enumerate(rows), key=lambda pair: priority[item_status(pair[1])])
        await summary.mount(*(EnvironmentSignal(self, row, row.check_id or str(index)) for index, row in ordered))

    @on(Button.Pressed, "#preflight-retry")
    def operate(self, event: Button.Pressed) -> None:
        event.stop()
        self.controller.quick_check()
