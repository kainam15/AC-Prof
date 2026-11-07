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

from acprof.host.preparation.preparation_log import redact_text
from acprof.tui.i18n import message_id
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.run.environment_checks import ENVIRONMENT_CHECK_IDS
from acprof.tui.ui.rendering import CjkScreen

if TYPE_CHECKING:
    from acprof.tui.run.preflight_controller import PreflightController


LABELS = {
    "environment": "environment", "docker": "docker", "cgroup": "cgroup", "uv": "uv",
    "process_resources": "process_resources", "latency": "latency", "throughput": "throughput",
    "container_cpu": "cgroup_cpu", "container_memory": "cgroup_memory", "process_cpu": "process_cpu", "process_memory": "process_memory",
    "process_runtime": "process_runtime", "ncu": "ncu", "nsys": "nsys", "intel_advisor": "advisor",
    "torch": "torch", "massif": "massif",
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
    """Present machine evidence without consulting the editable experiment."""
    rows = list(controller.environment_checks)
    if controller.environment_error:
        rows.append(PreflightCheck(message_id("run.environment.label.check_error"), "error", controller.environment_error,
                                   check_id="check_error"))
    if not controller.check_completed:
        rows.extend(PreflightCheck(name, "pending", message_id("run.environment.checking_all"),
                                   check_id=name, source="not_checked") for name in ENVIRONMENT_CHECK_IDS
                    if name not in {row.check_id for row in rows})
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
        status_label = self.panel.tr("signal." + status)
        label = Text(self.panel.label(self.row))
        if self.size.width:
            # Keep availability readable even when a long translated name is
            # shortened in the two-column layout.
            label.truncate(max(1, self.size.width - Text(status_label).cell_len - 5), overflow="ellipsis")
        return Text.assemble(
            (dot, self.get_component_rich_style("environment--" + role)),
            " ", label, " · " + status_label,
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
        state = controller.display_state
        completed = controller.completed_at.strftime("%Y-%m-%d %H:%M:%S %Z") if controller.completed_at else self.tr("no_check")
        self.query_one("#preflight-title", Static).update(self.tr("title", self.tr("state." + state),
                                                           self.tr("pinned" if self.pinned else "hover")))
        summary = self.tr("summary", completed)
        self.query_one("#preflight-summary", Static).update(summary)
        rows = display_checks(controller)
        lines = []
        if state == "error":
            lines.append(self.tr("check_error_hint"))
        self.query_one("#preflight-issues", Static).update("\n".join(lines))
        self.query_one("#preflight-issues").display = bool(lines)
        evidence = self.tr("lock_cached" if controller.lock_acquired else "lock_unknown")
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
