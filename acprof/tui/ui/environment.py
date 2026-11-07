"""Interactive, cached preflight presentation. No host probes or polling."""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markup import escape
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.geometry import Offset
from textual.timer import Timer
from textual.widgets import Button, Collapsible, Static

from acprof.capabilities import measurement_requested
from acprof.host.preparation.preparation_log import redact_text
from acprof.platform import capability_matrix
from acprof.tui.i18n import message_id
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.ui.rendering import CjkScreen

if TYPE_CHECKING:
    from acprof.tui.run.preflight_controller import PreflightController


FOUNDATIONS = {"environment", "runtime", "runtime_auto", "docker", "cgroup", "uv", "process_resources", "check_error"}
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
}


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


class EnvironmentPopover(Vertical):
    CLOSE_DELAY = 0.18

    def __init__(self, controller: PreflightController) -> None:
        super().__init__(id="environment-popover")
        self.controller = controller
        self.pinned = False
        self.expanded = False
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
            yield Static("", id="preflight-evidence", markup=False)
            yield Static("", id="preflight-capabilities", markup=False)
            with Vertical(id="preflight-details"):
                yield Vertical(id="preflight-rows")
        yield Static("", id="preflight-action-reason", markup=False)
        with Horizontal(id="preflight-buttons"):
            yield Button("", id="preflight-expand", compact=True)
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
        self.query_one("#preflight-scroll").styles.max_height = max(1, height - 6 - reason_height)

    def label(self, row: PreflightCheck) -> str:
        return self.tr("label." + LABELS[row.check_id]) if row.check_id in LABELS else self.controller.app.tr(row.label)

    def status(self, row: PreflightCheck) -> str:
        status = "error" if row.capability_status == "error" and row.status != "not_requested" else row.status
        return self.tr("item." + status)

    def remedy(self, row: PreflightCheck) -> str:
        return self.tr("remedy." + REMEDIES.get(row.check_id, "generic"))

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
        summary += "\n" + self.tr("lock_cached" if controller.lock_acquired else "lock_unknown")
        self.query_one("#preflight-summary", Static).update(summary)
        rows = display_checks(controller)
        # Summarize requested evidence, never treat declarations or disabled
        # metrics as passed probes or as an alternative launch policy.
        measurements = [row for row in rows if row.check_id in {*METRICS, "tcpdump", "tshark", "capture_permissions"}]
        capabilities = "\n".join(
            self.tr("capability_line", self.tr("item." + status),
                    ", ".join(self.label(row) for row in measurements if row.status == status))
            for status in ("fail", "warn", "ok", "declared", "pending", "not_requested")
            if any(row.status == status for row in measurements)
        )
        self.query_one("#preflight-capabilities", Static).update(capabilities)
        issues = sorted((row for row in rows if row.status in {"fail", "warn"}), key=lambda row: row.status != "fail")
        lines = [self.tr("stale_notice")] if state == "stale" else []
        if controller.error:
            lines.append(self.tr("check_error_hint"))
        lines.extend(self.tr("issue", self.label(row), self.status(row), self.remedy(row)) for row in issues)
        self.query_one("#preflight-issues", Static).update("\n".join(lines))
        self.query_one("#preflight-issues").display = bool(lines)
        self.query_one("#preflight-evidence", Static).update(self.tr("evidence"))
        allowed = controller.app.operation_state().allows("check")
        self.query_one("#preflight-retry", Button).disabled = not allowed
        reason = "" if allowed else self.tr("measurement_cached" if controller.app.execution.session.snapshot.measurement_active else "operation_busy")
        self.query_one("#preflight-action-reason", Static).update(reason)
        self.query_one("#preflight-action-reason").display = bool(reason)
        self.fit()
        for name, key in (("expand", "collapse" if self.expanded else "expand"), ("retry", "retry")):
            button = self.query_one(f"#preflight-{name}", Button)
            label = self.tr(key)
            if str(button.label) != label:
                button.label = label
                button.refresh(layout=True)
            button.tooltip = reason if button.disabled else None
        self.query_one("#preflight-details").display = self.expanded
        language = controller.app.configuration.preferences.language
        if self.expanded and (rows != self._rows or language != self._rows_language):
            self._rows, self._rows_language = rows, language
            self.call_later(self.render_rows, rows)

    async def render_rows(self, rows: tuple[PreflightCheck, ...]) -> None:
        if rows != self._rows or not self.is_mounted:
            return
        container = self.query_one("#preflight-rows")
        opened = {widget.id for widget in container.query(Collapsible) if not widget.collapsed}
        await container.remove_children()
        for foundation, key in ((True, "foundations"), (False, "measurements")):
            await container.mount(Static(self.tr(key), classes="environment-section"))
            for index, row in enumerate(rows):
                if (row.check_id in FOUNDATIONS) != foundation:
                    continue
                identifier = "preflight-item-" + (row.check_id or str(index))
                text = self.tr("source", row.source, row.capability_status or row.status)
                text += "\n" + self.controller.app.tr(row.detail)
                if row.status in {"warn", "fail"}:
                    text += "\n" + self.remedy(row)
                await container.mount(Collapsible(
                    Static(redact_text(text, limit=8192), markup=False), id=identifier,
                    title=escape(redact_text(self.tr("capability_line", self.label(row), self.status(row)))),
                    collapsed=identifier not in opened,
                ))

    @on(Button.Pressed)
    def operate(self, event: Button.Pressed) -> None:
        event.stop()
        action = event.button.id
        if action == "preflight-expand":
            self.expanded = not self.expanded
            self.update_cached()
            if self.expanded:
                self.call_after_refresh(self.query_one("#preflight-details").scroll_visible, animate=False, top=True)
            else:
                self.query_one("#preflight-scroll").scroll_home(animate=False)
        elif action == "preflight-retry":
            self.controller.quick_check()
