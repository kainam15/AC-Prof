"""Interactive, cached preflight presentation. No host probes or polling."""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.geometry import Offset, Size
from textual.timer import Timer
from textual.widgets import Button, Static

from acprof.host.preparation.preparation_log import redact_text
from acprof.messages import Message as CatalogText
from acprof.tui.i18n import message_id
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.run.environment_view import (
    CAPABILITY_SECTIONS,
    check_label,
    check_presentation,
    check_priority,
    global_check_view,
    global_presentation,
    item_status,
)
from acprof.tui.run.global_status import GlobalStatus, StatusItem
from acprof.tui.ui.navigation import NavigationContext
from acprof.tui.ui.rendering import CjkScreen
from acprof.tui.ui.structured import ActionButtons, ActionRequested, Callout, FactGrid, StatusBanner
from acprof.tui.ui.view_models import (
    CalloutContent,
    FactRow,
    StatusSummary,
    ViewAction,
    ViewSection,
    format_local_timestamp,
    status_symbol,
)

if TYPE_CHECKING:
    from acprof.tui.run.preflight_controller import PreflightController


def display_checks(controller: PreflightController) -> tuple[PreflightCheck, ...]:
    """Compatibility projection of the same snapshot used by launch admission."""
    return tuple(item.check for item in controller.status.items)


GROUP_SYMBOLS = {group: status_symbol(level) for group, level in (
    ("error", "error"), ("warning", "warning"), ("success", "success"), ("pending", "info"),
    ("not_requested", "neutral"),
)}


class EnvironmentIndicators(Horizontal):
    """Independent, content-width targets for each cached status."""

    def __init__(self, controller: PreflightController) -> None:
        super().__init__(id="current-environment")
        self.controller = controller

    def compose(self) -> ComposeResult:
        for group in GROUP_SYMBOLS:
            yield EnvironmentIndicator(self.controller, group)

    def update_cached(self, snapshot: GlobalStatus | None = None) -> None:
        snapshot = self.controller.status if snapshot is None else snapshot
        counts = snapshot.counts
        for entry in self.query(EnvironmentIndicator):
            entry.display = entry.group in {"error", "warning", "success"} or bool(counts[entry.group])
            entry.update(GROUP_SYMBOLS[entry.group] + " " + str(counts[entry.group]))
            label = message_id("run.environment.group." + entry.group)
            entry.tooltip = self.controller.app.tr(message_id("run.environment.group_count", label, counts[entry.group]))


class EnvironmentIndicator(Static, can_focus=True):
    """One status entry, with its own keyboard focus and pinned detail."""

    COMPONENT_CLASSES = {"environment--success", "environment--warning", "environment--error", "environment--pending", "environment--not_requested"}
    BINDINGS = [("enter,space", "toggle_environment", "Environment")]

    def __init__(self, controller: PreflightController, group: str) -> None:
        super().__init__("", id="environment-" + group, markup=False)
        self.controller = controller
        self.group = group

    def render(self) -> Text:
        content = self.content
        assert isinstance(content, str)
        label = Text(content)
        label.stylize(self.get_component_rich_style("environment--" + self.group), 0, 1)
        return label

    def action_toggle_environment(self) -> None:
        self.controller.panel.toggle(group=self.group)

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.action_toggle_environment()

    def on_enter(self) -> None:
        self.controller.panel.show(group=self.group)

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
    """A cached check with shared status vocabulary and keyboard inspection."""

    COMPONENT_CLASSES = {"environment--success", "environment--warning", "environment--error", "environment--neutral", "environment--info"}

    BINDINGS = [("enter,space", "inspect_issue", "Inspect issue")]

    def __init__(self, panel: EnvironmentPopover, row: PreflightCheck, identifier: str) -> None:
        super().__init__(id=f"preflight-signal-{identifier}", classes="environment-signal", markup=False)
        self.panel = panel
        self.row = row

    def render(self) -> Text:
        status = item_status(self.row)
        presentation = self.panel.presentation(self.row)
        suffix = "" if status in {"ok", "pending"} else " · " + self.panel.status(self.row)
        label = Text(self.panel.label(self.row))
        if self.size.width:
            label.truncate(max(1, self.size.width - Text(suffix).cell_len - 2), overflow="ellipsis")
        return Text.assemble(
            (presentation.symbol, self.get_component_rich_style("environment--" + presentation.level)),
            " ", label, suffix,
        )

    def on_enter(self) -> None:
        self.panel.preview(self.row)

    def on_leave(self) -> None:
        self.panel.end_preview(self.row)

    def on_focus(self) -> None:
        self.panel.select(self.row)

    def action_inspect_issue(self) -> None:
        self.panel.select(self.row)
        if self.row.check_id == 'docker' and self.row.status in {'fail', 'error', 'warn'}:
            self.panel.open_issue(self.row)

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.action_inspect_issue()


class EnvironmentPopover(Vertical):
    CLOSE_DELAY = 0.18

    def __init__(self, controller: PreflightController) -> None:
        super().__init__(id="environment-popover")
        self.controller = controller
        self.pinned_groups: set[str] = set()
        self.group = "error"
        self.close_timer: Timer | None = None
        self._rows: tuple[PreflightCheck, ...] = ()
        self._items: dict[str, StatusItem] = {}
        self._rows_language = ""
        self._selected_row: PreflightCheck | None = None
        self._hovered_row: PreflightCheck | None = None
        self._inspected_row: PreflightCheck | None = None
        self._origin: NavigationContext | None = None
        self._return_context: NavigationContext | None = None

    @property
    def pinned(self) -> bool:
        return self.group in self.pinned_groups

    @property
    def retry_button(self) -> Button:
        """The one visible recheck action, regardless of the selected check."""
        callout = self.query_one("#preflight-callout", Callout)
        actions = callout.query_one(ActionButtons) if callout.display else self.query_one("#preflight-recheck", ActionButtons)
        return actions.query(Button).first()

    def update_entries(self) -> None:
        for entry in self.controller.app.query(EnvironmentIndicator):
            entry.set_class(entry.group in self.pinned_groups, "pinned")

    def tr(self, key: str, *values: object) -> str:
        return self.controller.app.tr(message_id(f"run.environment.{key}", *values))

    def compose(self) -> ComposeResult:
        yield Static("", id="preflight-title", markup=False)
        yield StatusBanner(StatusSummary("neutral", ""), id="preflight-summary")
        with VerticalScroll(id="preflight-scroll"):
            yield Grid(id="preflight-capabilities")
        with VerticalScroll(id="preflight-inspection"):
            yield Static("", id="preflight-empty", markup=False)
            yield Callout(CalloutContent("neutral", "", "", ""), id="preflight-callout")
            yield FactGrid(ViewSection("", ()), id="preflight-facts")
            yield ActionButtons(id="preflight-recheck")
            yield Static("", id="preflight-action-reason", markup=False)
            yield FactGrid(ViewSection(message_id("run.environment.context.title"), (), collapsed=True),
                           id="preflight-context")

    def cancel_close(self) -> None:
        if self.close_timer is not None:
            self.close_timer.stop()
            self.close_timer = None

    def show(self, *, pin: bool = False, group: str = "error") -> None:
        if self.app.screen is not self.screen or not self.controller.app.form_ready:
            return
        self.cancel_close()
        if not self.display:
            self._origin = self._return_context or NavigationContext.capture(self.controller.app)
        self.group = group
        if pin:
            self.pinned_groups.add(group)
        self.update_entries()
        self.display = True
        self.update_cached()

    def toggle(self, *, group: str = "error") -> None:
        if group in self.pinned_groups:
            self.group = group
            self.close()
        else:
            self.show(pin=True, group=group)

    def close(self, *, restore_focus: bool = True) -> None:
        self.cancel_close()
        if self._hovered_row is not None:
            self.end_preview(self._hovered_row)
        focused = self.app.focused
        inside = focused is not None and (focused is self or self in focused.ancestors)
        self.pinned_groups.discard(self.group)
        self.update_entries()
        self.display = False
        if restore_focus and inside and self.app.screen is self.screen:
            self.controller.app.query_one("#environment-" + self.group).focus(scroll_visible=False)

    def dismiss_outside(self, point: Offset) -> None:
        entries = self.controller.app.query(EnvironmentIndicator)
        if self.display and not self.region.contains_point(point) and not any(entry.region.contains_point(point) for entry in entries):
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

    def fit(self, size: Size | None = None) -> None:
        if not self.is_mounted:
            return
        size = self.app.size if size is None else size
        width = min(68, size.width)
        bottom_panels = self.app.query("#bottom-panel")
        if bottom_panels and "command-hidden" not in bottom_panels.first().classes:
            bottom_reserve = 3 if size.height < 35 else 5
        else:
            bottom_reserve = 0
        available_height = max(1, size.height - 1 - bottom_reserve)
        popover_height = min(32, available_height)
        self.styles.offset = (max(0, size.width - width - 5), 1)
        self.styles.height = popover_height
        self.styles.max_height = available_height
        scroll_budget = max(1, min(10, popover_height // 3, max(1, popover_height - 8)))
        self.query_one("#preflight-scroll").styles.max_height = scroll_budget
        self.query_one("#preflight-capabilities").styles.grid_size_columns = 2 if size.width >= 60 else 1

    def label(self, row: PreflightCheck) -> str:
        return self.controller.app.tr(check_label(row))

    def status(self, row: PreflightCheck) -> str:
        return self.controller.app.tr(self.presentation(row).label)

    def presentation(self, row: PreflightCheck):
        item = self._items.get(row.check_id)
        return global_presentation(item) if item is not None else check_presentation(row)

    def open_issue(self, row: PreflightCheck) -> None:
        app = self.controller.app
        if app.is_busy() or self.controller.check_running or app.execution.session.snapshot.measurement_active:
            return
        # A removed or refreshed row must not navigate using stale evidence.
        item = next((item for item in self.controller.status.items if item.check == row), None)
        if item is None or not item.target:
            return
        origin = self._origin or NavigationContext.capture(app)
        self.close(restore_focus=False)
        if row.check_id == 'docker':
            from acprof.tui.run.docker_repair import DockerRepairScreen
            app.push_screen(DockerRepairScreen(self.controller), lambda _: origin.restore(app))
        elif item.target == 'resources':
            from acprof.tui.run.issue_view import IssueDiagnosticScreen
            app.push_screen(IssueDiagnosticScreen(self.controller, row.check_id), lambda _: origin.restore(app))
        elif item.target.startswith('field:'):
            self._return_context = origin
            app.query_one('#status-return').display = True
            app.configuration.open_config_field(item.target.removeprefix('field:'))
        elif item.target.startswith('permissions:'):
            app.configuration.open_environment_settings(permissions=True,
                permission_target=item.target.removeprefix('permissions:'), on_closed=lambda: origin.restore(app))
        elif item.target.startswith('diagnostic:'):
            from acprof.tui.run.issue_view import IssueDiagnosticScreen
            app.push_screen(IssueDiagnosticScreen(self.controller, row.check_id), lambda _: origin.restore(app))

    def return_to_origin(self) -> None:
        app = self.controller.app
        if self._return_context is None or app.execution.session.snapshot.measurement_active:
            return
        origin, self._return_context = self._return_context, None
        self.close(restore_focus=False)
        app.query_one('#status-return').display = False
        origin.restore(app)

    def select(self, row: PreflightCheck) -> None:
        if not self.display or row not in self._rows:
            return
        self._selected_row = row
        self.explain(self._hovered_row or row)

    def preview(self, row: PreflightCheck) -> None:
        if not self.display or row not in self._rows:
            return
        self._hovered_row = row
        self.explain(row)

    def end_preview(self, row: PreflightCheck) -> None:
        if self._hovered_row is row:
            self._hovered_row = None
            if self._selected_row is not None:
                self.explain(self._selected_row)

    def explain(self, row: PreflightCheck) -> None:
        if not self.display or row not in self._rows:
            return
        detail = self.controller.app.tr(row.detail) if isinstance(row.detail, CatalogText) else row.detail
        technical = redact_text("\n".join((
            f"check_id: {row.check_id}", f"status: {row.status}",
            f"capability_status: {row.capability_status}", f"source: {row.source}",
            f"runtime_kind: {row.runtime_kind}", "", detail,
        )), limit=None)
        view = global_check_view(
            self._items[row.check_id], technical_detail=technical, cause=redact_text(next((line for line in detail.splitlines() if line.strip()), ""), limit=180),
            check_allowed=self.controller.app.operation_state().allows("check"),
            navigate_allowed=not (self.controller.app.is_busy() or self.controller.check_running
                                  or self.controller.app.execution.session.snapshot.measurement_active),
        )
        callout = self.query_one("#preflight-callout", Callout)
        facts = self.query_one("#preflight-facts", FactGrid)
        callout.display = view.callout is not None
        facts.display = view.facts is not None
        self.query_one("#preflight-empty").display = False
        self.query_one("#preflight-recheck").display = view.callout is None
        if view.callout is not None and view.callout != callout.content:
            callout.update_content(view.callout)
        if view.facts is not None and view.facts != facts.section:
            facts.update_section(view.facts)
        if row != self._inspected_row:
            self.query_one("#preflight-inspection").scroll_home(animate=False)
        self._inspected_row = row

    def update_cached(self, snapshot: GlobalStatus | None = None) -> None:
        if not self.is_mounted or not self.display:
            return
        controller = self.controller
        snapshot = controller.status if snapshot is None else snapshot
        self._items = {item.check.check_id: item for item in snapshot.items}
        rows = tuple(item.check for item in snapshot.items if item.group == self.group)
        heading = self.tr("group_count", self.tr("group." + self.group), len(rows))
        self.query_one("#preflight-title", Static).update(
            self.tr("title", heading, self.tr("pinned" if self.pinned else "hover")))
        banner = self.query_one("#preflight-summary", StatusBanner)
        summary = StatusSummary(
            {"error": "error", "warning": "warning", "success": "success", "pending": "info"}.get(self.group, "neutral")
            if rows else "neutral", heading,
        )
        if summary != banner.summary:
            banner.update_summary(summary)
        allowed = controller.app.operation_state().allows("check")
        actions = (ViewAction("environment-retry", message_id("run.environment.retry"), primary=True, disabled=not allowed),)
        self.query_one("#preflight-recheck", ActionButtons).update_actions(actions)
        reason = "" if allowed else self.tr(
            "measurement_cached" if controller.app.execution.session.snapshot.measurement_active else "operation_busy")
        self.query_one("#preflight-action-reason", Static).update(reason)
        self.query_one("#preflight-action-reason").display = bool(reason)
        self.update_context()
        language = controller.app.configuration.preferences.language
        if rows != self._rows or language != self._rows_language:
            previous = self._selected_row
            rows_changed = rows != self._rows
            self._rows, self._rows_language = rows, language
            self._hovered_row = None
            self._selected_row = next(
                (row for row in rows if previous is not None and row.check_id and row.check_id == previous.check_id),
                min(rows, key=check_priority, default=None),
            )
            if rows_changed:
                self.query_one("#preflight-scroll").scroll_home(animate=False)
            self.call_later(self.render_rows, rows)
        if self._selected_row is not None:
            self.explain(self._hovered_row or self._selected_row)
        else:
            self._inspected_row = None
            self.query_one("#preflight-callout").display = False
            self.query_one("#preflight-facts").display = False
            empty = self.query_one("#preflight-empty", Static)
            empty.display = True
            empty.update(self.tr("checking_all" if self.group == "pending" and controller.display_state == "checking" else "empty_group"))
        self.retry_button.tooltip = reason or self.tr("retry")
        self.fit()

    def update_context(self) -> None:
        controller = self.controller
        timestamp = controller.completed_at.isoformat() if controller.completed_at else ""
        section = ViewSection(message_id("run.environment.context.title"), (
            FactRow(message_id("run.environment.context.platform"), controller.app.environment.label.split(" / ")[0]),
            FactRow(message_id("run.environment.context.checked_at"),
                    format_local_timestamp(timestamp) if timestamp else message_id("run.environment.no_check"),
                    detail=timestamp, copyable=bool(timestamp)),
            FactRow(message_id("run.environment.context.scope"), message_id("run.environment.unverified")),
            FactRow(message_id("run.environment.context.lock"),
                    message_id("run.environment.lock_cached" if controller.lock_acquired else "run.environment.lock_unknown")),
        ), collapsed=True)
        grid = self.query_one("#preflight-context", FactGrid)
        if section != grid.section:
            grid.update_section(section)

    async def render_rows(self, rows: tuple[PreflightCheck, ...]) -> None:
        if rows != self._rows or not self.is_mounted:
            return
        summary = self.query_one("#preflight-capabilities")
        sections: dict[str, list[EnvironmentSignal]] = {name: [] for name in (*CAPABILITY_SECTIONS, "other")}
        for index, row in sorted(enumerate(rows), key=lambda pair: check_priority(pair[1])):
            section = next((name for name, identifiers in CAPABILITY_SECTIONS.items() if row.check_id in identifiers), "other")
            signal = EnvironmentSignal(self, row, row.check_id or str(index))
            signal.tooltip = self.label(row) + " · " + self.status(row)
            sections[section].append(signal)
        incoming = [signal for signals in sections.values() for signal in signals]
        current = list(summary.query(EnvironmentSignal))
        if [signal.id for signal in current] == [signal.id for signal in incoming]:
            for old, new in zip(current, incoming, strict=True):
                old.row, old.tooltip = new.row, new.tooltip
                old.refresh(layout=True)
            for name in sections:
                for title in summary.query(f"#preflight-section-{name} .environment-section-title").results(Static):
                    title.update(self.tr("section." + name))
            return
        focused = self.app.focused
        focus_id = focused.id if isinstance(focused, EnvironmentSignal) and summary in focused.ancestors else None
        await summary.remove_children()
        await summary.mount(*(
            Vertical(
                Static(self.tr("section." + name), classes="environment-section-title", markup=False),
                *signals, id="preflight-section-" + name, classes="environment-section",
            )
            for name, signals in sections.items() if signals
        ))
        if focus_id and self.app.screen is self.screen and self.app.focused is None:
            target = next((signal for signal in self.query(EnvironmentSignal) if signal.id == focus_id), None)
            if target is not None:
                target.focus(scroll_visible=False)
            else:
                self.controller.app.query_one("#environment-" + self.group).focus(scroll_visible=False)

    @on(ActionRequested)
    def operate(self, event: ActionRequested) -> None:
        if event.action_id == "environment-retry":
            event.stop()
            self.controller.quick_check()
        elif event.action_id == "status-open" and self._inspected_row is not None:
            event.stop()
            self.open_issue(self._inspected_row)
