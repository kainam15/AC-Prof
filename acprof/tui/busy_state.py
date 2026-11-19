"""Cached default-screen control references for low-overhead busy-state updates."""
from __future__ import annotations

from typing import TYPE_CHECKING

from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Button, Input as TextualInput, TabbedContent

from acprof.tui.ui.table import ResizableDataTable

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui

GENERAL_BUSY_SELECTOR = (
    ".config-control, .experiment-picker, #model-candidates, "
    ".ui-preference, .profile-tool, .report-control, "
    "#save-run-default, #restore-ui-defaults, #save-ui-settings, #open-environment-settings"
)

ACTION_BUTTON_OPERATIONS: tuple[tuple[str, str], ...] = (
    ("#start-run", "run"),
    ("#probe-largest", "probe"),
    ("#refresh-experiment-state", "summary"),
    ("#plot-results", "plot"),
    ("#report-open", "report"),
    ("#report-calculate", "html-report"),
    ("#html-empty-generate", "html-report"),
    ("#report-window-stats", "stats"),
    ("#stop-run", "stop"),
    ("#html-report-generate", "html-report"),
    ("#html-report-open", "html-open"),
    ("#html-report-export", "html-export"),
)


def _set_disabled_if_changed(widget: Widget, disabled: bool) -> None:
    if widget.disabled != disabled:
        widget.disabled = disabled


class BusyControlCache:
    """Hold references to default-screen controls queried during mount."""

    def __init__(self) -> None:
        self.captured = False
        self.tables: tuple[ResizableDataTable, ...] = ()
        self.general_controls: tuple[Widget, ...] = ()
        self.image_controls: tuple[Widget, ...] = ()
        self.action_buttons: tuple[tuple[Button, str], ...] = ()
        self.cancel_html_button: Button | None = None
        self.source_inputs: tuple[TextualInput, ...] = ()
        self.main_tabs: TabbedContent | None = None
        self.nav_targets: dict[str, Widget] = {}

    def clear(self) -> None:
        self.captured = False
        self.tables = ()
        self.general_controls = ()
        self.image_controls = ()
        self.action_buttons = ()
        self.cancel_html_button = None
        self.source_inputs = ()
        self.main_tabs = None
        self.nav_targets = {}

    def capture(self, app: AcprofTui) -> None:
        self.tables = tuple(app.query(ResizableDataTable))
        self.general_controls = tuple(app.query(GENERAL_BUSY_SELECTOR))
        self.image_controls = tuple(app.query(".image-control"))
        self.action_buttons = tuple(
            (app.query_one(selector, Button), operation)
            for selector, operation in ACTION_BUTTON_OPERATIONS
        )
        self.cancel_html_button = app.query_one("#html-report-cancel", Button)
        self.source_inputs = (app.query_one("#experiment-source", TextualInput),)
        self.main_tabs = app.query_one("#main-tabs", TabbedContent)
        self.nav_targets = {}
        self.captured = True

    def focus_fallback(self, app: AcprofTui) -> None:
        tabs = self.main_tabs
        if tabs is None or not tabs.is_mounted:
            return
        active = tabs.active
        target = self.nav_targets.get(active)
        if target is None or not target.is_mounted:
            try:
                target = app.query_one(f"#nav-{active}")
            except NoMatches:
                return
            self.nav_targets[active] = target
        app.screen.set_focus(target, scroll_visible=False)


def apply_busy_state(app: AcprofTui, cache: BusyControlCache, busy: bool) -> None:
    """Update interactive widget state without repeated default-screen CSS scans."""
    if not cache.captured:
        cache.capture(app)
    state = app.operation_state()
    app.updates.refresh_controls()
    busy = busy or state.busy
    measuring = app.execution.session.snapshot.measurement_active
    refreshing_images = app.images.operation == "refresh"
    # Configuration changes during a run can queue preview redraws and
    # make the visible settings differ from the running subprocess.
    if busy:
        app.configuration.cancel_preview_timer()
        app.preflight.pause_auto_check()
    for table in cache.tables:
        if not table.is_mounted:
            continue
        refreshing_table = refreshing_images and table.has_class("image-control")
        resize_enabled = not ((busy and not refreshing_table) or measuring)
        if table.resize_enabled != resize_enabled:
            table.resize_enabled = resize_enabled
    for widget in cache.general_controls:
        if not widget.is_mounted:
            continue
        if widget.id in {"experiment-source", "report-open"} and state.allows("report"):
            _set_disabled_if_changed(widget, False)
            continue
        if busy and widget.has_focus:
            cache.focus_fallback(app)
        disabled = busy or (state.checking and widget.id in {
            "open-environment-settings", "open-model-store",
        })
        _set_disabled_if_changed(widget, disabled)
    image_disabled = (busy and not refreshing_images) or measuring
    for widget in cache.image_controls:
        if widget.is_mounted:
            _set_disabled_if_changed(widget, image_disabled)
    for button, operation in cache.action_buttons:
        if button.is_mounted:
            _set_disabled_if_changed(button, not state.allows(operation))
    cancel_html = cache.cancel_html_button
    if cancel_html is not None and cancel_html.is_mounted:
        _set_disabled_if_changed(
            cancel_html,
            not (app.execution.session.kind == "html-report" and state.allows("stop")),
        )
    summary_disabled = not state.allows("summary")
    for source_input in cache.source_inputs:
        if source_input.is_mounted:
            _set_disabled_if_changed(source_input, summary_disabled)
    app.refresh_bindings()
    if not busy:
        app.set_input_cursor_blink_enabled(True)
    app.images.update_image_controls()
    app.images.sync_image_refresh_timer()
    app.preflight.refresh_preflight_state()
    app.profiles.update_action_availability()
    app.results.update_report_actions()
    if not busy:
        if app.configuration.validation_pending and app.configuration.preview_timer is None:
            app.configuration.configuration_changed()
        app.preflight.resume_auto_check()
