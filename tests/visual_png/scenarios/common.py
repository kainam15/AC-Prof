"""Shared fixture-only UI stabilization; no hooks into production collection."""
from __future__ import annotations

from acprof.tui.app import AcprofTui


def freeze_for_screenshot(app: AcprofTui) -> None:
    app.clear_notifications()
    app.set_input_cursor_blink_enabled(False)
    app.set_focus(None)


def apply_fixture_preflight(app: AcprofTui) -> None:
    """Mark a synthetic host check as passed without probing a real host."""
    app.preflight.checks = ()
    app.preflight.error = ""
    app.preflight.checked_config = app.preflight.preflight_config()
    app.preflight.check_completed = True
    app.preflight.check_running = False
    app.preflight.refresh_preflight_state()
