"""Compact runtime guidance projected from the existing preflight cache."""
from __future__ import annotations

from typing import TYPE_CHECKING

from acprof.tui.i18n import message_id
from acprof.tui.ui.structured import StatusBanner
from acprof.tui.ui.view_models import StatusSummary

if TYPE_CHECKING:
    from acprof.tui.run.preflight_controller import PreflightController


def refresh(controller: PreflightController) -> None:
    app = controller.app
    runtime = app.configuration.select("runtime")
    level = "info"
    if runtime == "auto":
        resolved = controller.selected_runtime if not controller.check_running else ""
        text = (message_id("run.runtime.auto_resolved", resolved.title()) if resolved
                else message_id("run.runtime.auto_pending"))
    elif runtime == "process":
        text = message_id("run.runtime.process_limits_note")
    elif controller.check_running:
        text = message_id("run.runtime.docker_checking")
    elif not controller.check_completed or not controller.config_matches():
        text = message_id("run.runtime.docker_recheck")
    else:
        check = next((item for item in controller.checks if item.check_id == "docker"), None)
        if controller.error or (check and check.status == "fail"):
            level, text = "error", message_id("run.runtime.docker_unavailable")
        elif check and check.status == "ok":
            level, text = "success", message_id("run.runtime.docker_ready")
        else:
            text = message_id("run.runtime.docker_recheck")
    banner = app.query_one("#runtime-status", StatusBanner)
    banner.display = not ((runtime == 'auto' and resolved) or level == 'success')
    banner.update_summary(StatusSummary(level, text))
