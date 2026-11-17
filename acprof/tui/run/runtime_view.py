"""Project the preflight cache into the runtime control without changing intent."""
from __future__ import annotations

from typing import TYPE_CHECKING

from acprof.tui.ui.runtime_select import RuntimeSelect

if TYPE_CHECKING:
    from acprof.tui.run.preflight_controller import PreflightController


def refresh(controller: PreflightController) -> None:
    controller.app.query_one("#runtime", RuntimeSelect).show_resolved(controller.selected_runtime)
