"""Test-only deterministic TUI scenes shared by the fixture app and VHS launcher."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from tests.visual_png.scenarios import error_modal, form, measuring, near_match, task_selection


@dataclass(frozen=True)
class Scene:
    config: Callable[[], RunConfig]
    mount: Callable[[AcprofTui], None]
    columns: int = 120
    rows: int = 36


# Regexes must match actual visible terminal cells, not a background sentinel.
SCENARIOS: dict[str, Scene] = {
    "form": Scene(form.initial_config, form.mount),
    "form-small": Scene(form.initial_config, form.mount, columns=80, rows=24),
    "error-modal": Scene(error_modal.initial_config, error_modal.mount),
    "near-match": Scene(near_match.initial_config, near_match.mount),
    "measuring": Scene(measuring.initial_config, measuring.mount),
    "task-selection": Scene(task_selection.initial_config, task_selection.mount),
}
