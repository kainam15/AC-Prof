"""Test-only deterministic TUI scenes shared by the fixture app and VHS launcher."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from tests.visual_png.scenarios import error_modal, form, near_match


@dataclass(frozen=True)
class Scene:
    config: Callable[[], RunConfig]
    mount: Callable[[AcprofTui], None]
    ready_pattern: str | None


# Regexes must match actual visible terminal cells, not a background sentinel.
SCENARIOS: dict[str, Scene] = {
    "form": Scene(form.initial_config, form.mount, "模型 ID"),
    "error-modal": Scene(error_modal.initial_config, error_modal.mount, "模型接口校验失败"),
    "near-match": Scene(near_match.initial_config, near_match.mount, None),
}
