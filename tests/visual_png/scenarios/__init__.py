"""Test-only deterministic TUI scenes shared by the fixture app and VHS launcher."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from tests.visual_png.scenarios import (
    docker_frontend,
    error_modal,
    form,
    measuring,
    near_match,
    statistics,
    task_selection,
)


@dataclass(frozen=True)
class Scene:
    config: Callable[[], RunConfig]
    mount: Callable[[AcprofTui], None]
    columns: int = 120
    rows: int = 36
    language: str = "zh"


# Regexes must match actual visible terminal cells, not a background sentinel.
SCENARIOS: dict[str, Scene] = {
    "form": Scene(form.initial_config, form.mount),
    "form-small": Scene(form.initial_config, form.mount, columns=80, rows=24),
    "error-modal": Scene(error_modal.initial_config, error_modal.mount),
    "near-match": Scene(near_match.initial_config, near_match.mount),
    "measuring": Scene(measuring.initial_config, measuring.mount),
    "docker-frontend": Scene(docker_frontend.initial_config, docker_frontend.mount),
    "docker-frontend-small": Scene(docker_frontend.initial_config, docker_frontend.mount, columns=80, rows=24),
    "docker-frontend-en": Scene(docker_frontend.initial_config, docker_frontend.mount, language="en"),
    "docker-frontend-en-small": Scene(docker_frontend.initial_config, docker_frontend.mount, columns=80, rows=24, language="en"),
    "task-selection": Scene(task_selection.initial_config, task_selection.mount),
    "statistics": Scene(statistics.initial_config, statistics.mount, columns=80, rows=24),
    "statistics-en": Scene(statistics.initial_config, statistics.mount, columns=80, rows=24, language="en"),
}
