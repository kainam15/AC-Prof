"""Test-only deterministic TUI scenes shared by the fixture app and VHS launcher."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

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
    ready_pattern: str
    columns: int = 120
    rows: int = 36
    language: str = "zh"
    page: str = "run-tab"


# Regexes must match actual visible terminal cells, not a background sentinel.
_FORM = Scene(form.initial_config, form.mount, "1 个正式窗口")
_DOCKER_FRONTEND = Scene(
    docker_frontend.initial_config, docker_frontend.mount, r"registry-1\.docker\.io", page="monitor-tab",
)
_STATISTICS = Scene(
    statistics.initial_config, statistics.mount, "交互式 HTML 报告", columns=80, rows=24, page="results-tab",
)
SCENARIOS: dict[str, Scene] = {
    "form": _FORM,
    "form-small": replace(_FORM, columns=80, rows=24),
    "error-modal": Scene(error_modal.initial_config, error_modal.mount, "× RAPL DRAM 能耗 · 无权限"),
    "near-match": Scene(near_match.initial_config, near_match.mount, "3 个正式窗口"),
    "measuring": Scene(measuring.initial_config, measuring.mount, r"当前 case\s+2", page="monitor-tab"),
    "docker-frontend": _DOCKER_FRONTEND,
    "docker-frontend-small": replace(_DOCKER_FRONTEND, columns=80, rows=24),
    "docker-frontend-en": replace(_DOCKER_FRONTEND, language="en"),
    "docker-frontend-en-small": replace(_DOCKER_FRONTEND, columns=80, rows=24, language="en"),
    "task-selection": Scene(task_selection.initial_config, task_selection.mount, "开始所选任务"),
    "statistics": _STATISTICS,
    "statistics-en": replace(_STATISTICS, ready_pattern="Interactive HTML report", language="en"),
}
