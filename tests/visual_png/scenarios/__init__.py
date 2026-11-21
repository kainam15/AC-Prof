"""Test-only deterministic TUI scenes shared by the fixture app and VHS launcher."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

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
    visible_controls: tuple[str, ...] = ()


# Regexes must match actual visible terminal cells, not a background sentinel.
_FORM = Scene(form.initial_config, form.mount, "采集计划 · Docker · 1 case")
_DOCKER_FRONTEND = Scene(
    docker_frontend.initial_config, docker_frontend.mount, r"registry-1\.docker\.io", page="monitor-tab",
    visible_controls=("#download-progress", "#stop-run"),
)
_STATISTICS = Scene(
    statistics.initial_config, statistics.mount, "选择实验后可查看分析报告", columns=80, rows=24, page="results-tab",
    visible_controls=("#experiment-source", "#choose-experiment-source", "#html-report-status"),
)
SCENARIOS: dict[str, Scene] = {
    "form": _FORM,
    "form-small": replace(_FORM, columns=80, rows=24, visible_controls=("#model", "#start-run")),
    "error-modal": Scene(
        error_modal.initial_config, partial(error_modal.mount, inspect=True), "Permission denied reading the requested DRAM",
        visible_controls=("#preflight-inspection", "#preflight-callout .structured-action"),
    ),
    "near-match": Scene(near_match.initial_config, near_match.mount, "采集计划 · Docker · 9 case"),
    "measuring": Scene(measuring.initial_config, measuring.mount, r"当前 case\s+2", page="monitor-tab"),
    "docker-frontend": _DOCKER_FRONTEND,
    "docker-frontend-small": replace(_DOCKER_FRONTEND, columns=80, rows=24),
    "docker-frontend-en": replace(_DOCKER_FRONTEND, language="en"),
    "docker-frontend-en-small": replace(_DOCKER_FRONTEND, columns=80, rows=24, language="en"),
    "task-selection": Scene(
        task_selection.initial_config, task_selection.mount, "开始所选任务",
        visible_controls=("#preparation-cancel", "#preparation-select-all", "#preparation-apply"),
    ),
    "statistics": _STATISTICS,
    "statistics-en": replace(_STATISTICS, ready_pattern="Select an experiment to view its report", language="en"),
}
