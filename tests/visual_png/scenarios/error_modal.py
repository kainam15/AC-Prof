"""Cached environment overlay with a blocking diagnostic; no live checks."""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from acprof.tui.run.diagnostics import PreflightCheck
from tests.visual_png.scenarios.common import apply_fixture_preflight, freeze_for_screenshot


def initial_config() -> RunConfig:
    return replace(RunConfig.smoke("fixture/model"), runtime="docker", profiling_mode="full")


def mount(app: AcprofTui) -> None:
    freeze_for_screenshot(app)
    apply_fixture_preflight(app)
    app.preflight.checks = (
        PreflightCheck("Runtime", "ok", "Docker", runtime_kind="docker", check_id="runtime"),
        PreflightCheck("perf instructions", "fail", "Permission denied: cross-user process access is unavailable.",
                       "permission_denied", check_id="cpu_instructions", source="perf_stat"),
    )
    app.preflight.completed_at = datetime(2026, 11, 7, 8, 0, tzinfo=UTC)
    app.preflight.lock_acquired = True
    app.preflight.refresh_preflight_state()
    app.preflight.panel.show(pin=True)
