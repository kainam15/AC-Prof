"""Cached environment overlay with a blocking diagnostic; no live checks."""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.run.environment_checks import ENVIRONMENT_CHECK_IDS
from tests.visual_png.scenarios.common import apply_fixture_preflight, freeze_for_screenshot


def initial_config() -> RunConfig:
    return replace(RunConfig.smoke("fixture/model"), runtime="docker", idle_seconds=1, extra_options={"dram_energy": "required"})


def mount(app: AcprofTui, *, group: str = "error") -> None:
    freeze_for_screenshot(app)
    apply_fixture_preflight(app)
    app.preflight.checks = (
        PreflightCheck("Docker", "ok", "Docker daemon accessible", runtime_kind="docker", check_id="docker"),
        PreflightCheck("Required DRAM energy", "fail", "Permission denied reading the requested DRAM counter.",
                       "permission_denied", check_id="dram_energy", source="rapl_topology"),
    )
    app.preflight.environment_checks = (
        *app.preflight.checks,
        *(PreflightCheck(name, "pending" if name in {"torch", "massif"} else "ok",
                        "Runtime execution unverified" if name in {"torch", "massif"} else "Environment interface readable",
                        check_id=name, source="fixture") for name in ENVIRONMENT_CHECK_IDS if name not in {"docker", "dram_energy"}),
    )
    app.preflight.completed_at = datetime(2026, 11, 7, 8, 0, tzinfo=UTC)
    app.preflight.lock_acquired = True
    app.preflight.refresh_preflight_state()
    app.preflight.panel.show(pin=True, group=group)
    identifier = "docker" if group == "success" else "dram_energy"
    app.preflight.panel.call_later(
        lambda: app.query_one("#preflight-signal-" + identifier).focus(scroll_visible=False)
    )
