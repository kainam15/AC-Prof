"""A silent Dockerfile frontend resolution, with no inferred download bytes."""

from __future__ import annotations

from unittest.mock import Mock

from acprof.events.image import ImageProgress
from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from acprof.tui.i18n import message_id
from acprof.tui.run.progress import ProgressSnapshot
from acprof.tui.run.stages import Stage
from tests.visual_png.scenarios.common import freeze_for_screenshot


def initial_config() -> RunConfig:
    return RunConfig.smoke("fixture/model")


def mount(app: AcprofTui) -> None:
    process = Mock(pid=None, returncode=None)
    process.poll.return_value = None
    app.execution.lifecycle.process = process
    app.execution.session.kind = "run"
    app.execution.session.command = ("acprof", "run")
    snapshot = ProgressSnapshot(
        stage=Stage.PREPARING_ENVIRONMENT,
        detail=message_id("run.docker.activity.frontend"),
        interface_status="passed",
    )
    progress = ImageProgress(
        "platform", "acprof-platform-cpu:fixture", "building",
        "resolve image config for docker-image://docker.io/docker/dockerfile:1",
        activity="frontend", operation_id="fixture-frontend", transfer_status="unavailable",
        source_image="docker.io/docker/dockerfile:1", source_registry="docker.io",
        source_domain="registry-1.docker.io", source_kind="official", source_category="docker_hub",
        network_owner="buildkit", elapsed_seconds=58, idle_seconds=58,
        uncovered_sources=("metadata", "protocol_overhead", "buildkit_layers", "unreported_build"),
    )
    app.activate_tab("monitor-tab")
    app.consume_process_line("", snapshot, True)
    app.set_busy(True)
    app.preflight.download_progress(progress)
    # No elapsed timer is created; the 58-second silent state is deterministic.
    app.execution.session.started_monotonic = 0
    freeze_for_screenshot(app)
