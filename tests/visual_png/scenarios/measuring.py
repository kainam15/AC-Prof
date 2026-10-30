"""Simulated active measurement screen: no subprocess, GPU or perf collection."""

from __future__ import annotations

from unittest.mock import Mock

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.stages import Stage
from tests.visual_png.scenarios.common import freeze_for_screenshot


def initial_config() -> RunConfig:
    return RunConfig.smoke("fixture/model")


def mount(app: AcprofTui) -> None:
    process = Mock(pid=12345, returncode=None)
    process.poll.return_value = None
    app.lifecycle.process = process
    app.process_kind = "run"
    app.active_command = ("acprof", "run")
    snapshot = ProgressSnapshot(
        stage=Stage.MEASURING,
        detail="固定测量窗口 · 中文边框、状态与日志排版",
        current_case=2,
        total_cases=4,
        cpu="2",
        mem="4",
        gpu="off",
        measurement_active=True,
        interface_status="passed",
        runtime_status="passed",
        measurement_status="running",
    )
    app.activate_tab("monitor-tab")
    app.consume_process_line("正在执行固定测试任务：不启动真实推理。", snapshot, True)
    app.logs.flush()
    app.started_monotonic = 0
    freeze_for_screenshot(app)
