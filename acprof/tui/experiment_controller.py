"""Confirm a proposed experiment or probe without taking process ownership.

The parent App remains responsible for starting, stopping and reaping processes,
as well as the strict measurement-window lifecycle.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import Select, Static

from acprof import platform
from acprof.experiment import RunConfigError, build_run_command
from acprof.messages import join_messages
from acprof.platform import collection_policy_error
from acprof.tui.commands import PendingLaunch, build_probe_command, format_command
from acprof.tui.i18n import message_id
from acprof.tui.views import ConfirmActionScreen

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui

class ExperimentController:
    """Own experiments state and operations; the App owns lifecycle and scheduling."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app
        self.pending_launch: PendingLaunch | None = None

    def preset_smoke(self) -> None:
        self.apply_preset("smoke")

    def preset_main(self) -> None:
        self.apply_preset("main")

    def apply_preset(self, preset: str) -> None:
        if not self.app.allow_operation("configure"):
            return
        try:
            current = self.app.configuration.collect_config(allow_empty_model=True)
            config = current.with_preset(preset)
        except RunConfigError as exc:
            self.app.configuration.show_config_error(exc)
            with self.app.prevent(Select.Changed):
                self.app.query_one("#run-preset", Select).value = "custom"
            return
        self.app.configuration.apply_config(config, preset=preset)
        self.app.notify("已应用基础 CPU Smoke 预设" if preset == "smoke" else "已应用主矩阵预设（分析器关闭）", timeout=3)

    def start_run_button(self) -> None:
        self.request_run()

    def probe_largest_button(self) -> None:
        self.request_probe()

    def request_probe(self) -> None:
        if not self.app.allow_operation("probe"):
            return
        try:
            config = self.app.configuration.collect_config()
            command = build_probe_command(
                config,
                project_dir=self.app.project_dir,
                python_executable=self.app.python_executable,
            )
        except RunConfigError as exc:
            self.app.configuration.show_config_error(exc)
            return

        cpu = min(int(value) for value in config.cpus.split(","))
        memory_candidates = sorted(
            set(int(value) for value in config.mems.split(","))
        )
        gpu_modes = config.gpus.split(",")
        gpu = "off" if "off" in gpu_modes else "on"
        largest_scale = (
            max(float(value) for value in config.input_scales.split(","))
            if config.input_scales
            else None
        )
        scale_text = f"{largest_scale:g}" if largest_scale is not None else message_id('ui.0756')
        memory_text = ",".join(
            f"{value}GB" for value in memory_candidates
        )
        preview = format_command(command)
        self.pending_launch = PendingLaunch(tuple(command), "probe", config)
        self.app.push_screen(
            ConfirmActionScreen(
                "探测最低配置的最大输入？",
                join_messages("", (
                    message_id(
                        'ui.0758',
                        cpu, gpu, memory_text, scale_text,
                    ),
                    preview,
                )),
                "开始探测",
            ),
            self.confirmed_launch,
        )

    def request_run(self) -> None:
        if not self.app.preflight.allow_collection() or not self.app.allow_operation("run"):
            return
        try:
            config = self.app.configuration.collect_config()
            environment = platform.detect_environment()
            error = collection_policy_error(environment, profiling_mode=config.profiling_mode,
                                            compute_tool=config.compute_profile_tool,
                                            execution_tool=config.execution_profile_tool)
            if environment.environment == "wsl2" and error:
                self.app.notify(error, severity="error", timeout=10)
                return
            command = build_run_command(
                config,
                project_dir=self.app.project_dir,
                python_executable=self.app.python_executable,
            )
        except RunConfigError as exc:
            self.app.configuration.show_config_error(exc)
            return
        preview = format_command(command)
        self.app.localization.set_text(self.app.query_one('#command-preview', Static), preview)
        self.app.recovery.review_run_destination(PendingLaunch(tuple(command), "run", config))

    def confirmed_launch(self, confirmed: bool | None) -> None:
        pending = self.pending_launch
        self.pending_launch = None
        self.app.images.sync_image_refresh_timer()
        if not confirmed or pending is None:
            return
        self.app.launch(pending)
