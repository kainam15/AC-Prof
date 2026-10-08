"""Confirm a proposed experiment or probe without taking process ownership.

The parent App remains responsible for starting, stopping and reaping processes,
as well as the strict measurement-window lifecycle.
"""
from __future__ import annotations

from dataclasses import replace

from textual import on
from textual.message_pump import MessagePump
from textual.widgets import Button, Select, Static

from acprof.experiment import RunConfigError, build_run_command
from acprof.messages import join_messages, message
from acprof.platform import collection_policy_error
from acprof.tui.commands import PendingLaunch, build_probe_command, format_command
from acprof.tui.i18n import error_message
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.views import ConfirmActionScreen


class ExperimentActions(MessagePump):
    def preset_smoke(self) -> None:
        self._apply_preset("smoke")

    def preset_main(self) -> None:
        self._apply_preset("main")

    def _apply_preset(self, preset: str) -> None:
        if not self._allow_operation("configure"):
            return
        try:
            current = self._collect_config(allow_empty_model=True)
            config = current.with_preset(preset)
        except RunConfigError as exc:
            self._show_config_error(exc)
            with self.prevent(Select.Changed):
                self.query_one("#run-preset", Select).value = "custom"
            return
        self._apply_config(config, preset=preset)
        self.notify("已应用基础 CPU Smoke 预设" if preset == "smoke" else "已应用主矩阵预设（分析器关闭）", timeout=3)

    @on(Button.Pressed, "#start-run")
    def start_run_button(self) -> None:
        self.action_request_run()

    @on(Button.Pressed, "#probe-largest")
    def probe_largest_button(self) -> None:
        self.action_request_probe()

    def action_request_probe(self) -> None:
        if not self._allow_operation("probe"):
            return
        try:
            config = self._collect_config()
            command = build_probe_command(
                config,
                project_dir=self._project_dir,
                python_executable=self._python_executable,
            )
        except RunConfigError as exc:
            self._show_config_error(exc)
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
        scale_text = f"{largest_scale:g}" if largest_scale is not None else message("自动规划后的最大值")
        memory_text = ",".join(
            f"{value}GB" for value in memory_candidates
        )
        preview = format_command(command)
        self._pending_launch = PendingLaunch(tuple(command), "probe", config)
        self.push_screen(
            ConfirmActionScreen(
                "探测最低配置的最大输入？",
                join_messages("", (
                    message(
                        "资源：CPU={0}、GPU={1}\n内存候选：{2}（从小到大）\n输入规模：{3}\n\n"
                        "每档使用全新容器并最多执行一次最大输入请求；OOM 时自动尝试下一档，"
                        "第一个成功值就是最低可用内存。结果单独写入 "
                        "probes/，不会写入或修改正式实验 CSV。最大输入请求不设超时，"
                        "可用 /stop 手动终止。\n\n",
                        cpu, gpu, memory_text, scale_text,
                    ),
                    preview,
                )),
                "开始探测",
            ),
            self._confirmed_launch,
        )

    def action_request_run(self) -> None:
        if not self._allow_collection() or not self._allow_operation("run"):
            return
        try:
            config = self._collect_config()
            environment = self._current_environment()
            error = collection_policy_error(environment, profiling_mode=config.profiling_mode,
                                            compute_tool=config.compute_profile_tool,
                                            execution_tool=config.execution_profile_tool)
            if environment.environment == "wsl2" and error:
                self.notify(error, severity="error", timeout=10)
                return
            command = build_run_command(
                config,
                project_dir=self._project_dir,
                python_executable=self._python_executable,
            )
        except RunConfigError as exc:
            self._show_config_error(exc)
            return
        preview = format_command(command)
        self._set_text(self.query_one('#command-preview', Static), preview)
        self._review_run_destination(PendingLaunch(tuple(command), "run", config))

    def _confirmed_launch(self, confirmed: bool | None) -> None:
        pending = self._pending_launch
        self._pending_launch = None
        self._sync_image_refresh_timer()
        if not confirmed or pending is None:
            return
        self._launch(pending)

    def _remember_last_used(
        self, *, model: str = "", result_dir: str = "", result_csv: str = "",
    ) -> None:
        """Remember confirmed inputs, preserving explicitly saved preferences."""
        updates = {}
        report_source = self.query_one("#report-source", Input)
        if result_csv and report_source.value in {"", self._saved_settings.last_result_csv}:
            report_source.value = result_csv
        if model.strip():
            updates["last_model"] = model.strip()
        for widget_id, value in (("result-dir", result_dir), ("result-csv", result_csv)):
            if value:
                self.query_one(f"#{widget_id}", Input).value = value
                updates[f"last_{widget_id.replace('-', '_')}"] = value
        settings = replace(self._saved_settings, **updates)
        if settings == self._saved_settings:
            return
        if self._settings_warning:
            self.notify(
                "本地设置无法读取，已保留原文件。可在设置页主动保存后恢复自动记忆。",
                title="自动记忆未保存", severity="warning",
            )
            return
        try:
            self._save_local_settings(self.settings_path, settings, self._project_dir)
        except (OSError, ValueError, RunConfigError) as exc:
            self.notify(error_message(exc), title="自动记忆未保存", severity="warning")
            return
        self._saved_settings = settings
