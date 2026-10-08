"""Post-hoc profiling requests and explicit confirmation of profiler writes."""
from __future__ import annotations

from pathlib import Path

from textual import on
from textual.message_pump import MessagePump
from textual.widgets import Button

from acprof.experiment import RunConfigError
from acprof.messages import join_messages, message
from acprof.tui.commands import PendingLaunch, format_command, prepare_profile
from acprof.tui.views import ConfirmActionScreen


class ProfileActions(MessagePump):
    @on(Button.Pressed, "#profile-dry-run")
    def profile_dry_run_button(self) -> None:
        self._launch_profile(dry_run=True)

    @on(Button.Pressed, "#profile-run")
    def profile_run_button(self) -> None:
        self._request_profile_run()

    def _profile_command(
        self,
        *,
        dry_run: bool,
        result_dir: str | None = None,
        tools: str | None = None,
    ) -> tuple[list[str], Path] | None:
        directory = result_dir or self._input("result-dir")
        selected_tools = tools if tools is not None else ",".join(
            checkbox.name
            for checkbox in self.query("#profile-tools Checkbox")
            if checkbox.value and checkbox.name is not None
        )
        if not directory:
            self.notify("请填写结果目录", severity="warning")
            return None
        if not selected_tools:
            self.notify("请至少勾选一个补采工具", severity="warning")
            return None
        try:
            pending = prepare_profile(directory, tools=selected_tools, dry_run=dry_run,
                                      project_dir=self._project_dir, python_executable=self._python_executable)
            return list(pending.command), Path(pending.result_dir)
        except FileNotFoundError as exc:
            self.notify(str(exc), severity="error")
            return None
        except RunConfigError as exc:
            self._show_config_error(exc)
            return None

    def _launch_profile(
        self,
        *,
        dry_run: bool,
        result_dir: str | None = None,
        tools: str | None = None,
    ) -> None:
        if not self._allow_operation("profile"):
            return
        prepared = self._profile_command(
            dry_run=dry_run,
            result_dir=result_dir,
            tools=tools,
        )
        if prepared is not None:
            command, result_path = prepared
            self._launch(PendingLaunch(
                tuple(command), "profile-dry-run" if dry_run else "profile", result_dir=str(result_path),
            ))

    def _request_profile_run(
        self,
        result_dir: str | None = None,
        tools: str | None = None,
    ) -> None:
        if not self._allow_operation("profile"):
            return
        prepared = self._profile_command(
            dry_run=False,
            result_dir=result_dir,
            tools=tools,
        )
        if prepared is None:
            return
        command, result_path = prepared
        self._pending_launch = PendingLaunch(tuple(command), "profile", result_dir=str(result_path))
        self.push_screen(
            ConfirmActionScreen(
                "执行 profiler 补采？",
                join_messages("", (
                    message(
                        "该操作会启动隔离 profiler，并在成功后原子回填现有结果。"
                        "原文件会按项目规则备份。\n\n"
                    ),
                    format_command(command),
                )),
                "执行补采",
                variant="warning",
            ),
            self._confirmed_launch,
        )
