"""Post-hoc profiling requests and explicit confirmation of profiler writes."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.widgets import Checkbox

from acprof.experiment import RunConfigError
from acprof.messages import join_messages
from acprof.tui.commands import PendingLaunch, format_command, prepare_profile
from acprof.tui.i18n import message_id
from acprof.tui.views import ConfirmActionScreen

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui

class ProfileController:
    """Own profiles state and operations; the App owns lifecycle and scheduling."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app

    def profile_dry_run_button(self) -> None:
        self.launch_profile(dry_run=True)

    def profile_run_button(self) -> None:
        self.request_profile_run()

    def profile_command(
        self,
        *,
        dry_run: bool,
        result_dir: str | None = None,
        tools: str | None = None,
    ) -> tuple[list[str], Path] | None:
        directory = result_dir or self.app.configuration.input("result-dir")
        selected_tools = tools if tools is not None else ",".join(
            checkbox.name
            for checkbox in self.app.query("#profile-tools Checkbox").results(Checkbox)
            if checkbox.value and checkbox.name is not None
        )
        if not directory:
            self.app.notify("请填写结果目录", severity="warning")
            return None
        if not selected_tools:
            self.app.notify("请至少勾选一个补采工具", severity="warning")
            return None
        try:
            pending = prepare_profile(directory, tools=selected_tools, dry_run=dry_run,
                                      project_dir=self.app.project_dir, python_executable=self.app.python_executable)
            return list(pending.command), Path(pending.result_dir)
        except FileNotFoundError as exc:
            self.app.notify(str(exc), severity="error")
            return None
        except RunConfigError as exc:
            self.app.configuration.show_config_error(exc)
            return None

    def launch_profile(
        self,
        *,
        dry_run: bool,
        result_dir: str | None = None,
        tools: str | None = None,
    ) -> None:
        if not self.app.allow_operation("profile"):
            return
        prepared = self.profile_command(
            dry_run=dry_run,
            result_dir=result_dir,
            tools=tools,
        )
        if prepared is not None:
            command, result_path = prepared
            self.app.launch(PendingLaunch(
                tuple(command), "profile-dry-run" if dry_run else "profile", result_dir=str(result_path),
            ))

    def request_profile_run(
        self,
        result_dir: str | None = None,
        tools: str | None = None,
    ) -> None:
        if not self.app.allow_operation("profile"):
            return
        prepared = self.profile_command(
            dry_run=False,
            result_dir=result_dir,
            tools=tools,
        )
        if prepared is None:
            return
        command, result_path = prepared
        self.app.experiments.pending_launch = PendingLaunch(tuple(command), "profile", result_dir=str(result_path))
        self.app.push_screen(
            ConfirmActionScreen(
                "执行 profiler 补采？",
                join_messages("", (
                    message_id(
                        'ui.0817'
                    ),
                    format_command(command),
                )),
                "执行补采",
                variant="warning",
            ),
            self.app.experiments.confirmed_launch,
        )
