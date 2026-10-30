"""Explicit posthoc profiling of an existing experiment."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.widgets import Button, Static

from acprof.experiments.configuration import RunConfigError
from acprof.messages import Message, join_messages
from acprof.tui.commands import PendingLaunch, format_command, prepare_profile, resolve_result_path
from acprof.tui.i18n import message_id
from acprof.tui.profile_screen import ProfilerChoiceScreen
from acprof.tui.profile_status import ProfilerOverview
from acprof.tui.views import ConfirmActionScreen

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


class ProfileController:
    """Own posthoc decisions; the App owns all subprocess lifecycle."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app
        self.overview: ProfilerOverview | None = None

    def reset_overview(self) -> None:
        self.overview = None
        self.app.localization.set_text(
            self.app.query_one("#profile-status", Static), message_id("profiling.overview.select_experiment"))
        self.update_action_availability()

    def show_overview(self, overview: ProfilerOverview | None, error: str = "") -> None:
        self.overview = overview
        if overview is None:
            status: Message = message_id("profiling.overview.unavailable", error) if error else message_id("profiling.overview.select_experiment")
        else:
            from acprof.tui.profile_screen import tool_status_message

            summary = (
                message_id("profiling.overview.complete_summary", overview.complete, overview.applicable)
                if not overview.pending else
                message_id("profiling.overview.available_summary", overview.complete, overview.applicable, overview.pending)
            )
            status = join_messages("\n", (
                summary,
                *(message_id("profiling.overview.tool_status", item.tool, tool_status_message(item))
                  for item in overview.tools),
            ))
        self.app.localization.set_text(self.app.query_one("#profile-status", Static), status)
        self.update_action_availability()

    def update_action_availability(self) -> None:
        if self.app.is_running:
            self.app.query_one("#open-result-profiling", Button).disabled = (
                self.overview is None or not self.overview.pending
                or not self.app.operation_state().allows("profile")
            )

    def open_selection_dialog(self) -> None:
        overview = self.overview
        if overview is None or not overview.pending:
            self.app.notify(message_id("profiling.backfill.complete_experiment_required"), severity="warning")
            return
        if not self.app.allow_operation("profile"):
            return
        self.app.push_screen(
            ProfilerChoiceScreen(overview),
            lambda choice: self._selection_finished(overview, choice),
        )

    def _selection_finished(
        self, overview: ProfilerOverview,
        choice: tuple[str, tuple[str, ...]] | None,
    ) -> None:
        if choice is None:
            return
        mode, tools = choice
        if not tools or not self.app.allow_operation("profile"):
            return
        selected = ",".join(tools)
        if mode == "dry-run":
            self.launch_profile(dry_run=True, result_dir=str(overview.result_dir), tools=selected)
        elif mode == "run":
            self.request_profile_run(result_dir=str(overview.result_dir), tools=selected)

    def profile_command(
        self, *, dry_run: bool, result_dir: str | None = None, tools: str | None = None,
    ) -> tuple[list[str], Path] | None:
        source = result_dir or self.app.configuration.input("result-csv")
        if not source:
            self.app.notify("请先选择实验并读取摘要", severity="warning")
            return None
        path = resolve_result_path(source, self.app.project_dir)
        if path.name == "result_layers.json":
            path = path.parent
        selected = tools
        if selected is None:
            if self.overview is not None and path.resolve() == self.overview.result_dir.resolve():
                selected = ",".join(self.overview.suggested_tools)
            if not selected:
                self.app.notify("请在实验结果中选择补充分析工具，或在命令中指定工具", severity="warning")
                return None
        try:
            pending = prepare_profile(str(path), tools=selected, dry_run=dry_run,
                                      project_dir=self.app.project_dir,
                                      python_executable=self.app.python_executable)
            return list(pending.command), Path(pending.result_dir)
        except (FileNotFoundError, RunConfigError) as exc:
            self.app.notify(str(exc), severity="error")
            return None

    def launch_profile(
        self, *, dry_run: bool, result_dir: str | None = None, tools: str | None = None,
    ) -> None:
        if not self.app.allow_operation("profile"):
            return
        prepared = self.profile_command(dry_run=dry_run, result_dir=result_dir, tools=tools)
        if prepared is not None:
            command, result_path = prepared
            self.app.launch(PendingLaunch(
                tuple(command), "profile-dry-run" if dry_run else "profile",
                result_dir=str(result_path),
            ))

    def request_profile_run(
        self, result_dir: str | None = None, tools: str | None = None,
    ) -> None:
        if not self.app.allow_operation("profile"):
            return
        prepared = self.profile_command(dry_run=False, result_dir=result_dir, tools=tools)
        if prepared is None:
            return
        command, result_path = prepared
        self.app.experiments.pending_launch = PendingLaunch(
            tuple(command), "profile", result_dir=str(result_path))
        self.app.push_screen(
            ConfirmActionScreen(
                "执行 profiler 补采？",
                join_messages("", (
                    message_id("profiling.backfill.confirm_details"),
                    format_command(command),
                )),
                "执行补采",
                variant="warning",
            ),
            self.app.experiments.confirmed_launch,
        )
