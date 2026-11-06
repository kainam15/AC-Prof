"""Explicit posthoc profiling of an existing experiment."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

from textual.widgets import Button, Static

from acprof.experiments.configuration import RunConfigError
from acprof.messages import Message, join_messages
from acprof.tui.commands import PendingLaunch, format_command, prepare_profile, resolve_result_path
from acprof.tui.i18n import error_message, message_id
from acprof.tui.results.data_status import ExperimentDataStatus, inspect_experiment_data
from acprof.tui.results.profile_screen import ProfilerChoiceScreen
from acprof.tui.results.profile_status import ProfilerOverview
from acprof.tui.ui.dialogs import ConfirmActionScreen

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


class ProfileController:
    """Own posthoc decisions; the App owns all subprocess lifecycle."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app
        self.overview: ProfilerOverview | None = None
        self.validation_request: Event | None = None

    def cancel_validation(self) -> None:
        if self.validation_request is not None:
            self.validation_request.set()
            self.validation_request = None

    def reset_overview(self, status: str = "") -> None:
        self.cancel_validation()
        self.overview = None
        self.app.localization.set_text(
            self.app.query_one("#profile-status", Static), status or message_id("profiling.overview.select_experiment"))
        self.app.localization.set_text(
            self.app.query_one("#result-profile-status", Static), status or message_id("profiling.overview.select_experiment"))
        self.update_action_availability()

    def show_overview(self, overview: ProfilerOverview | None, error: str = "") -> None:
        self.overview = overview
        if overview is None:
            status: Message = message_id("profiling.overview.unavailable", error) if error else message_id("profiling.overview.select_experiment")
        else:
            from acprof.tui.results.profile_screen import tool_status_message

            summary = (
                message_id("results.overview.no_tools") if not overview.applicable else
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
        self.app.localization.set_text(self.app.query_one("#result-profile-status", Static), summary if overview else status)
        self.update_action_availability()

    def update_action_availability(self) -> None:
        if self.app.is_running:
            disabled = (
                self.overview is None or not self.overview.pending
                or not self.app.operation_state().allows("profile")
            )
            for selector in ("#open-result-profiling", "#overview-profiling"):
                self.app.query_one(selector, Button).disabled = disabled

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
        if overview is not self.overview:
            self.app.notify(message_id("results.overview.stale_profile"), severity="warning")
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
        source = result_dir or self.app.configuration.input("experiment-source")
        if not source:
            self.app.notify(message_id("profiling.overview.select_experiment"), severity="warning")
            return None
        path = resolve_result_path(source, self.app.project_dir)
        if path.name == "result_layers.json":
            path = path.parent
        selected = tools
        if selected is None:
            if self.overview is not None and path.resolve() == self.overview.result_dir.resolve():
                selected = ",".join(self.overview.suggested_tools)
            if not selected:
                self.app.notify(message_id("profiling.backfill.choose_tools"), severity="warning")
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
            self.validate_launch(PendingLaunch(
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
            partial(self.confirmed_profile, self.app.experiments.pending_launch,
                    self.app.results.data_status, self.app.configuration.input("experiment-source")),
        )

    def confirmed_profile(self, pending: PendingLaunch, expected: ExperimentDataStatus | None,
                          selection: str, confirmed: bool | None) -> None:
        self.app.experiments.pending_launch = None
        if not confirmed or not self.app.allow_operation("profile"):
            return
        if selection != self.app.configuration.input("experiment-source") or expected is not self.app.results.data_status:
            self.reset_overview(message_id("results.overview.stale_profile"))
            return
        self.validate_launch(pending)

    def validate_launch(self, pending: PendingLaunch) -> None:
        """Recheck provenance and permissions after the user's explicit action."""
        expected = self.app.results.data_status
        selection = self.app.configuration.input("experiment-source")
        token = self.validation_request = Event()
        self.app.read_jobs.add(token)
        self.app.set_busy(True)
        self.app.run_worker(
            partial(self._validate_worker, token, pending, expected, selection),
            thread=True, group="profile-validation", exit_on_error=False,
        )

    def _validate_worker(self, token: Event, pending: PendingLaunch,
                         expected: ExperimentDataStatus | None, selection: str) -> None:
        try:
            with self.app.results.offline_reads.read():
                current = inspect_experiment_data(Path(pending.result_dir), cancelled=token.is_set)
                if current.profiler_overview is None:
                    raise ValueError(current.profiler_error)
                selected = pending.command[pending.command.index("--tools") + 1].split(",")
                eligible = {item.tool for item in current.profiler_overview.tools if item.needs_collection}
                if not set(selected) <= eligible:
                    raise ValueError(message_id("results.overview.stale_profile"))
                if expected is not None and expected.path.parent == Path(pending.result_dir).resolve():
                    if (expected.sha256, expected.metadata_identity) != (current.sha256, current.metadata_identity):
                        raise ValueError(message_id("results.overview.stale_profile"))
            error = ""
        except Exception as exc:
            error = error_message(exc)
        self.app.process.safe_process_callback(self.validation_finished, token, pending, selection, error)

    def validation_finished(self, token: Event, pending: PendingLaunch, selection: str, error: str) -> None:
        self.app.read_jobs.discard(token)
        if not self.app.is_running or self.app.ui_closing:
            return
        self.app.set_busy(self.app.is_busy())
        if token is not self.validation_request or token.is_set():
            return
        self.validation_request = None
        if error or selection != self.app.configuration.input("experiment-source"):
            self.reset_overview(error or message_id("results.overview.stale_profile"))
            return
        if self.app.allow_operation("profile"):
            self.app.launch(pending)
