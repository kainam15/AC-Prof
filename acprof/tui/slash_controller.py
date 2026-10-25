"""Slash-command dispatch: map text input onto existing App actions."""
from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import Input as TextualInput

from acprof.experiment import RunConfigError
from acprof.tui.commands import parse_slash_command
from acprof.tui.i18n import message_id
from acprof.tui.log import SelectableLog

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


class SlashController:
    """Own slash state and operations; the App owns lifecycle and scheduling."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app

    def slash_command_submitted(self, event: TextualInput.Submitted) -> None:
        value = event.value
        event.input.value = ""
        try:
            command, args = parse_slash_command(value)
        except RunConfigError as exc:
            self.app.configuration.show_config_error(exc)
            return

        if command == "run":
            self.app.experiments.request_run()
        elif command == "probe":
            self.app.experiments.request_probe()
        elif command == "check":
            self.app.preflight.quick_check()
        elif command == "cancel" and self.app.read_jobs:
            self.app.results.cancel_result_reads()
        elif command in {"stop", "cancel"}:
            self.app.action_request_stop()
        elif command == "status":
            snapshot = self.app.latest_snapshot
            self.app.query_one("#run-log", SelectableLog).write(
                f"[TUI] status={self.app.localization.tr(snapshot.stage.label)}; "
                f"case={snapshot.completed_cases}/{snapshot.total_cases}; "
                f"resource=CPU {snapshot.cpu}, MEM {snapshot.mem}GB, GPU {snapshot.gpu}; "
                f"warnings={snapshot.warnings}; errors={snapshot.errors}"
            )
            self.app.activate_tab("monitor-tab")
        elif command == "smoke":
            self.app.experiments.preset_smoke()
        elif command == "main":
            self.app.experiments.preset_main()
        elif command == "preview":
            self.app.configuration.refresh_command_preview()
            self.app.activate_tab("run-tab")
        elif command == "plot":
            self.app.results.launch_plot(args[0] if args else None)
        elif command == "stats":
            self.app.results.launch_stats(args[0] if args else None)
        elif command == "report":
            self.app.results.open_report(args[0] if args else None)
        elif command == "images":
            self.app.images.show_images()
        elif command == "profile":
            self.app.profiles.launch_profile(
                dry_run=True,
                result_dir=args[0] if args else None,
                tools=args[1] if len(args) > 1 else None,
            )
        elif command in {"profile-run", "profile!"}:
            self.app.profiles.request_profile_run(
                result_dir=args[0] if args else None,
                tools=args[1] if len(args) > 1 else None,
            )
        elif command in {"results", "summary"}:
            path = args[0] if args else self.app.configuration.input("result-csv")
            self.app.results.update_result_summary(path)
            self.app.activate_tab("plot-tab")
        elif command in {"log", "logs"}:
            self.app.configuration.toggle_log_view()
        elif command == "clear":
            self.app.results.clear_log()
        elif command == "settings":
            self.app.configuration.show_settings()
        elif command == "help":
            self.app.query_one("#run-log", SelectableLog).write(
                self.app.localization.tr(message_id('ui.0818'))
            )
            self.app.activate_tab("monitor-tab")
        elif command in {"quit", "exit"}:
            self.app.action_request_quit()
        else:
            self.app.notify(message_id('ui.0819', command), severity="error")
