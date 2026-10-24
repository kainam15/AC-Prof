"""Slash-command dispatch: map text input onto existing App actions."""
from __future__ import annotations

from typing import TYPE_CHECKING

from textual import on
from textual.message_pump import MessagePump
from textual.widgets import Input as TextualInput

from acprof.experiment import RunConfigError
from acprof.tui.commands import parse_slash_command
from acprof.tui.i18n import message_id
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.log import SelectableLog

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


class SlashCommandActions(MessagePump):
    @on(Input.Submitted, "#slash-command")
    def slash_command_submitted(self, event: TextualInput.Submitted) -> None:
        if TYPE_CHECKING:
            assert isinstance(self, AcprofTui)
        value = event.value
        event.input.value = ""
        try:
            command, args = parse_slash_command(value)
        except RunConfigError as exc:
            self._show_config_error(exc)
            return

        if command == "run":
            self.action_request_run()
        elif command == "probe":
            self.action_request_probe()
        elif command == "check":
            self.action_quick_check()
        elif command == "cancel" and self._read_jobs:
            self._cancel_result_reads()
        elif command in {"stop", "cancel"}:
            self.action_request_stop()
        elif command == "status":
            snapshot = self._latest_snapshot
            self.query_one("#run-log", SelectableLog).write(
                f"[TUI] status={self.tr(snapshot.stage.label)}; "
                f"case={snapshot.completed_cases}/{snapshot.total_cases}; "
                f"resource=CPU {snapshot.cpu}, MEM {snapshot.mem}GB, GPU {snapshot.gpu}; "
                f"warnings={snapshot.warnings}; errors={snapshot.errors}"
            )
            self._activate_tab("monitor-tab")
        elif command == "smoke":
            self.preset_smoke()
        elif command == "main":
            self.preset_main()
        elif command == "preview":
            self._refresh_command_preview()
            self._activate_tab("run-tab")
        elif command == "plot":
            self._launch_plot(args[0] if args else None)
        elif command == "stats":
            self._launch_stats(args[0] if args else None)
        elif command == "report":
            self._open_report(args[0] if args else None)
        elif command == "images":
            self.action_show_images()
        elif command == "profile":
            self._launch_profile(
                dry_run=True,
                result_dir=args[0] if args else None,
                tools=args[1] if len(args) > 1 else None,
            )
        elif command in {"profile-run", "profile!"}:
            self._request_profile_run(
                result_dir=args[0] if args else None,
                tools=args[1] if len(args) > 1 else None,
            )
        elif command in {"results", "summary"}:
            path = args[0] if args else self._input("result-csv")
            self._update_result_summary(path)
            self._activate_tab("plot-tab")
        elif command in {"log", "logs"}:
            self.action_toggle_log_view()
        elif command == "clear":
            self.action_clear_log()
        elif command == "settings":
            self.action_show_settings()
        elif command == "help":
            self.query_one("#run-log", SelectableLog).write(
                self.tr(message_id('ui.0818'))
            )
            self._activate_tab("monitor-tab")
        elif command in {"quit", "exit"}:
            self.action_request_quit()
        else:
            self.notify(message_id('ui.0819', command), severity="error")
