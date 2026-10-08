"""Slash-command dispatch: map text input onto existing App actions."""
from __future__ import annotations

from textual import on
from textual.message_pump import MessagePump

from acprof.experiment import RunConfigError
from acprof.messages import message
from acprof.tui.commands import parse_slash_command
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.log import SelectableLog


class SlashCommandActions(MessagePump):
    @on(Input.Submitted, "#slash-command")
    def slash_command_submitted(self, event: Input.Submitted) -> None:
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
                f"[TUI] status={self.tr(snapshot.stage)}; "
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
                self.tr("[TUI] /run 采集 · /probe 最大输入探测 · /check 环境检查 · "
                "/status 状态 · /stop 终止 · "
                "/smoke 最小预设 · /main 主矩阵 · /preview 命令预览 · "
                "/plot [csv] 绘图 · /profile [dir] [tools] 补采计划 · "
                "/profile-run [dir] [tools] 执行补采 · /results [csv] 摘要 · "
                "/stats [csv/dir] 统计 · /report [json] 报告 · /images 镜像管理 · "
                "/settings 应用设置 · /log 放大日志 · /clear 清日志 · /quit 退出")
            )
            self._activate_tab("monitor-tab")
        elif command in {"quit", "exit"}:
            self.action_request_quit()
        else:
            self.notify(message('未知快捷命令：/{0}', command), severity="error")
