"""Bilingual messages for logs; keys name stable display purposes."""
from __future__ import annotations

from .entry import Entry

MESSAGES: dict[str, Entry] = {
    "logs.action.clear": Entry(zh="清空日志", en="Clear log"),
    "logs.action.expand": Entry(zh="放大日志", en="Expand log"),
    "logs.capture.pipe_mode": Entry(
        zh="[TUI] 子进程输出通过管道读取；tmux pane 捕获已对该子进程禁用。",
        en="[TUI] Reading subprocess output through a pipe; tmux pane capture is disabled for this subprocess.",
    ),
    "logs.clipboard.all_copied": Entry(zh="已复制当前日志视图的全部内容", en="Copied all text in the current log view"),
    "logs.clipboard.copy_all": Entry(zh="复制全部", en="Copy all"),
    "logs.directory.home": Entry(zh="主目录", en="Home"),
    "logs.directory.loading": Entry(zh="正在读取目录……", en="Reading directory…"),
    "logs.directory.navigation_hint": Entry(
        zh="Enter 打开所选目录；“使用此目录”返回保存窗口。",
        en="Enter opens a directory; Use directory returns to the save dialog.",
    ),
    "logs.directory.parent": Entry(zh="上一级", en="Parent"),
    "logs.directory.select": Entry(zh="使用此目录", en="Use directory"),
    "logs.directory.title": Entry(zh="选择主机上的目录", en="Choose a directory on the host"),
    "logs.export.absolute_directory_required": Entry(zh="请输入绝对目录或以 ~ 开头的目录。", en="Enter an absolute directory or a directory starting with ~."),
    "logs.export.browse": Entry(zh="浏览目录", en="Browse"),
    "logs.export.completed": Entry(zh="日志已导出", en="Logs exported"),
    "logs.export.copy_path": Entry(zh="复制路径", en="Copy path"),
    "logs.export.directory_label": Entry(zh="保存目录", en="Save directory"),
    "logs.export.disk_full": Entry(zh="磁盘空间不足：{0}", en="Not enough disk space: {0}"),
    "logs.export.failed": Entry(zh="日志导出失败：{0}", en="Log export failed: {0}"),
    "logs.export.filename_label": Entry(zh="文件名", en="Filename"),
    "logs.export.host_location": Entry(
        zh="保存到运行 AC-Prof 的主机：{0}（SSH 时为远程主机）。",
        en="Save on the host running AC-Prof: {0} (the remote host over SSH).",
    ),
    "logs.export.invalid_filename": Entry(zh="请输入文件名，不要包含目录分隔符。", en="Enter a filename without directory separators."),
    "logs.export.path_copied": Entry(zh="路径已复制", en="Path copied"),
    "logs.export.permission_denied": Entry(zh="权限不足：{0}", en="Permission denied: {0}"),
    "logs.export.save": Entry(zh="保存日志", en="Save logs"),
    "logs.export.save_hint": Entry(
        zh="确认后创建缺失目录；同名文件自动加序号。Enter 保存，Esc 取消。",
        en="Missing directories are created on save; duplicate names get a suffix. Enter saves, Esc cancels.",
    ),
    "logs.export.title": Entry(zh="导出日志", en="Export logs"),
    "logs.export.view_path": Entry(zh="查看路径", en="View path"),
    "logs.filter.all_levels": Entry(zh="ALL", en="ALL"),
    "logs.filter.toggle": Entry(zh="筛选 / 搜索", en="Filter / Find"),
    "logs.measurement.deferred_heading": Entry(
        zh="[TUI] 测量窗口结束，显示期间延迟刷新的重要消息：",
        en="[TUI] Measurement window ended; showing deferred important messages:",
    ),
    "logs.measurement.suppressed_count": Entry(
        zh="[TUI] 为降低测量干扰，本窗口隐藏了 {0} 行常规输出。",
        en="[TUI] Hid {0} routine output lines in this window to reduce measurement interference.",
    ),
    "logs.navigation.follow_tail": Entry(zh="回到底部", en="Follow"),
    "logs.navigation.restore_monitor": Entry(zh="返回监控", en="Restore"),
    "logs.process.command_failed": Entry(zh="[TUI][ERROR] 无法运行命令：{0}", en="[TUI][ERROR] Cannot run command: {0}"),
    "logs.process.completed": Entry(zh="[TUI] {0} 任务完成，退出码 0", en="[TUI] {0} task completed, exit code 0"),
    "logs.process.failed": Entry(zh="[TUI][ERROR] {0} 任务失败，退出码 {1}", en="[TUI][ERROR] {0} task failed, exit code {1}"),
    "logs.process.started": Entry(zh="[TUI] {0} 进程已启动，PID={1}", en="[TUI] {0} process started, PID={1}"),
    "logs.process.stopping": Entry(zh="[TUI] 正在请求采集进程安全停止……", en="[TUI] Requesting a graceful stop of the collection process..."),
    "logs.process.terminate_escalation": Entry(zh="[TUI] SIGINT 超时，升级为 SIGTERM。", en="[TUI] SIGINT timed out; escalating to SIGTERM."),
    "logs.process.user_stopped": Entry(zh="[TUI] 任务已由用户终止，退出码 {0}", en="[TUI] Task stopped by user, exit code {0}"),
    "logs.retention.omitted_records": Entry(
        zh="[WARN] 诊断缓冲已省略 {0} 条日志记录，其中 {1} 条超过单条长度上限；保留的日志不完整。",
        en="[WARN] Diagnostic buffering omitted {0} log records, including {1} oversized records; retained output is incomplete.",
    ),
    "logs.retention.summary": Entry(zh="保留 {0} 行 · 显示 {1} 行 · 匹配 {2} 处 · 超限丢弃 {3} 行", en="Retained {0} · Shown {1} · Matches {2} · Evicted {3}"),
    "logs.search.placeholder": Entry(zh="搜索保留日志", en="Search retained logs"),
    "logs.stage.transition": Entry(zh="[phase][{0}] {1}（上一阶段 {2}s）", en="[phase][{0}] {1} (previous phase {2}s)"),
    "logs.tools.label": Entry(zh="工具", en="Tools"),
}
