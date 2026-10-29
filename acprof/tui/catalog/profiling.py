"""Bilingual messages for profiling; keys name stable display purposes."""
from __future__ import annotations

from .entry import Entry

MESSAGES: dict[str, Entry] = {
    "profiling.backfill.complete_experiment_required": Entry(
        zh="请先选择可补采的完整实验",
        en="Select a complete experiment that supports posthoc profiling first",
    ),
    "profiling.backfill.confirm_details": Entry(
        zh="该操作会启动隔离 profiler，并在成功后原子回填现有结果。原文件会按项目规则备份。\n\n",
        en="Run isolated profilers and atomically backfill existing results on success. Original files are backed up according to project rules.\n\n",
    ),
    "profiling.backfill.confirm_title": Entry(zh="执行 profiler 补采？", en="Run profiler backfill?"),
    "profiling.backfill.isolation_note": Entry(
        zh="只运行勾选的独立分析，不改变正式测量数据。",
        en="Selected profilers run separately from and do not change measured samples.",
    ),
    "profiling.backfill.massif_toggle_hint": Entry(
        zh="CPU-only 进程生命周期内存峰值；点击或按空格切换。",
        en="CPU-only process-lifetime peak memory; click or press Space to toggle.",
    ),
    "profiling.backfill.ncu_toggle_hint": Entry(zh="GPU 实际执行 FLOP；点击或按空格切换。", en="GPU executed FLOP; click or press Space to toggle."),
    "profiling.backfill.no_tools": Entry(zh="请至少勾选一个补采工具", en="Select at least one backfill tool"),
    "profiling.backfill.nsys_toggle_hint": Entry(
        zh="CUDA API、kernel 和 memcpy 时间线；点击或按空格切换。",
        en="CUDA API, kernel and memcpy timeline; click or press Space to toggle.",
    ),
    "profiling.backfill.plan": Entry(zh="补采计划（dry-run）", en="Plan backfill"),
    "profiling.backfill.preview": Entry(zh="查看补采计划", en="Preview profiling plan"),
    "profiling.backfill.run": Entry(zh="执行补采", en="Run backfill"),
    "profiling.backfill.tools": Entry(zh="补采工具", en="Backfill tools"),
    "profiling.backfill.torch_toggle_hint": Entry(zh="逻辑 FLOP；点击或按空格切换。", en="Logical FLOP; click or press Space to toggle."),
    "profiling.metric.cpu_memory_peak": Entry(zh="CPU 内存峰值", en="CPU memory peak"),
    "profiling.metric.cuda_timeline": Entry(zh="CUDA 执行时间线", en="CUDA execution timeline"),
    "profiling.metric.gpu_flop": Entry(zh="GPU 实际 FLOP", en="Actual GPU FLOP"),
    "profiling.metric.logical_flop": Entry(zh="逻辑 FLOP · CPU / GPU", en="Logical FLOP · CPU / GPU"),
    "profiling.mode.basic": Entry(zh="基础（延迟 / CPU / 内存）", en="Basic (latency / CPU / memory)"),
    "profiling.mode.full": Entry(zh="完整（RAPL / perf / 抓包）", en="Full (RAPL / perf / packets)"),
    "profiling.mode.label": Entry(zh="画像模式", en="Profiling mode"),
    "profiling.options.first_run_advice": Entry(
        zh="按需选择；分析器分别运行，可能增加准备时间。首次采集建议关闭。",
        en="Optional isolated profilers; may extend setup time. Leave off for a first run.",
    ),
    "profiling.options.off_summary": Entry(zh="分析器关闭", en="profilers off"),
    "profiling.options.selected_summary": Entry(zh="计算={0} · 执行={1}", en="compute={0} · execution={1}"),
    "profiling.options.title": Entry(zh="深度性能分析", en="Deep performance profiling"),
    "profiling.overview.available_summary": Entry(
        zh="已完成 {0}/{1} 项适用分析 · 可补充 {2} 项",
        en="Completed {0}/{1} applicable tools · {2} available to collect",
    ),
    "profiling.overview.complete_summary": Entry(zh="已完成 {0}/{1} 项适用分析，无需补采。", en="Completed {0}/{1} applicable tools; nothing to collect."),
    "profiling.overview.open": Entry(zh="补充性能分析…", en="Additional profiling…"),
    "profiling.overview.select_experiment": Entry(
        zh="选择实验并读取摘要后，可查看补充分析。",
        en="Select an experiment and read its summary to inspect additional analysis.",
    ),
    "profiling.overview.title": Entry(zh="性能深度分析", en="Advanced performance analysis"),
    "profiling.overview.tool_status": Entry(zh="{0} 分析：{1}", en="{0} profiling: {1}"),
    "profiling.overview.unavailable": Entry(zh="当前结果无法补采：{0}", en="Profiling unavailable for this result: {0}"),
    "profiling.status.complete_count": Entry(zh="已完成 {0}/{1}", en="Complete {0}/{1}"),
    "profiling.status.failed_count": Entry(zh="失败 {0} · 已完成 {1}/{2}", en="{0} failed · {1}/{2} complete"),
    "profiling.status.not_collected": Entry(zh="尚未采集", en="Not collected"),
    "profiling.status.partial_count": Entry(zh="部分完成 {0}/{1}", en="Partial {0}/{1}"),
    "profiling.tools.massif_description": Entry(zh="Valgrind Massif — CPU 内存分析", en="Valgrind Massif — CPU memory analysis"),
    "profiling.tools.ncu_description": Entry(zh="NVIDIA NCU — GPU 内核性能分析", en="NVIDIA NCU — GPU kernel analysis"),
    "profiling.tools.nsys_description": Entry(zh="NVIDIA Nsys — CPU/GPU 执行时序分析", en="NVIDIA Nsys — CPU/GPU timeline analysis"),
    "profiling.tools.torch_description": Entry(zh="PyTorch Profiler — 算子与框架计算分析", en="PyTorch Profiler — framework/operator analysis"),
}
