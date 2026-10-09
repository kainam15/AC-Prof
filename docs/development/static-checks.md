# 类型与模块边界检查

[← 返回专题目录](testing.md)

## 渐进类型检查与边界回归
`requirements/dev.lock` 固定 mypy 2.3.1；`pyproject.toml` 的白名单覆盖 RunConfig、artifact/layout、
extension schema、Handler boundary、Monitor interface、MonitorGroup 与 command runner，
以及 matrix plan、run state、compute/execution plan、profiler support/纯解析器和
comparison/independent comparison/uncertainty/precision、窗口边界诊断、latency report、runtime validation 与测试身份/分片。
runtime validation 的任务、镜像和输入计划采用具体类型，报告顶层采用 TypedDict；
容器返回的任务专有 JSON 仍在实际运行时验证。当前清单以 `pyproject.toml` 为准，仍只维护 mypy。
初期允许未标注函数和缺失第三方 stubs，`follow_imports=skip` 防止隐式扩大检查范围；
已经列出的模块仍检查已标注代码。不能用全包 `ignore_errors` 隐藏白名单内的问题。
独立 CI `types` job 与本地运行同一条命令：

```bash
.venv/bin/python -m mypy
```

类型工具只进入开发锁，不改变 host/runtime 的运行依赖。版本选择参考
[mypy 的 Python 支持范围](https://github.com/python/mypy/blob/master/mypy/defaults.py)
及 [Ruff 的 import sorting 说明](https://github.com/astral-sh/ruff/blob/main/docs/faq.md)；
两者采用 MIT 许可并持续维护，检查目标保持 Python 3.10，开发检查没有测量期开销。
Ruff 的 `combine-as-imports` 保留显式重导出分组；脚本先设置路径的 `E402` 注释留在对应 import 语句上。

`test_host_command.py` 使用真实短子进程验证 timeout、异常、编码、环境、cwd、耗时与脱敏。
`test_architecture.py` 检查 host 中未登记的直接同步 subprocess 调用（含 import aliases）。
`test_monitor_cleanup.py` 和 `test_perf_mips.py` 验证 preparation-before-start、窗口中无 Docker discovery，
以及成功、取消、超时后才发布请求/结果；`test_resource_usage.py` 验证采样不重复扫描 CPU 拓扑。
这些回归不代替真实 Docker/GPU/perf/NCU 采集或用户终端显示证据。

`test_compute_profile_runner.py` 的导入 fixture 恢复容器入口设置的离线环境变量，
并覆盖主机原有变量缺失、关闭和启用三种状态，避免污染后续 Hub SDK 的首次导入。

`test_window_boundary_diagnostics.py` 使用受控时钟和延迟 stop 验证请求后的采样尾部、默认无诊断开销、
无请求对照隔离、失败/取消与幂等收尾；四个真实 monitor 的边界 getter 只读取已有字段。
`test_overhead.py` 和 `test_overhead_entrypoint.py` 覆盖源 CPU/内存/输入尺度选择、
收尾后写出 sidecar、失败保留原始异常及默认不输出。Native sidecar 仍需另用已有固定镜像与输入验证。

`test_precision.py` 与 stats CLI 回归用独立手算数据验证区间半宽、失败窗口筛选、
退化重采样和默认 JSON 兼容，不运行正式采集。

`test_output_boundaries.py` 用独立进程验证 library import 不输出、不配置 root logger，
验证 DEBUG 与用户 stderr 的边界，并用真实短子进程检查 TUI 的 stdout/stderr 合并。
`test_monitor_cleanup.py` 在成功、超时和取消路径启用 DEBUG，断言采样开始至停止间没有日志。
`test_progress_events.py` 继续保护 machine events；`test_execution_profile.py` 在工具所属模块
模拟命令，验证 checkpoint、恢复、失败条目、清理与通知顺序。

`internal-testing/`、`result-past/` 和 `results/` 都由仓库 `.gitignore` 排除。忽略规则不授权删除：
失败或中断的 CSV、pcap、jsonl 与恢复状态应按实验保留；清理前先列出路径、占用和是否仍用于诊断或恢复。
开发环境副本与旧构建目录可单独评估，不能仅按文件扩展名批量删除实验依据。
全局 `*.csv` 保留；`tests/fixtures/**/*.csv` 与 `examples/**/*.csv` 显式放行。
`test_git_ignore.py` 在临时 Git 仓库验证根层及嵌套 fixture/example 可跟踪、结果与实验 CSV 仍被忽略。

## 跨模块 private API 检查
`scripts/check_private_api.py` 不导入业务代码，只解析 `acprof/` 的 Python 源码。
识别 `from x import _foo`、`import x; x._foo`、`from pkg import mod; mod._foo`，
以及 import alias、相对 import、函数局部作用域、lambda 与 comprehension 参数遮蔽。
同模块访问、dunder、第三方模块和实例私有属性不进入本 gate；测试的 private helper 引用单独统计。
这是显式静态依赖检查，不追踪动态 `importlib`/`getattr`、运行时 re-export 或任意赋值的数据流。

`tests/private_api_baseline.json` 保存排序且去重的 `source`、`target`、`symbol`，不保存行号或单一总数。
新增 tuple 与过期 tuple 都令 CI 和 architecture test 失败；删除依赖后必须同步删除条目，
以后重新加入会再次被拒绝。确需新增共享 private dependency 时，审查具体 tuple 和职责理由，
不能仅用重建整份 baseline 或扩大豁免消除失败。

```bash
.venv/bin/python scripts/check_private_api.py
# 仅删除已不存在的条目；不会批准新增依赖。
.venv/bin/python scripts/check_private_api.py --prune
.venv/bin/python scripts/run_tests.py --pattern test_private_api_guard.py \
  --pattern test_architecture.py --report internal-testing/private-api.json
```

设计参考 [Import Linter protected contract](https://github.com/seddonym/import-linter/blob/main/src/importlinter/contracts/protected.py)
的具体依赖与 unmatched ignore 检查（BSD-2-Clause），并核对
[Pylint private import checker](https://github.com/pylint-dev/pylint/blob/main/pylint/extensions/private_import.py)
（GPL-2.0）和 [Pyright 对测试目录的独立设置](https://github.com/microsoft/pyright/discussions/8193)
（MIT）。Import Linter 面向模块图，Pylint 该规则主要针对外部 private import；本 gate 需要
项目内 symbol tuple 和只减不增的历史基线，因此独立使用 Python 3.10 标准库 AST 实现，
不复制这些项目的源码，也不引入额外 linter 或第二套类型工具。它仅在开发与 CI 中运行。

## 关键模块导入方向检查
`scripts/check_import_boundaries.py` 使用标准库 AST 检查源码层的依赖方向，
禁止 `analysis → host`、`host → cli/tui`、`container → host/tui` 等反向运行时导入。
明确跳过 `if TYPE_CHECKING` 内的类型引用，并且只允许两条已有的
`monitors.* → host.command` 例外，避免扩大隐式耦合。
CI 与 `tests/test_import_boundaries.py` 同时验证真实仓库与反例；动态 import 不在静态检测范围。

```bash
.venv/bin/python scripts/check_import_boundaries.py
.venv/bin/python -m pytest -q tests/test_import_boundaries.py
```

本阶段 mypy 增加共享硬件证据协议和纯 `client_metrics`，逐步扩大范围而非一次改成全局严格模式。

## TUI 控制器拆分回归
`tests/test_tui_action_ownership.py` 约束各 Textual handler 所属模块、App 的进程生命周期与代码体量。
交互与线程语义以 `tests/test_tui*.py`、`tests/test_environment_tui.py` 和既有 visual snapshots
作为行为证据；只通过源码级 ownership 检查不能替代真实的 event-loop/worker 回归。
