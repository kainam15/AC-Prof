# 类型与模块边界检查

[← 返回专题目录](testing.md)

## 渐进类型检查与边界回归
`uv.lock` 固定 `typing` 组的 mypy；类型检查唯一配置在 `pyproject.toml`。
`packages = ["acprof"]` 覆盖整个 Python 源码包；新增模块自动进入类型检查，无需手工
维护数百个路径白名单。CI、开发机和发行包继续使用同一份 `pyproject.toml` 配置。

`check_untyped_defs = true` 让尚未完整注解的函数体也受检查；这不等于参数和返回值
都已有类型。`follow_imports = "silent"` 仍用于依赖类型推断，但不能排除 `acprof` 包中的模块；
包内所有源文件均被显式检查。
缺失的第三方 stubs 仍暂时允许；不通过 `ignore_errors`、新增全局排除或强行改为全局 `strict`
掩盖未解决的问题。对一批已验证全注解的模块，`[[tool.mypy.overrides]]` 局部启用
`disallow_any_generics`、`disallow_untyped_defs` 和 `warn_return_any`。
这些规则只约束对应模块，不能代表整个项目没有隐式 `Any`；保留外部 JSON / 动态框架更新等必要的动态边界。

已经完成整包类型覆盖，后续新增的模型解析、容器运行、TUI 和 CLI 源文件默认参加检查。
Textual 框架动态事件和 subclass 关键字的局部类型例外仍保留在对应实现旁，未启用
全局 `ignore_errors`；未来逐模块收紧 `Any` 和未注解函数，不通过跳过函数体提高覆盖率。

TUI 的业务能力使用具体类型的 controller 组合；每个 controller 显式接收 `AcprofTui`，
其拥有的状态在构造函数初始化。跨职责调用写明 `app.configuration`、`app.results`、
`app.preflight` 等接收方，mypy 会检查实际方法名、参数与返回值。
App 仅保留 Textual 事件/快捷键入口及进程、测量和退出生命周期；不通过业务 MRO、
`if TYPE_CHECKING: assert isinstance(self, AcprofTui)`、动态方法查找或宽泛 Protocol 获取隐式成员。
普通 controller 的 worker 直接提交给 App 的 worker manager；工作线程函数和启动方法均保留明确签名。
弹窗和小部件通过 Textual 的泛型 `self.app` 获取宿主时，继续使用
`acprof/tui/app_contracts.py` 的三个小型接口；候选选择与镜像文案接口暴露具体 controller，
静态接口检查验证真实 App 满足这些要求。这里的既有框架类型边界不用于业务 controller 内部。
测试 mock 位于实际所属模块或 `app.<controller>`，不使用 `create=True` 增补已删除的业务接口。
未来收紧 `Any` 和未注解函数仍按模块推进；不能通过跳过函数体或增加 ignore 掩盖接口错误。

GitHub Actions 的 `types` job 在 Python 3.12 中同时安装项目依赖和 `typing` 组：

```bash
uv sync --locked --no-default-groups --group typing
uv pip check
uv run --locked --no-default-groups --group typing mypy
```

日常开发使用 `uv sync`、`uv run mypy` 即可；默认 `dev` 已包含 `typing`。
如需保留其他解释器的工作环境，可以显式隔离：

```bash
UV_PROJECT_ENVIRONMENT=internal-testing/mypy-312 uv sync --locked --python 3.12 --no-default-groups --group typing
UV_PROJECT_ENVIRONMENT=internal-testing/mypy-312 uv run --locked --python 3.12 --no-default-groups --group typing mypy
```

仅给 mypy 指定另一解释器而未在对应环境安装相同主机依赖和开发工具不等于复现 CI：
如果目标解释器没有安装 Textual 等主机依赖，第三方声明可能被视为未知类型，产生误报或**错误的通过结果**。
这一环境要求遵循 [mypy 的导入与解释器说明](https://github.com/python/mypy/blob/master/docs/source/running_mypy.rst)。
NumPy `.pyi` 包含 Python 3.12 的 `type` 语句时，必须确保目标版本和实际环境一致，
不能通过全局关闭第三方类型检查掩盖版本不匹配。类型工具只进入开发锁，不增加 host/runtime 依赖与测量期开销。
版本与实施策略参考 [mypy 官方渐进式检查指南](https://mypy.readthedocs.io/en/stable/existing_code.html)、
[mypy 的 Python 支持范围](https://github.com/python/mypy/blob/master/mypy/defaults.py)
和 [Pydantic 的 mypy 测试配置](https://github.com/pydantic/pydantic/blob/main/tests/mypy/configs/pyproject-default.toml)，
只借鉴逐模块严格化的方法，不复制配置中的无关依赖豁免。
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
uv run python scripts/check_private_api.py
# 仅删除已不存在的条目；不会批准新增依赖。
uv run python scripts/check_private_api.py --prune
uv run python scripts/run_tests.py --pattern test_private_api_guard.py \
  --pattern test_architecture.py --report internal-testing/private-api.json
```

设计参考 [Import Linter protected contract](https://github.com/seddonym/import-linter/blob/main/src/importlinter/contracts/protected.py)
的具体依赖与 unmatched ignore 检查（BSD-2-Clause），并核对
[Pylint private import checker](https://github.com/pylint-dev/pylint/blob/main/pylint/extensions/private_import.py)
（GPL-2.0）和 [Pyright 对测试目录的独立设置](https://github.com/microsoft/pyright/discussions/8193)
（MIT）。Import Linter 面向模块图，Pylint 该规则主要针对外部 private import；本 gate 需要
项目内 symbol tuple 和只减不增的历史基线，因此独立使用 Python 标准库 AST 实现，
不复制这些项目的源码，也不引入额外 linter 或第二套类型工具。它仅在开发与 CI 中运行。

## 关键模块导入方向检查
`scripts/check_import_boundaries.py` 使用标准库 AST 检查源码层的依赖方向，
禁止 `analysis → host`、`host → cli/tui`、`container → host/tui` 等反向运行时导入。
明确跳过 `if TYPE_CHECKING` 内的类型引用，并且只允许两条已有的
`monitors.* → host.command` 例外，避免扩大隐式耦合。
CI 与 `tests/test_import_boundaries.py` 同时验证真实仓库与反例；动态 import 不在静态检测范围。

```bash
uv run python scripts/check_import_boundaries.py
uv run pytest -q tests/test_import_boundaries.py
```

`analysis.latency_validation` 与共享硬件证据协议、纯 `measurement.metrics` 等已验证模块使用
局部严格检查；后续继续逐模块收紧，而非一次改成全局严格模式。延迟报告和绘图使用公共数值 API，
`latency_model` 的私有函数不进入跨模块 baseline。

## TUI 控制器拆分回归
`tests/test_tui_action_ownership.py` 约束各 Textual handler 所属模块、App 的进程生命周期与代码体量。
交互与线程语义以 `tests/test_tui*.py`、`tests/test_environment_tui.py` 和既有 visual snapshots
作为行为证据；只通过源码级 ownership 检查不能替代真实的 event-loop/worker 回归。
