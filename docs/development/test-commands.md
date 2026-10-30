# 自动化测试入口

[← 返回专题目录](testing.md)

## 自动化验证入口

测试树以功能域划分；目录索引及示例参见 [`tests/README.md`](../../tests/README.md)。
`tests/support/` 存放共用构造器，`tests/fixtures/` 存放测试数据，`tests/visual/` 保留截图基线。
按子目录运行时使用 `uv run pytest tests/<domain>`；CI 仍从 `tests/` 递归收集和分片。

安装分发修改除普通回归外，还需构建 sdist/wheel，在隔离环境从空目录执行
`scripts/check_distribution.py`；standalone 用 `--binary <path>` 执行相同验收。
该脚本核对全部公共帮助入口、参数错误、无 Docker 时的 doctor JSON、内置资源和 packet worker 的实际输出。
日常 CI 的独立 `wheel` job 分别从当前 checkout 和 sdist 构建 wheel，
在两个新的 venv 中按 `uv.lock` 导出的临时哈希约束安装依赖及对应 wheel，并检查归档内容与 Docker context 暂存。
验证脚本复制到源码树外，从空目录执行，清除 `PYTHONPATH` 并使用 Python `-I`；同时核对
`acprof.__file__` 位于安装环境内，拒绝意外使用源码或 editable 安装。公共入口从安装包 dispatcher
读取，包含 `compare`、`load`、`model-store`；实际生成离线 HTML 并运行 packet worker，验证
静态资源和子进程随包完整分发。报告和日志作为 `wheel` artifact 保存。
standalone 和真实 `uv tool install` 保留在发布或按需流程；构建步骤见[发行包说明](distribution.md#linux-standalone)。
这些检查不代替 Docker/GPU 推理和完整 profiling。

初始化入口修改运行 `test_setup.py` 和 `test_tui_onboarding.py`，覆盖缺失 uv、安装/诊断失败、
重复执行、非交互终端、首次配置与已有设置保留；使用隔离 uv 目录实际执行
`./setup.sh --no-tui --no-modify-path`。真实推理另用安装后的命令运行最小 basic CPU 实验并审计结果。

测试使用 pytest 原生函数或有明确共同主题的 `Test*` class；文件名与函数名为 `test_*`。
保留 `unittest.mock`；临时目录、环境、输出、日志与异常优先使用 `tmp_path`、`monkeypatch`、
`capsys`、`caplog` 和 `pytest.raises`。初始化与清理使用 function scope 的 yield fixture；
共同的数据构造器放在领域 fixture 模块，避免实例化另一测试 class。
模拟网络、Docker、硬件与通知边界；避免测试触发真实采集或改写用户设置。

隔离整份 `os.environ` 时，使用 `tests/support/environment_fixtures.py` 的 `isolated_environment()`
保留 pytest 的会话状态，并由外层 fixture 恢复完整字典，包括业务代码直接新增的键。
autouse fixture 只请求实际使用的依赖：未使用的 `monkeypatch` 依赖会让其撤销发生在环境恢复之后，
再次删掉父环境的颜色或代理设置。依赖顺序遵循
[pytest 的 autouse 规则](https://github.com/pytest-dev/pytest/blob/9.0.3/doc/en/reference/fixtures.rst#L377-L389)。
`test_environment_fixtures.py` 在独立进程中检查正常结束、断言失败和中断后的环境恢复与测试报告。

```bash
# 按受影响的行为选择测试文件
uv run pytest tests/runtime/environment/test_env_utils.py -v
# 符合“验证范围”中的全套回归条件时
uv run pytest -v
acprof run --help
uv run python -m compileall -q acprof scripts packaging
git diff --check
```

单测试、标记与异步测试：

```bash
uv run pytest tests/runtime/environment/test_env_utils.py::test_local_settings_override_env_file_but_not_explicit_process_values
uv run pytest --collect-only
uv run pytest -m unit
uv run pytest -m "runtime and not hardware"
uv run pytest -m hardware
uv run pytest tests/tui/widgets/test_tui_input.py
```

单个 node ID 以 `--collect-only -q` 的实际输出为准。`asyncio_mode="auto"` 收集
`async def test_*`，测试与 async fixture 的 event loop 均为 function scope，
Textual `run_test()` 通过 async context manager 完成退出；不要遗留后台 task。
`--strict-markers` 拒绝未注册标记，`--import-mode=importlib` 允许领域目录中同名测试模块。
`tui` 自动标记 `tests/tui/` 功能域的测试（包括纯 TUI 逻辑），与其它平台/运行时 marker 独立；
`unit` 自动赋予无 tui/integration/runtime/visual/platform/hardware 标记的测试。
`unit` 只表示不要求专项外部环境，不承诺绝不启动 UI 或恒定低耗时；
跨功能域调用 TUI 的测试仍须按影响范围审查。平台 skip 只作用于显式 `wsl`、`native_linux`，`hardware` 本身不跳过失败。
原有缺依赖或显式 opt-in skip 原因保留；CI runtime 使用 `--require-no-skips` 拒绝缺失验证。

`scripts/check_test_framework.py` 在 pre-commit 与 CI 中禁止旧执行框架的导入，允许 mock；
不通过排除文件或忽略报错维持门禁。正式测试环境暂不安装或启用 pytest-xdist：全套测试包含
全局模块 patch、进程锁、TUI 和资源敏感场景。未来仅在完成 port、临时目录、全局 cache、
Docker 名称与 host 状态审计后，对独立 unit 子集比较串行/并行结果；CI 沿用
[确定性分片与完整性验收](ci.md#ci-与环境测试)，数量以 workflow 的 `HOST_TEST_SHARDS` 为准。

### 本地测试反馈与安全分片

先按受影响模块运行 `uv run pytest tests/<domain>/<subdomain>/test_file.py -q`。
按 `unit` 快速反馈时不会执行 `tests/tui/` 的交互与纯逻辑测试；
修改 TUI 或跨域界面流程时仍须执行对应测试文件及所需尺寸、语言、视觉回归。
完整测试验收仍由 CI 的全部分片覆盖，不以快速筛选代替。

```bash
# 非 TUI 的隔离单元测试；仍可能包含跨域的 TUI 辅助测试
uv run pytest -m unit -q
# TUI 专题测试（独立串行）
uv run pytest tests/tui -q
# 从已有测试 JSON 报告找累计最慢的文件和参数化测试组，无需重跑
uv run python scripts/report_test_durations.py internal-testing/<task>/full-tests.json --top 30
# 本地受控并行：只开放已审查的 analysis、quality 单元测试域
uv run python scripts/run_local_tests.py --scope fast --jobs 2 --path tests/analysis
# 其余域、完整套件保持串行；输出写入 internal-testing/local-tests/
uv run python scripts/run_local_tests.py --scope full --jobs 1
```

本地脚本复用 `scripts/run_tests.py` 和 CI 的 `aggregate_test_reports.py` 完整性校验，
分片分别保存 `shard-N.log/json`，再输出 `summary.json`；
任一失败、缺少分片、重复/遗漏测试、源码或依赖身份不一致均返回非零退出码。
报告和临时目录只写入被忽略的 `internal-testing/local-tests/`，不会写入实验 `results/`。
显式 `--jobs 2` 仅支持 `--scope fast --path tests/analysis...` 或 `tests/quality...`：
每片有独立 pytest basetemp、缓存禁用、子进程组清理，并将内部数值线程上限设为 1。
成功后清理临时 basetemp，失败时保留诊断文件；`--output` 只接受 `internal-testing/` 内的新路径。
其它域包含共享状态、TUI 和资源敏感场景，未经隔离验证不开放本地并行。
同一机器的多个 worktree 仍需人工协调总并发，正式采集窗口不能与测试并行。
本地不安装 `pytest-xdist`，不改变 CI 的完整八分片协议。

耗时报告复用已有 evidence 的每条用例 `duration_s`（setup/call/teardown 之和），
聚合按文件与去除参数后缀的测试组排序；这是**累计用例耗时**，并非所有用例可以并行节省的实际墙钟时间。
旧提交的报告可以找热点，但不能直接当作当前提交、当前负载的基线。
TUI 参数组重构必须保留每个原语言/尺寸/视图或字段组合的断言，
并用重新采集的执行记录验证，而不凭测试数量判断覆盖。

静态门槛由 pre-commit 与 CI 共用：全仓库启用 Ruff `S110`，禁止新增无说明的
`try/except/pass`。诊断回调自身失败的两个保留位置使用局部、带理由的豁免，原始失败和失败状态仍被记录。
共享参数、输入预算、图像解码与 GPU 探测模块额外检查 `BLE001` 与 `C901`，复杂度上限为 12；
这是控制流复杂度上限，不是函数行数。需要捕获第三方任意异常的 GPU 边界必须保留结构化错误，
并在局部说明宽捕获的理由。历史模块没有批量新增复杂度豁免。
这些共享模块与 `host.measurement.config` 同时纳入严格 mypy 注解检查；完整类型入口仍为
`uv run --locked python -m mypy`，定向规则可用以下命令复现：

```bash
uv run --locked ruff check --select BLE001,C901 acprof/experiments/constraints.py acprof/input_budget.py acprof/image_inputs.py acprof/gpu_probe.py acprof/host/metadata/gpu.py
uv run --locked python -m pre_commit run shared-boundaries --all-files
```

`test_architecture.py` 禁止根目录出现任何 `.py` 文件，检查标准库 `profile/cProfile` 可直接导入。
`test_distribution.py`、`test_tui_interaction.py` 与 terminal-log 回归保护公开帮助、
TUI 预览及日志中的 `acprof <command>` 展示，并保留含空格或 shell 特殊字符的参数。

| 实现范围 | 测试入口示例 |
| --- | --- |
| 模块依赖与导入副作用 | `tests/quality/test_architecture.py` |
| 环境、依赖与镜像 | `tests/runtime/environment/test_runtime_profiles.py`、`tests/runtime/environment/test_environment_identity.py`、`tests/runtime/environment/test_lock_compiler.py`、`tests/runtime/containers/test_image_layers.py`、`tests/runtime/containers/test_runtime_image_build.py`、`tests/runtime/containers/test_image_reuse.py`、`tests/runtime/environment/test_runtime_validation.py` |
| 模型文件与离线加载 | `tests/models/storage/test_model_files.py`、`tests/models/storage/test_model_download.py`、`tests/workloads/structured/test_offline_model_loading.py` |
| 字段、能耗、资源与补采 | `tests/measurement/hardware/test_energy_cpu.py`、`tests/measurement/windows/test_resource_usage.py`、`tests/measurement/profiling/test_posthoc.py` |
| 页面、设置与焦点 | `tests/tui/widgets/test_tui_layout_settings.py`、`tests/tui/widgets/test_tui_interaction.py`、`tests/tui/widgets/test_tui_input.py` |
| 启动环境检查 | `tests/tui/workflows/test_tui_startup_preflight.py`；首屏可编辑、静默成功、异常入口、重试、全部采集入口阻断、配置失效、测量锁与迟到回调 |
| 固定页头、底栏与操作颜色 | `tests/tui/widgets/test_tui_page_chrome.py`；中英文、三种尺寸、滚动与缩窗、命令框显隐、八种主题和按钮状态 |
| 日志、语言与命令 | `tests/tui/logging/test_tui_log_view.py`、`tests/tui/widgets/test_tui_i18n.py`、`tests/tui/workflows/test_tui.py` |
| 终端色深与 RGB 输出 | `tests/tui/widgets/test_tui_colors.py`；检查缺失/空 `COLORTERM`、真彩色、256 色与自动检测 |
| 中文浮层缺字 | `tests/tui/widgets/test_tui_cjk_rendering.py`；通知覆盖按钮、真实遮挡、宽字符两半的局部刷新、中英文和奇偶列宽缩放 |
| 统计报告、异步读取与计算 | `tests/tui/results/test_tui_reports.py`、`tests/analysis/reporting/test_report_views.py`、`tests/analysis/statistics/test_uncertainty.py`、`tests/analysis/statistics/test_stats.py` |
| Docker 镜像树、层空间、标签删除与采集互斥 | `tests/runtime/containers/test_image_management.py`、`tests/tui/catalog/test_tui_images.py` |
| 表头拖动、固定列、滚动范围与鼠标释放 | `tests/tui/widgets/test_tui_table_resize.py`、`tests/tui/widgets/test_tui_all_tables.py`、`tests/tui/catalog/test_tui_images.py` |

以上是定位入口，不是每次必须运行的清单。先用 `rg --files tests` 查实际受影响的测试；新改动、失败或未解决问题才需要扩大或重复验证。

`test_runtime_image_build.py` 在 Docker 边界模拟环境中验证四层构建、跨 profile 的环境共享、
模型 commit、Torch 来源、主机下载令牌不进入 Docker 构建、标签错配、额外包、输入变化及构建失败停止。
环境身份测试覆盖当前声明的 profile / environment、无 Torch 环境、跨任务族精确共享及 cu128 的版本差异；CV 新增 timm 后按新包集计算身份。注释、锁文件名
和条目顺序不影响身份，版本、制品、来源、平台和系统锁影响身份。配方变化改变构建缓存，业务
代码变化只重建服务层。它们不能替代实际容器构建与推理验证。

镜像分层测试的临时目录需包含指纹依赖的全部文件，包括 `acprof/models/spec.py`；
修改模型文件筛选或声明解析代码时，断言模型层和服务层指纹变化、依赖层身份不变。
模型接口新增解析字段时，同步更新 CLI 委派测试中独立声明的完整 `model_resolution`
期望值，保留完整对象比较，不从被测函数的返回值生成期望。

自动解析与编排回归使用 `test_resolution_decisions.py`、`test_auto.py`、`test_model_coverage.py`
和 `test_model_inspection.py`，覆盖同源证据、显式冲突处理、验证不提升静态裁决、固定 SHA、
主机失败与旧 CLI 选项拒绝、原生模型验证、冻结覆盖率分母及人工语义参考。已有模型／契约／镜像／
恢复测试继续保护协议。滚动模型检查使用 [coverage 命令](../usage/cli-discovery.md#acprof-coverage)，
其静态、接口检查、容器运行验证与正式测量证据分别验收；4 GiB 或超时限制不等同于模型语义错误。

采集准备与重试使用 `test_collection_workflow.py`、`test_run_native_docker.py`、
`test_run_notifications.py` 和 `test_tui_collection_workflow.py`。当前顺序为采集平台策略检查、
开始通知、模型解析、主机预检、运行环境准备和验证、正式测量；预检失败必须阻止后续准备。
策略拒绝时不发送开始通知、不访问 Hub 或 Docker；策略通过后，开始通知早于模型解析和 Docker 预检。
预检和通知测试在
`acprof.host.detect.detect_task` 边界提供静态 fixture，避免假模型 ID 访问真实 Hub。
监控布局测试同时检查状态网格、准备状态行和日志的衔接，以及各尺寸下日志和按钮的可用空间。
