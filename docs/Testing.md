# 测试与验证

唯一测试入口是 `python -m pytest`；CI 的 `scripts/run_tests.py` 仅转发 pytest 参数并启用同一插件。

本次迁移的基线、集合映射与验证限制见 [2026-10-03 pytest 迁移验收记录](reviews/2026-10-03-pytest-migration.md)。

选择本次改动能改变的行为和失败路径。下列命令均从仓库根目录执行，使用已有 `.venv`。
测试数量、设备余量和镜像可用性由本次执行确认，不把历史通过记录作为当前验证结果。

## 验证范围

WSL2 是开发与 PARTIAL 采集平台。pytest 注册 `unit`、`wsl`、`native_linux`、`hardware`
markers；未标集成边界的测试归入 unit。WSL 默认执行 `.venv/bin/python -m pytest -m "not native_linux"`，
Native Linux 执行完整测试。仅显式平台集成测试按真实环境 skip，不因 WSL 跳过普通代码异常。
模拟 sysfs/NVML 的单元测试仍需执行；hardware marker 本身不隐藏失败。
环境检测、能力矩阵、历史 unknown、CSV 合并与比较隔离回归在 `test_environment_policy.py`、
`test_result_comparison.py`，详见 [WSL2 支持范围](platforms/wsl2.md)。
修改 RAPL、PMU/perf、cgroup、NVML、CPU topology、affinity、cold start 或 energy 时必须报告
`Native validation: verified / required / not applicable` 中的一项；WSL/mock 通过不能替代 Native 证据。

| 改动 | 应取得的证据 |
| --- | --- |
| 文档、导航或链接迁移 | 本地文件与章节锚点可达、旧入口仍可跳转、代码块与差异格式正确；无需为措辞运行模型 |
| Skill | frontmatter、名称与描述匹配、相对路径、流程边界和可用验证器的格式检查 |
| 单个逻辑或失败路径 | 能观察目标行为的相关 pytest；修复缺陷时先复现，再验证修复 |
| 模块搬迁、依赖方向或兼容入口 | 当前入口成功、已删除入口拒绝、全套 pytest、CLI 帮助和编译；mock 放到函数实际查找依赖的模块 |
| 指标或产物协议 | 独立推导的期望值、当前 schema 成功与旧 schema 拒绝、缺失/失败/不适用字段和受影响消费者 |
| 模型、backend、依赖或 Dockerfile | 路由与离线加载测试、镜像构建、所声明设备的真实推理；profiler 分别验证 |
| TUI | 受影响的交互与尺寸检查；原生终端问题还需对应终端证据 |

下载策略的定向回归包含 `test_download_network.py`、`test_model_store.py`、
`test_network_preflight.py`、`test_dependency_download_cache.py`、`test_lock_compiler.py`
与 `test_tui_downloads.py`。Hub transport 用真实 SDK 加受控 HTTP transport 检查重定向前阻断，
不下载真实权重；Model Store 覆盖 SHA256、空间、预算、独立 dependency refs 和活动 lease。
依赖下载测试使用本地 HTTP 服务覆盖客户端标识、重定向后的 HEAD 方法、SHA256 校验和缓存命中，
无需外网；锁生成测试在 Python 3.10 验证明确拒绝且不修改原锁，在 Python 3.11+ 验证镜像解析及 hash 保护。
依赖、模型、Dockerfile 变更还需分别说明新构建、已有 runtime、小型 fixture、真实 checkpoint 的验证范围。

## Python 修改工作流

修改 Python 代码时优先使用 PyCharm MCP，覆盖 `acprof/`、`packaging/`、`scripts/` 和 `tests/`。
只使用本次任务涉及的工具；纯文档修改按[文档检查](#文档与-skill-检查)验证。

| 场景 | 工具与约束 |
| --- | --- |
| 定位程序符号 | 使用 `search_symbol`；`rg` 用于文件、普通文本和配置检索 |
| 分析调用或依赖 | 优先使用 `analyze_calls`，结合源码确认动态调用；不得仅凭文本搜索推断 Python 符号关系 |
| 重命名 Python 符号 | 优先使用 `rename_refactoring`，核对引用更新和实际差异 |
| 检查修改后的文件 | 使用 `lint_files` / `get_file_problems` 检查受影响文件的 IDE diagnostics，处理本次改动引入的问题 |
| 执行 IDE 测试或 smoke test | 用 `get_run_configurations` 选择相关的已有 Run Configuration，通过 `execute_run_configuration` 执行 |
| 核对最终改动 | 使用 `git_status` 并结合 diff，检查新增、被忽略文件，确认没有混入无关变更 |

按[验证范围](#验证范围)运行相关 pytest / evidence、Ruff 及真实 workload；
命令见[开发质量检查](#开发质量检查)与[自动化验证入口](#自动化验证入口)。
局部修改不默认跑完整测试集；只有新改动、失败或未解决问题才扩大或重复验证。

MCP 不可用、索引不完整或没有适用 Run Configuration 时，说明限制并用源码分析和项目 CLI 入口继续；
不把空调用树当作没有依赖。pytest 缺失时先安装开发锁；IDE、pytest 与 evidence 的边界见 [PyCharm MCP 的验证边界](#pycharm-mcp-的验证边界)。
真实 workload 缺少 Docker、GPU、模型等运行条件时，明确标为未验证，不用 IDE diagnostics 或 smoke test 代替。

## 开发质量检查

产物布局改动的定向入口包括 `test_artifact_layout.py`、`test_run_recovery.py`、
`test_auto.py`、`test_posthoc.py`、`test_result_audit.py`、`test_result_comparison.py`、
`test_latency_model_report.py` 和 `test_tui_reports.py`。覆盖新目录写入、flat 目录读取及恢复、
中断文件保留、补采备份与分析输出路由；这些离线用例不代替真实 Docker/GPU 采集。

矩阵计划或 RAPL/DRAM 协议改动的定向入口：

```bash
.venv/bin/python scripts/run_tests.py \
  --pattern 'test_matrix_plan.py' --pattern 'test_startup_probe.py' \
  --pattern 'test_run_recovery.py' --pattern 'test_energy_cpu.py' \
  --pattern 'test_capabilities.py' --pattern 'test_client_energy_warnings.py' \
  --pattern 'test_metric_registry.py' \
  --report internal-testing/matrix-dram-tests.json
.venv/bin/python scripts/render_metric_reference.py --check
```

这些用例验证同 seed 的实际计划、resume 不重排、probe 不写性能结果、异常不扩大 OOM
前缀，以及模拟 sysfs 的 alias、不同 package、回绕、部分覆盖和读取失败。
mock 测试不证明物理服务器的 DRAM 域可读或实际 Docker OOM 行为；真实验收须另查
拓扑、启动失败的 Docker State，以及可读 RAPL/DRAM 设备上的 full 测量产物。

RAPL 的模拟 sysfs 必须包含用于识别域类型的 `name`（如 `package-0`、`core`、`dram`），
不能只创建 `energy_uj`。`tests/test_tui.py` 的循环链接用例也调用生产拓扑读取器，
修改发现逻辑时应一并验证该用例。

指标登记表新增字段时，在 `test_metric_registry.py` 中显式列出新增字段，保留历史字段
顺序的基准哈希；运行 `scripts/render_metric_reference.py` 更新速查文档后再执行 `--check`。
生成器显式读写 UTF-8，写入无 BOM 的 LF 文本，不依赖 Windows 或 Linux 的默认 locale。
新字段插入对应用途组，整体 `status`、`error` 保持在最后两列。列顺序调整还需验证
旧表头的追加、case 合并、packet 回填和 profiler 补采，确保按列名保留数值及未知扩展列。

Ruff、pre-commit 和锁生成工具 uv 由 [`requirements/dev.in`](../requirements/dev.in) 声明，
完整版本与制品哈希保存在 [`requirements/dev.lock`](../requirements/dev.lock)。开发锁以主机锁
为约束，避免在同一个 `.venv` 安装时引入冲突；不加入主机运行依赖或容器环境身份。

```bash
.venv/bin/python -m pip install --require-hashes -r requirements/dev.lock
.venv/bin/python -m pip check
.venv/bin/python -m pre_commit install
.venv/bin/python -m pre_commit run --all-files --show-diff-on-failure
```

`install` 给当前 clone 安装 `pre-commit` 和 `commit-msg` 两个 Git hooks；新 clone 需执行一次。
手动运行和 CI 读取同一份
[`.pre-commit-config.yaml`](../.pre-commit-config.yaml)，检查尾随空白、文件末尾换行、YAML、JSON、
TOML、冲突标记、文件大小和 Python 代码。大文件检查对所有文件执行，限额为 1 MiB，覆盖现有
约 938 KiB 的固定音频输入；Markdown 的两个行尾空格保留为换行。临时证据目录和禁止修改的
`docs/Original_Project_Definition.md` 不参与 hooks。

### AI 提交信息与格式校验

VS Code 的 Source Control 面板中，点击 ✨ **Generate Commit Message** 生成提交信息。
在工作区 `.vscode/settings.json` 中合并以下设置，保留已有配置：

```json
{
  "github.copilot.chat.localeOverride": "en",
  "github.copilot.chat.commitMessageGeneration.instructions": [
    { "file": ".github/commit-message.instructions.md" }
  ]
}
```

生成模板只在 [commit-message.instructions.md](../.github/commit-message.instructions.md)
维护：`type(scope): concise summary` 标题、空行，以及核心改动、测试或兼容性说明的列表。
scope 可选，标题保持简洁但不设字符数上限。工作区设置只引用该文件。
先暂存本次提交的改动再生成，并在提交前核对内容。

工作区设置若被个人 `.git/info/exclude` 或全局规则忽略，换电脑或新 clone 时需重新合并
上述设置。规则文件本身可随仓库保存。模型沿用 VS Code 当前的 utility model 配置；
需要更换时，在 Settings 中搜索 `Chat: Utility Small Model` 并选择账号可用的模型。

Copilot 0.67.0 的提交生成提示包含 `ResponseTranslationRules`；默认 `localeOverride: auto`
会按 VS Code 界面语言加入系统级语言要求。上述工作区配置使用 `en` 去掉自动追加的中文
要求，使英文提交规则不再与它冲突；该设置也影响当前项目 Copilot Chat 的默认输出语言，
不会改变 VS Code 界面语言。修改设置后清空旧提交消息，再点击 ✨ 重新生成；旧消息不会
自动重写，必要时执行 **Developer: Reload Window**。

[`commitlint`](https://github.com/conventional-changelog/commitlint) 在 `commit-msg` 阶段读取
[.commitlintrc.json](../.commitlintrc.json)，强制检查 Conventional Commits 结构、允许的类型、
scope 大小写及正文/页脚前的空行；标题、正文和页脚均不限制行长度。
不合规的消息会阻止提交；修改消息后重试。提交内容由生成模板指导并由提交者复核，
格式校验不能证明内容属实。
保留 commitlint 对 Git 自动生成的 merge、revert、fixup!/squash! 等消息的默认豁免。

commitlint 需要 Node.js 22.12+，依赖安装在 pre-commit 的隔离缓存中，不进入 Python 开发锁
或推理环境。安装 hooks 后，首次提交可能需要联网初始化依赖；可提前执行：

```bash
.venv/bin/python -m pre_commit install --install-hooks
```

手动校验一个已存在的消息文件（不创建 Git commit）：

```bash
.venv/bin/python -m pre_commit run commitlint --hook-stage commit-msg \
  --commit-msg-filename /path/to/commit-message.txt
```

`pre_commit run --all-files` 执行文件检查，不校验提交消息或历史提交；当前 CI 的 `lint` job
也仅检查文件。AI 生成功能还需要当前 VS Code 中的 Copilot 或可用的 utility model；
本地格式校验通过不代表已经验证界面中的模型生成结果。

VS Code 提交失败弹窗可能只显示 hook 输出的第一行；点击“显示命令输出”查找真正的
`Failed` 和具体规则名。`body-leading-blank` 表示标题后直接出现正文，缺少空行；应改为
一个标题，或在标题与正文之间加入空行。不要通过关闭该规则来接受拼接的多个标题。

### Ruff 与代码检查

Ruff 版本由 [`pyproject.toml`](../pyproject.toml) 的 `required-version` 强制核验，Python 目标为
3.10，显式启用 `E4`、`E7`、`E9`、`F`，以及 `B006`（可变默认值）、`B012`（finally 跳转）、
`B904`（异常链）、`I`（import 排序），以及 `RUF010`（f-string 显式转换）、
`RUF013`（显式 Optional）、`SIM101`（合并同一对象的 isinstance）。不启用全量 RUF/SIM、`E501` 或 formatter；
`line-length = 100` 本身不检查行长。Ruff hook 只检查，不自动修复；空白和末尾换行 hooks
会修正文件并返回失败，检查 `git diff` 后重新运行。不得用扩大 `ignore` 或排除目录掩盖新问题。
公共导出用显式重导出或 `__all__` 表达；必须先设置路径、环境或验证缺失依赖的 import，
只在对应行标注具体规则及原因，不统一忽略 `__init__.py`。

本地 Python 修改先运行 Ruff，再按受影响行为选择 pytest 文件、node ID 或 marker。
需要 evidence 时追加 `--report`，或使用 CI wrapper；后者的 `--pattern` 可重复。
局部修改默认验证受影响范围。Git hook 不运行业务测试或硬件采集。

```bash
.venv/bin/ruff check acprof/host/env_utils.py tests/test_env_utils.py
.venv/bin/python -m pytest tests/test_env_utils.py \
  --report internal-testing/env-tests.json
git diff --check
```

更新开发工具时，修改输入并使用开发锁中的 uv 版本重新生成；Ruff 需同步修改版本约束及 hook 的
完整 commit SHA，pre-commit 需同步最低版本。随后核对锁和 diff，重新安装开发锁并运行完整 hooks。

```bash
.venv/bin/uv pip compile requirements/dev.in --python-version 3.10 --universal \
  --generate-hashes --no-annotate --no-header --output-file requirements/dev.lock
```

`scripts/compile_locks.py --check` 仍只验证既有容器锁与 profile 映射，不代替开发锁的重新解析。
主机依赖变更还需用 Python 3.11+ 执行 `scripts/compile_locks.py --host-only --check`，核对发行声明、
已验证 pin 与主机 lock；重新生成方式见[运行兼容](Runtime_Compatibility.md#当前配置)。CI 的 Python 3.12 job
运行该检查；Python 3.10 job 保留容器锁检查。
这些开发工具只在编辑、提交和 CI 验证时运行，不进入正式测量窗口。

离线可视化的定向入口为 `test_metric_registry.py`、`test_analysis_model.py`、`test_report.py`；检查旧 CSV、
分组隔离、能量范围、缺失值、冷启动去重、转义及公共 report 入口。浏览器交互由 CI 的独立
`report-browser` job 执行，Python 3.12、Playwright 和绘图依赖锁在 `requirements/browser.lock`，
不安装到共享 `.venv` 或 runtime。锁使用现有 uv 0.12.13 生成：

```bash
.venv/bin/uv pip compile requirements/browser.in --python-version 3.12 \
  --generate-hashes --no-annotate --no-header -o requirements/browser.lock
.venv/bin/uv venv internal-testing/browser-venv --python .venv/bin/python
.venv/bin/uv pip install --python internal-testing/browser-venv/bin/python \
  --require-hashes -r requirements/browser.lock
internal-testing/browser-venv/bin/python -m playwright install chromium
ACPROF_BROWSER_TESTS=1 ACPROF_BROWSER_ARTIFACT_DIR=internal-testing/browser-evidence \
  internal-testing/browser-venv/bin/python scripts/run_tests.py --pattern test_report_browser.py \
  --report internal-testing/browser-evidence/tests.json --require-no-skips
```

CI 在准备阶段用 `playwright install --with-deps chromium` 安装锁定 Playwright 对应的 Chromium。
默认使用这份浏览器；`ACPROF_BROWSER_EXECUTABLE` 仅供显式验证其他浏览器版本时覆盖，不自动选择系统 Chrome。
测试使用 `offline=True`，页面不得产生 HTTP(S) 请求或 JavaScript 异常；覆盖原有五项 baseline、筛选、
排序、跨图选择、Pareto、窄屏交互，以及质量筛选和原始证据展示。未设置 `ACPROF_BROWSER_TESTS=1` 时
普通主机测试明确 skip，浏览器 job 使用 `--require-no-skips` 要求实际执行。
报告固定保存在 artifact 目录，测试失败另保存页面截图、DOM、console/page error、原报告和 Playwright trace。
测试报告和日志无论成功失败均由 CI 上传。浏览器安装需要网络，页面交互本身离线。
条件回归检查输入顺序不匹配时保留散点但暂停改善比例及 baseline frontier，并对照 Python/JavaScript
的精确 workload 分布判断。颜色范围回归检查 200 个配置、2 个指标只读取 400 次数值，验证线性/log、
neutral、零值和分组。合成页面依次渲染 100、400、1000 个配置的六指标矩阵，检查排序颜色及筛选数值，
将首次就绪、排序与筛选耗时写入 `matrix-scaling.json`；这些时间包含浏览器与共享开发机开销，
不是模型性能数据，也不设脆弱的跨机器绝对时限。当前范围保留完整表格；更大规模是否分页或虚拟化需另行实测。
Headless Chromium 证据不等于实际 Windows 浏览器验收，也不证明 Docker/GPU 采集正确。
所有验证均避开正式测量窗口。

### 渐进类型检查与边界回归

`requirements/dev.lock` 固定 mypy 2.3.1；`pyproject.toml` 的白名单覆盖 RunConfig、artifact/layout、
extension schema、Handler boundary、Monitor interface、MonitorGroup 与 command runner，
以及 matrix plan、run state、compute/execution plan、profiler support/纯解析器和
comparison/independent comparison/uncertainty。当前清单以 `pyproject.toml` 为准，仍只维护 mypy。
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

### 跨模块 private API 检查

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

### 辅助开发工具

`requirements/test.in` / `test.lock` 统一固定 pytest、pytest-asyncio、pytest-cov 和
pytest-textual-snapshot；`dev.in` 包含它，主机和推理 runtime 的依赖声明不包含测试工具。
Textual 与主机锁一致；pytest 8.4.2 / syrupy 4.8.0 保留已有 SVG 序列化格式。
`textual-dev` 与 Hypothesis 仍是按需工具，添加普通测试依赖前须同步开发锁和 CI。

| 工具 | 适用场景 | 运行入口 |
| --- | --- | --- |
| `textual-dev` | Textual 开发日志、事件和调试界面 | `.venv/bin/textual` |
| Hypothesis | 同步输入与状态转换的性质测试 | pytest 中的 `@given` / `strategies` |
| `pytest-textual-snapshot` | 固定场景 SVG 视觉回归 | `.venv/bin/python -m pytest tests/visual` |

```bash
.venv/bin/python -m pip install --require-hashes -r requirements/host.lock -r requirements/dev.lock
.venv/bin/python -m pytest --version
.venv/bin/python -m pip check
```

交互调试时，在两个终端中分别从仓库根目录运行：

```bash
# 终端 A：开发控制台
.venv/bin/textual console
# 终端 B：连接控制台运行 TUI
TEXTUAL=devtools,debug acprof tui
```

Hypothesis 可用于同步 pytest 测试，并使用 `python -m pytest <文件>`
执行选定用例。每个生成样例应隔离可变状态；发现失败后保留最小输入，加入稳定回归。
不要直接用 `@given` 包装异步 `Pilot` 交互；优先生成同步输入处理或状态转换的样例。
新增依赖这些工具的常规测试时，先补齐相应开发依赖与 CI 环境，不能仅依赖本机安装。

快照用例使用 `snap_compare` fixture，显式指定 `terminal_size`，固定语言、主题和输入，
仓库的 `tests/visual/test_snapshots.py` 固定九个场景：中文窄终端、英文常规尺寸、宽终端、
弹窗覆盖、实际拖动表格之后、测量中、清理未完成和中英文 WSL2 模型检测等待；覆盖 `80×24`、`120×30`、`150×45`。
通用场景固定 Native Linux 身份，WSL2 场景固定 PARTIAL，避免基线随运行测试的主机变化。
基线在 `tests/visual/__snapshots__/`。测试隔离设置、固定主题和显示路径，不启动采集或外部服务。
普通功能测试与 SVG 回归共用开发环境和 pytest 配置：

```bash
TZ=UTC PYTHONHASHSEED=0 .venv/bin/python -m pytest tests/visual -q \
  --snapshot-report internal-testing/tui-snapshot-report.html
```

更新测试工具和开发锁（保持 `host.lock` 约束，Python 3.10+）：

```bash
.venv/bin/uv pip compile requirements/runtime-test.in --python-version 3.10 --universal \
  --generate-hashes --no-annotate --no-header -o requirements/runtime-test.lock
.venv/bin/uv pip compile requirements/test.in --python-version 3.10 --universal \
  --generate-hashes --no-annotate --no-header -o requirements/test.lock
.venv/bin/uv pip compile requirements/dev.in --python-version 3.10 --universal \
  --generate-hashes --no-annotate --no-header -o requirements/dev.lock
```

快照用例固定主题、语言、路径与颜色模式，规范化 SVG 行末空白以兼容仓库格式检查；
截取前检查目标页面已经激活，避免把错误场景保存成基线。
正式测量场景须显式设置接口解析与运行验证为 `passed`、测量为 `running`；
不能只设置 `measurement_active=True` 而让阶段状态保留 `not_started`。
测量中和清理未完成的场景还须模拟由应用管理、`poll()` 返回 `None` 的进程，
并隔离停止操作；截取前断言停止按钮可用，不启动真实采集或向真实进程发送信号。

先查看失败报告的 HTML / SVG 差异，再在预期变更或首次建立基线时对选定用例追加
`--snapshot-update`；普通验证不更新基线。快照与功能测试使用同一 pytest runner，
不能代替行为断言、evidence JSON 或[真实终端证据](#tui-与终端证据)。
上述调试、样例生成和截图均在正式测量窗口之外运行；按任务选择流程见
[TUI 回归 Skill](../.agents/skills/acprof-textual-regression/SKILL.md#按需选择辅助工具)。

用法参考上游维护的 [textual-dev](https://github.com/Textualize/textual-dev)
与 [pytest-textual-snapshot](https://github.com/Textualize/pytest-textual-snapshot)（MIT），
以及 [Hypothesis](https://github.com/HypothesisWorks/hypothesis)（MPL-2.0）。
复用现有 CLI、性质测试和快照机制；通过统一测试锁及按需运行控制依赖与维护成本，
不向推理环境或正式采集进程加入开发工具。

### PyCharm MCP 的验证边界

使用项目的 `.venv` 解释器；符号搜索按需限定 `paths=["acprof/**", "tests/**"]`，
避免将临时虚拟环境中的第三方代码视为项目实现。调用分析无法解析已找到的 Python 符号时，
结合符号文档和源码核对调用者；空结果不能证明没有依赖。

若 `analyze_calls` 返回两条相同的候选标识，使用 `search_symbol(include_external=true)`
核对是否同时找到源码和 `.venv/.../acprof/_bundle` 中的安装副本。`.venv` 的项目排除规则
不排除 Python SDK 库索引。editable 安装不应复制这份 bundle；修复 build hook 后，执行
`uv pip install --python .venv/bin/python --no-deps --reinstall-package acprof -e .` 更新安装，
再验证入向和出向调用。资源打包约定见[安装包说明](Distribution.md#工作目录与资源)。

PyCharm 2026.2.3（build `262.10968.92`）已复现一种 MCP 兼容问题：
`analyze_calls` 的 `isCallableSymbol` 依赖显示文本中的 `name(...)`，
而 Python 函数的 Usage View 文本只有名称，因此真实 `PyFunction` 也会被过滤。
这类错误应修复 IDE 工具的函数识别，不需要改动业务函数或重建 Python 环境。
修复验收应包含真实函数的入向、出向调用和不存在符号的错误路径；
本机兼容补丁还需验证版本匹配、撤销与启动加载，并区分独立 JVM 检查和完整 IDE 重启。
实现依据见 [JetBrains Call Hierarchy](https://github.com/JetBrains/intellij-community/blob/master/plugins/mcp-server/mcpserver.toolsets/src/general/CallHierarchyAnalysisSupport.kt)
与 [Python Usage View](https://github.com/JetBrains/intellij-community/blob/master/python/src/com/jetbrains/python/findUsages/PyElementDescriptionProvider.java)。

PyCharm 在 **Settings → Tools → Python Integrated Tools → Testing** 中选择 **pytest**，
使用项目 `.venv`；已有 Run Configuration 的 target 改为 pytest 的文件或 node ID。
VS Code 的 Python Test Explorer 使用 `python.testing.pytestEnabled: true`、
`python.testing.pytestArgs: ["tests"]`，解释器选择同一 `.venv`。
CLI、IDE、host CI 与 SVG job 使用同一配置。需要 CI evidence 时追加 `--report <文件.json>`；
普通控制台输出不能代替 JSON，证据必须包含真实输出和退出码。
`build_project` 若提示无法收集构建诊断，不能替代 Python 编译和相关测试；
依赖查询返回空列表也不能证明 Python 环境没有安装依赖。
临时重命名和工具测试文件放在任务独立的 `internal-testing/` 子目录中。

## 自动化验证入口

安装分发修改除普通回归外，还需构建 sdist/wheel，在隔离环境从空目录执行
`scripts/check_distribution.py`；standalone 用 `--binary <path>` 执行相同验收。
该脚本核对全部公共帮助入口、参数错误、无 Docker 时的 doctor JSON、内置资源和 packet worker 的实际输出。
日常 CI 的独立 `wheel` job 分别从当前 checkout 和 sdist 构建 wheel，
在两个新的 venv 中按 `host.lock` 安装依赖及对应 wheel，并检查归档内容与 Docker context 暂存。
验证脚本复制到源码树外，从空目录执行，清除 `PYTHONPATH` 并使用 Python `-I`；同时核对
`acprof.__file__` 位于安装环境内，拒绝意外使用源码或 editable 安装。公共入口从安装包 dispatcher
读取，包含 `compare`、`load`、`model-store`；实际生成离线 HTML 并运行 packet worker，验证
静态资源和子进程随包完整分发。报告和日志作为 `wheel` artifact 保存。
standalone 和真实 `uv tool install` 保留在发布或按需流程；构建步骤见[发行包说明](Distribution.md#linux-standalone)。
这些检查不代替 Docker/GPU 推理和完整 profiling。

初始化入口修改运行 `test_setup.py` 和 `test_tui_onboarding.py`，覆盖缺失 uv、安装/诊断失败、
重复执行、非交互终端、首次配置与已有设置保留；使用隔离 uv 目录实际执行
`./setup.sh --no-tui --no-modify-path`。真实推理另用安装后的命令运行最小 basic CPU 实验并审计结果。

测试使用 pytest 原生函数或有明确共同主题的 `Test*` class；文件名与函数名为 `test_*`。
保留 `unittest.mock`；临时目录、环境、输出、日志与异常优先使用 `tmp_path`、`monkeypatch`、
`capsys`、`caplog` 和 `pytest.raises`。初始化与清理使用 function scope 的 yield fixture；
共同的数据构造器放在领域 fixture 模块，避免实例化另一测试 class。
模拟网络、Docker、硬件与通知边界；避免测试触发真实采集或改写用户设置。

```bash
# 按受影响的行为选择测试文件
.venv/bin/python -m pytest tests/test_env_utils.py -v
# 跨模块变更的全套回归
.venv/bin/python -m pytest -v
acprof run --help
.venv/bin/python -m compileall -q acprof scripts packaging
git diff --check
```

单测试、标记与异步测试：

```bash
python -m pytest tests/test_env_utils.py::test_local_settings_override_env_file_but_not_explicit_process_values
python -m pytest --collect-only
python -m pytest -m unit
python -m pytest -m "runtime and not hardware"
python -m pytest -m hardware
python -m pytest tests/test_tui_input.py
```

单个 node ID 以 `--collect-only -q` 的实际输出为准。`asyncio_mode="auto"` 收集
`async def test_*`，测试与 async fixture 的 event loop 均为 function scope，
Textual `run_test()` 通过 async context manager 完成退出；不要遗留后台 task。
`--strict-markers` 拒绝未注册标记，`--import-mode=importlib` 允许领域目录中同名测试模块。
`unit` 自动赋予无 integration/runtime/visual/platform/hardware 标记的测试；
平台 skip 只作用于显式 `wsl`、`native_linux`，`hardware` 本身不跳过失败。
原有缺依赖或显式 opt-in skip 原因保留；CI runtime 使用 `--require-no-skips` 拒绝缺失验证。

`scripts/check_test_framework.py` 在 pre-commit 与 CI 中禁止旧执行框架的导入，允许 mock；
不通过排除文件或忽略报错维持门禁。正式测试环境暂不安装或启用 pytest-xdist：全套测试包含
全局模块 patch、进程锁、TUI 和资源敏感场景。未来仅在完成 port、临时目录、全局 cache、
Docker 名称与 host 状态审计后，对独立 unit 子集比较串行/并行结果；CI 继续四个确定性分片。

`test_architecture.py` 禁止根目录出现任何 `.py` 文件，检查标准库 `profile/cProfile` 可直接导入。
`test_distribution.py`、`test_tui_interaction.py` 与 terminal-log 回归保护公开帮助、
TUI 预览及日志中的 `acprof <command>` 展示，并保留含空格或 shell 特殊字符的参数。

| 实现范围 | 测试入口示例 |
| --- | --- |
| 模块依赖与导入副作用 | `tests/test_architecture.py` |
| 环境、依赖与镜像 | `tests/test_runtime_profiles.py`、`tests/test_environment_identity.py`、`tests/test_lock_compiler.py`、`tests/test_image_layers.py`、`tests/test_runtime_image_build.py`、`tests/test_image_reuse.py`、`tests/test_runtime_validation.py` |
| 模型文件与离线加载 | `tests/test_model_files.py`、`tests/test_model_download.py`、`tests/test_offline_model_loading.py` |
| 字段、能耗、资源与补采 | `tests/test_energy_cpu.py`、`tests/test_resource_usage.py`、`tests/test_posthoc.py` |
| 页面、设置与焦点 | `tests/test_tui_layout_settings.py`、`tests/test_tui_interaction.py`、`tests/test_tui_input.py` |
| 启动环境检查 | `tests/test_tui_startup_preflight.py`；首屏可编辑、静默成功、异常入口、重试、全部采集入口阻断、配置失效、测量锁与迟到回调 |
| 固定页头、底栏与操作颜色 | `tests/test_tui_page_chrome.py`；中英文、三种尺寸、滚动与缩窗、命令框显隐、八种主题和按钮状态 |
| 日志、语言与命令 | `tests/test_tui_log_view.py`、`tests/test_tui_i18n.py`、`tests/test_tui.py` |
| 终端色深与 RGB 输出 | `tests/test_tui_colors.py`；检查缺失/空 `COLORTERM`、真彩色、256 色与自动检测 |
| 中文浮层缺字 | `tests/test_tui_cjk_rendering.py`；通知覆盖按钮、真实遮挡、宽字符两半的局部刷新、中英文和奇偶列宽缩放 |
| 统计报告、异步读取与计算 | `tests/test_tui_reports.py`、`tests/test_report_views.py`、`tests/test_uncertainty.py`、`tests/test_stats.py` |
| Docker 镜像树、层空间、标签删除与采集互斥 | `tests/test_image_management.py`、`tests/test_tui_images.py` |
| 表头拖动、固定列、滚动范围与鼠标释放 | `tests/test_tui_table_resize.py`、`tests/test_tui_all_tables.py`、`tests/test_tui_images.py` |

以上是定位入口，不是每次必须运行的清单。先用 `rg --files tests` 查实际受影响的测试；新改动、失败或未解决问题才需要扩大或重复验证。

`test_runtime_image_build.py` 在 Docker 边界模拟环境中验证四层构建、跨 profile 的环境共享、
模型 commit、Torch 来源、BuildKit secret、标签错配、额外包、输入变化及构建失败停止。
环境身份测试覆盖 40 个 profile / 27 个环境、无 Torch 环境、跨任务族精确共享及 cu128 的版本差异；CV 新增 timm 后按新包集计算身份。注释、锁文件名
和条目顺序不影响身份，版本、制品、来源、平台和系统锁影响身份。配方变化改变构建缓存，业务
代码变化只重建服务层。它们不能替代实际容器构建与推理验证。

镜像分层测试的临时目录需包含指纹依赖的全部文件，包括 `acprof/model_spec.py`；
修改模型文件筛选或声明解析代码时，断言模型层和服务层指纹变化、依赖层身份不变。
模型接口新增解析字段时，同步更新 CLI 委派测试中独立声明的完整 `model_resolution`
期望值，保留完整对象比较，不从被测函数的返回值生成期望。

自动解析与编排回归使用 `test_resolution_decisions.py`、`test_auto.py`、`test_model_coverage.py`
和 `test_model_inspection.py`，覆盖同源证据、显式冲突处理、验证不提升静态裁决、固定 SHA、
主机失败与旧 CLI 选项拒绝、原生模型验证、冻结覆盖率分母及人工语义参考。已有模型／契约／镜像／
恢复测试继续保护协议。滚动模型检查使用 [coverage 命令](CLI_Reference.md#acprof-coverage)，
其静态、接口检查、容器运行验证与正式测量证据分别验收；4 GiB 或超时限制不等同于模型语义错误。

采集准备与重试使用 `test_collection_workflow.py`、`test_run_native_docker.py`、
`test_run_notifications.py` 和 `test_tui_collection_workflow.py`。当前顺序为采集平台策略检查、
开始通知、模型解析、主机预检、运行环境准备和验证、正式测量；预检失败必须阻止后续准备。
策略拒绝时不发送开始通知、不访问 Hub 或 Docker；策略通过后，开始通知早于模型解析和 Docker 预检。
预检和通知测试在
`acprof.host.detect.detect_task` 边界提供静态 fixture，避免假模型 ID 访问真实 Hub。
监控布局测试同时检查状态网格、准备状态行和日志的衔接，以及各尺寸下日志和按钮的可用空间。

## CI 与环境测试

独立 `lint` job 在 Python 3.10 上只安装开发锁，执行 `pip check` 和完整 pre-commit hooks；
不安装推理依赖，也不需要 Docker/GPU。检查内容及 hook 版本与本地一致。

`.github/workflows/ci.yml` 在 Python 3.10 / 3.12 上安装哈希锁并执行主机回归；
每个版本将完整测试集按排序后的 test ID 轮转分成八片，保留 20 分钟作业超时。
工作流的 `HOST_TEST_SHARDS` 同时传入测试 runner、证据汇总与 coverage 完整性检查；
修改分片数时同步矩阵 index，避免 coverage 开销使单片超时并丢失最终报告。
所有分片都执行完整 discovery，新增测试会自动分配；不使用手写文件白名单。
指标文档的编码与同步检查在独立的命名步骤中先于测试分片执行；失败时可直接定位到
`render_metric_reference.py --check`，不会混入 CLI 帮助步骤。
每片分别上传 `host.json` 和实时保存的 `host.log`，失败时继续执行其他分片。
同时运行 `compile_locks.py --check`。七个任务族分别执行 CPU 接口测试，以随机小模型或明确导出的
样例验证真实加载与推理；audio 和 multimodal 在同一作业共享一个 CPU 依赖环境，仍分别执行测试。
网络在容器测试期间关闭。CI Actions 固定为已核验的 commit SHA，作业只授予仓库读取权限。
另有 `nlp-transformers560-cpu` 矩阵项运行新版原生架构与图像 processor 测试，避免主机缺少推理依赖的 skip 掩盖环境回归。
`custom-multimodal-cpu` 矩阵项运行 `test_custom_multimodal_runtime.py`：未知 `auto_map` 架构的
随机小模型经音频、图像、视频输入映射执行，比较共享四阶段与上游 pipeline 结果，并检查
重复推理、真实 Torch profiler 算子及独立 runtime validation；容器禁止网络和跳过。
两个 Transformers 版本线均执行 `test_audio_generation_runtime.py`：保存小型原生模型 snapshot，
再经过共享 Auto 加载、原生音频消息、真实 `generate` 和输出验证；逐参数核对加载权重，防止
组合模型的前缀处理丢失权重。4.57.6 验证 Voxtral/Qwen2 Audio；5.6.0 另验证 Omni 文本子模型。
这些随机权重验证接口，不证明完整 checkpoint 的容量、质量或正式 profiling 指标。

`check_runtime.py` 将所选 `ACPROF_RUNTIME_PROFILE` 与 `ACPROF_MODEL_ADAPTER` 显式传入容器，
让版本核对、adapter、精度和 remote-code 策略都绑定该 profile。共享同一依赖环境的不同任务仍
各用合法 profile：5.6.0 的 NLP 与音频生成分别为 `nlp-transformers560-cpu` 和
`multimodal-transformers560-cpu`，输出目录独立。生成的 NLP custom-code fixture 只在测试
进程注册允许/拒绝 profile，生产信任策略不放宽；basic probe fixture 显式设置 backend、profile 和 adapter，
避免继承主机环境或在导入测试前因缺少配置而失败。
容器测试继续保留版本检查、禁网和 `--require-no-skips`。

`test_cv_runtime.py` 只包含 CPU 可执行的 CV 接口回归；CUDA 专项放在
`test_cv_cuda_runtime.py`，两者复用 `cv_runtime_fixtures.py` 的微型 snapshot 与 dtype 断言。
CUDA 专项须在已安装 CV 依赖且可访问 GPU 的环境中单独执行
`python scripts/run_tests.py --pattern test_cv_cuda_runtime.py --require-no-skips --report <report.json>`。
缺少 CUDA 时跳过不能作为 GPU 验收；CPU job 仍拒绝任何 skip。

本地入口：

```bash
.venv/bin/python scripts/run_tests.py --report internal-testing/host-tests.json
.venv/bin/python scripts/compile_locks.py --check
.venv/bin/python scripts/check_runtime.py --family audio --variant cpu --output-dir internal-testing/audio-runtime
.venv/bin/python scripts/check_runtime.py --profile moss-transformers560 --build-only --output-dir internal-testing/moss-dependencies
.venv/bin/python scripts/check_runtime.py --profile nlp-transformers560-cpu --test-pattern test_transformers5_runtime.py --output-dir internal-testing/transformers5-runtime
.venv/bin/python scripts/check_runtime.py --profile multimodal-transformers560-cpu --test-pattern test_audio_generation_runtime.py --output-dir internal-testing/audio-generation-runtime
.venv/bin/python scripts/render_metric_reference.py --check
```

`run_tests.py` 是 pytest CLI wrapper，保留 `--directory`、重复 `--pattern`、`--report`、
`--require-no-skips` 与分片参数；普通 pytest 也能直接使用后五项参数。
wrapper 显式加载仓库配置；直接 pytest 写入仓库外已有报告时，追加 `-c pyproject.toml`，
避免外部路径影响 pytest 的配置发现和 rootdir 推断。
`tests/conftest.py` 只注册 `acprof.testing.plugin`，evidence 与 sharding 分别位于独立模块。
报告保留 schema v1 的 test ID、outcome、reason、duration、Python/platform/package 版本、
UTC timestamp、discovery counts、完整 suite hash 和 shard 元数据。ID 使用 pytest node ID；
参数化 case 各有 ID，不再合并失败。call 的断言失败为 `failed`，未处理异常及 setup/teardown
失败为 `error`；同一 case 的各阶段合为一条记录并累计耗时，保留失败原因。
xfail/xpass 分别对应 `expected_failure` / `unexpected_success`；后者始终使 evidence 失败。
`--require-no-skips` 同时拒绝 skip 与 xfail。空 pattern、空分片、收集错误和中断不报告成功；
`--collect-only` 输出的 `collected_ids` 可检查稳定性，但其 evidence 不代表测试已执行。
筛选先于分片：对完整筛选后 node ID 排序，用 `index::count` 分配，SHA256 包含整个筛选集。
新增测试或参数改变 suite hash；同一 revision、参数与依赖环境的重复收集必须一致。

容器接口验证按 runtime platform 的 Python、Linux wheel tags 和依赖 markers，
从 `runtime-test.lock` 生成保留原始 hashes 的 `runtime-test-target.lock`，下载对应 wheelhouse。
这避免 Python 3.12 主机漏掉 Python 3.10 容器所需的 backport；普通主机安装仍遵守版本条件。
依赖条件沿用 [pytest-asyncio 的声明](https://github.com/pytest-dev/pytest-asyncio/blob/v1.2.0/pyproject.toml)，
目标筛选复用项目锁工具，避免 [pip 按主机解释 markers](https://github.com/pypa/pip/issues/6117) 的跨版本下载问题。
下载后在 `--network none` 容器内
创建 `/tmp/acprof-tests` 临时环境，复用镜像的推理依赖；不向推理镜像添加 pytest，也不改变
环境身份或硬件采集入口。硬件 workflow 仍调用真实 `check_hardware.py`，pytest 通过不能替代它。
Runtime policy 的最小固定回归在 `test_runtime_preflight.py`：覆盖 GLM-OCR task registry、
SAM/SAM2 dtype、RMBG/skimage、manga-ocr/fugashi、缺少结构化模型 contract，均不下载权重。
`test_runtime_validation.py` 验证预算耗尽与阶段存证；`test_runtime_evidence.py` 验证
CSV/TUI/audit/report 统一原因、质量与能力独立、selected artifact 预算；`test_loading_quality.py`
验证 loading info 返回约定及失败后的恢复。对应真实 smoke 仍须另外记录 checkpoint SHA、
镜像 ID、Transformers 版本、CPU/GPU、实际 dtype 和输出验证；mock 与随机权重不替代该证据。

`test_metric_reference.py` 在关闭 UTF-8 mode、启用 `EncodingWarning` 错误的独立进程中
验证指标文档生成与检查：生成固定使用 UTF-8（无 BOM）和 LF；检查接受 UTF-8 的 LF/CRLF
工作区文件，对缺失、GBK 编码或内容过期返回非零并提示重新生成，不改写文档。
本地可用 `--shard-index 0 --shard-count 8` 重现一个 CI 分片；省略参数执行完整测试集。
pytest 的 session autouse fixture 为测量锁注入临时目录，覆盖 setup/call/teardown；
CLI、IDE 和 wrapper 均受保护。生产入口仍固定使用 `/tmp` 的同用户锁，`TMPDIR` 不能绕过。
本地汇总可执行 `python scripts/aggregate_test_reports.py <下载目录> --shard-count 8 --report <新报告.json>`；
单版本验证用 `--python-versions 3.12`。缺分片、失败、计数／摘要不符或测试归属错误均使汇总失败。
报告的 `shard` 记录编号、总片数、完整发现数、选中数和排序后 test ID 列表的 SHA256。
`host-summary` job 自动下载两个 Python 版本的全部分片到各自目录；
`scripts/aggregate_test_reports.py` 确认同一 Python 版本分片齐全且测试集摘要相同，所有 test ID 无重复，
总执行数等于完整发现数；单片通过不代表主机回归完成。

### Host coverage baseline

`requirements/dev.in` 与带 hash 的开发锁固定 `coverage[toml]==7.15.2`，支持 Python 3.10+。
CI 仅在 Python 3.12 的八个 host shard 使用 branch coverage；Python 3.10 使用相同 pytest wrapper。
每片单独上传 `.coverage.*`，显式开启 hidden files；`host-summary` 先验收测试证据，
再确认八个 coverage 目录齐全，执行 `coverage combine --keep`、`xml`、`json`、`html`，
发布 `coverage-baseline-3.12` artifact。缺分片不能生成完整 baseline。

第一阶段只记录覆盖情况，不设全仓百分比 gate，也不排除错误与清理路径来提高数字。
`pyproject.toml` 固定 `source=["acprof"]`、branch 和 relative paths；未运行的模块仍进入报告。
默认本地产物位于已忽略的 `internal-testing/coverage/`，CI 用 `COVERAGE_FILE` 指向独立 evidence 目录。

```bash
# 使用装有 host/dev lock 的 Python 3.12 环境，每个 index 执行一次。
python -m coverage erase
for shard in 0 1 2 3 4 5 6 7; do
  python -m coverage run --branch --parallel-mode scripts/run_tests.py \
    --shard-index "$shard" --shard-count 8 \
    --report "internal-testing/coverage/host-3.12-$shard/host.json" || exit 1
done
python scripts/aggregate_test_reports.py internal-testing/coverage \
  --python-versions 3.12 --shard-count 8 --report internal-testing/coverage/host-summary.json
python -m coverage combine --keep
python -m coverage xml
python -m coverage json
python -m coverage html
```

优先查看 schema/merge、matrix planning、run recovery、energy math、error/cleanup 的未覆盖分支。
该 baseline 只度量 host runner 进程；未启用对子进程的自动注入，不能代表子进程、Docker、
GPU、RAPL 或 perf 的实际路径。测试中的 mock 覆盖也不等于 Native validation。
设计复用 [Coverage.py 官方合并流程](https://github.com/coveragepy/coveragepy/blob/main/coverage/data.py)
（Apache-2.0）与统一 pytest/evidence 插件；锁定版本、仅安装开发依赖，不进入采集环境。
hidden files 行为见 [upload-artifact #602](https://github.com/actions/upload-artifact/issues/602)。

### 恢复与运行环境回归

空测试集必定失败；容器作业带 `--require-no-skips`，跳过或 expected failure 都不算环境验证通过。
普通主机测试允许缺少推理依赖时跳过，报告明确列出范围。`check_runtime.py` 的目录必须为空；
`runtime.json` 另记录逻辑 profile、环境 ID、平台/环境 image ID、完整运行清单和退出结果，不覆盖旧验证。
`--build-only` 仅证明依赖构建和清单核验；`--family` / `--profile` 共用主构建的环境准备入口。

可靠性回归包括 `test_run_identity.py`（跨 TMPDIR 的进程互斥、声明/资源变化拒绝续跑）、
`test_monitor_cleanup.py`（实际 client 循环的清理失败与请求证据）、`test_container_ownership.py`
（独立名称、失败/取消时按完整 ID 回收）和 `test_release_gate.py`（同提交的发布前置依赖）。
`test_tui_process_lifecycle.py` 补充回调异常、忽略信号的真实子进程、锁释放及清理状态；
`test_progress_events.py` 检查控制事件不受日志措辞影响，`test_hardware_conditions.py` 与
`test_result_comparison.py` 检查 affinity、比较目的和未知证据，`test_run_recovery.py` 检查 CPU 集合续跑身份。
服务上下文及 host/TUI 变化不重建服务的约束由镜像构建测试覆盖。mock 测试不代替实际 Docker 采集。
普通接口测试固定使用 CPU，即使选择 CUDA wheel；GPU 和自定义 MOSS adapter 由完整服务镜像的
独立 `runtime_validation` 验证。依赖迁移需逐一构建全部唯一环境，再分别验证共享环境的各 profile。

更新锁前后比较原完整包版本，验证目标 wheel 的 ABI/平台和全部制品 SHA256。默认锁生成保留原
版本，显式 `--upgrade` 才更新环境包；uv 固定 0.12.13，生成 wheel 锁的脚本使用 Python 3.11+，
只读检查兼容 Python 3.10+。系统锁可指定同一 snapshot 再生成并比较，过程只修改一次性容器和
指定输出锁。具体命令见[当前配置](Runtime_Compatibility.md#当前配置)。

源代码、模型权重和依赖层保持分离，CPU 容器测试不下载 Hub 模型，不代替真实 GPU/PMU/抓包实验。

通用接口回归见 `test_generic_model_interfaces.py`：任意 checkpoint 名称的路由、固定 revision
元数据读取、任务／架构到版本线的选择、制品拒绝、prompt 预算及 workload 参数重放。
`test_cv_runtime.py` 在离线容器比较两类原生 timm 快照与官方模型输出；`test_timeseries_runtime.py`
比较 Chronos-Bolt／Chronos-2 的多序列预测；`test_nlp_runtime.py` 比较完整句向量模块图、默认 prompt
和归一化数值。`test_transformers5_runtime.py` 用旧环境没有的 Qwen3.5 原生小配置验证共享 NLP 入口，
并验证新版图像 processor 的输入形状、检测框及依赖 OpenCV 的多边形输出。该测试需在
`nlp-transformers560-cpu` 共享环境显式运行；随机小配置不是热门 checkpoint 的采集或准确率证据。

候选发现、冲突处理、本地模型声明、镜像／恢复身份和输入宽度回归见
`test_model_discovery.py`；`test_validation_stages.py` 检查成功步骤、失败位置及原异常保留。
`test_custom_pipeline_runtime.py` 在断网 NLP 容器创建微型随机 BERT snapshot 与自定义 pipeline，
比较原生接口和自定义接口的 label/score，并执行真实独立验证、失败阶段检查及自定义 `auto_map`
架构加载。它已加入默认 NLP 容器测试集，由现有 CI 的 NLP job 执行；也可单独运行：

```bash
.venv/bin/python scripts/check_runtime.py --family nlp --variant cpu \
  --test-pattern test_custom_pipeline_runtime.py \
  --output-dir internal-testing/custom-pipeline-runtime
```

该证据覆盖锁定环境中的标准文本分类桥接，不证明任意自定义模型、其它任务或 GPU 兼容。
多模态共享接口的离线测试可运行：

```bash
.venv/bin/python scripts/check_runtime.py --profile custom-multimodal-cpu \
  --test-pattern test_custom_multimodal_runtime.py \
  --output-dir internal-testing/custom-multimodal-runtime
```

主机协议／环境选择回归在 `test_custom_multimodal.py`；依赖 commit、文件哈希与镜像复用检查在
`test_model_download.py`、`test_image_reuse.py`；TUI 输入、命令、持久化和语言切换在
`test_tui_model_spec.py`。真实 Ultravox checkpoint 还需对应基础仓库访问权限和设备容量。

自动契约的主机回归使用 `test_model_contract.py` 和
`tests/fixtures/custom_pipeline_audio_like/`：覆盖 SHA 绑定、来源状态、输入映射、确定性参数、
动态表达式／冲突拒绝、依赖候选、缓存身份和报告导出；恶意源码检查在隔离进程中确认没有执行
模型顶层代码，也没有导入 Torch／Transformers。共享 fixture 的 Hub mock 按仓库 ID 区分主模型
与依赖；依赖测试通过 `dependency_lookup` 注入响应并检查调用，避免内外两层 patch 同一个
`HfApi.model_info` 时覆盖依赖 SHA 和文件列表。生成契约与通用 handler 的真实衔接使用随机小权重：

```bash
.venv/bin/python scripts/run_tests.py --pattern test_model_contract.py \
  --report internal-testing/model-contract-tests.json
.venv/bin/python scripts/check_runtime.py --profile custom-multimodal-cpu \
  --test-pattern test_model_contract_runtime.py \
  --output-dir internal-testing/model-contract-runtime
```

第二条命令核验锁定环境并在断网 CPU 容器执行加载、预处理、推理及输出验证，拒绝跳过。
同一容器测试还覆盖 Interface Probe 仅导入／签名、不加载权重，以及经显式审阅生成的嵌套 `turns` 模板。
M4～M6 的主机回归使用 `test_model_dependencies.py`、`test_model_review.py`、`test_model_transforms.py`、
`test_model_probe.py`、`test_model_inspection.py`：覆盖依赖角色／SHA／过滤、条件和动态路径、逐字段决策、
DSL 深度和引用限制、只读断网命令、证据完整性、CLI 导出及失败状态。`test_interface_probe.py` 验证
不进入模型准备或 Model Store、缓存零联网、缺图拒绝、文件边界、SHA、递归源码与取消清理。
TUI 的一键开始、只读字段、resolver 重新确认、同窗错误／重试和中英文三种终端尺寸由
`test_tui_model_resolution.py` 与 `test_tui_collection_workflow.py` 验证；后者用真实子进程检查取消后
临时目录、锁和进程释放。`test_collection_workflow.py` 阻断 postprocess／validate_output 失败及非成功报告；
`test_runtime_validation.py` 检查 CPU/GPU、最小输入和完整阶段证据。
真实断网容器检查之外，外部依赖验收应保留 Hub SHA、实际选择文件和缓存内容，确认未下载无关权重；
真实只读 Probe 应使用独立测试镜像和目录，不能只用 mock Docker 命令代替。
`test_dependency_flow.py` 覆盖 active/inactive/unknown、参数绑定、main-model 转发、primary/fallback、
角色分离及动态条件／kwargs／装饰器／递归边界。`test_ultravox_dependency_flow.py` 使用保留许可与 SHA256
的固定上游源码文本，验证零未决项、与示例声明的角色语义一致、未知 Transformers 版本保持 review，
以及换成任意主模型 ID 仍得到相同规划。原有
`test_conditional_weight_load_does_not_download_a_potential_base_model` 必须继续通过；不能简单取消条件拦截。
这些测试不导入或执行 snapshot Python，不访问真实 Hub、不下载权重。

真实 Ultravox 的静态 draft 验证不下载权重、不证明所有动态依赖完备或大模型推理成功；GPU／profiler
需各自取得运行证据，不能从这项 CPU fixture 验证外推。

生产模型声明与服务镜像的完整链路可使用 [Iris 示例](Runtime_Compatibility.md#本地模型声明与自定义-pipeline)，
按 `resolve → interface validation → prepare runtime → runtime validation → matrix measurement` 分层验收；
接口检查、Smoke、预热、正式行和 profiler 结果分别计数，Smoke 不写正式 CSV。

### 无 Torch 运行时验收

```bash
.venv/bin/python scripts/check_runtime.py --profile onnxruntime-cpu \
  --basic-e2e --output-dir internal-testing/onnx-runtime
```

ONNX 的三个 profile 按 runtime 选择专用测试集，不按 family 误选 Torch 用例。
`.github/workflows/ci.yml` 的独立 ONNX CPU job 在 GitHub-hosted runner 必跑上述命令；
它检查 Torch、Transformers **未安装**，并要求 ORT/ONNX/Pillow/Tokenizers 等实际可导入。
任何指定测试模式为空、skip、expected failure、依赖缺失、超时、输出／必需字段错误或清理失败
都会失败。普通宿主测试仍可明确 skip 缺失的可选推理依赖，不代表容器通过。
`--test-pattern` 可重复覆盖默认测试，报告记录每个 pattern 的发现数；专用 CI 不覆盖默认集。

离线 fixture 固定 IR10/opset17，真实执行表格、图像、多输入文本，覆盖数值参考、动态维度、
样本顺序、dtype/shape 与固定形状拒绝；不会下载 Hub 模型。basic 回归复用生产 client、
ResourceUsageMonitor、CSV 合并及 audit，验证应用延迟、原有 batch/latency 吞吐口径、真实
cgroup CPU/内存、actual workload、输出验证和能力状态。合成图只证明接口与执行链路。
验收脚本将能力报告中的实际主机环境身份同步写入 `static_meta.json` 和 `run_state.json`，
与 client 写入 CSV 的身份交叉核对；不根据 CSV 猜测主机，也不放宽环境不一致的审计错误。
`test_onnx_basic_audit.py` 在 Docker/HTTP 边界模拟下覆盖 Native Linux、WSL2 的身份保存和不一致拒绝，
保留真实 CSV 合并及审计链路；此测试不代替上述容器端到端验证。
`--basic-e2e` 依次执行表格、图像、文本三种场景，每种包含一次 warmup 和两次正式请求；
任一场景失败即返回失败。图像检查原始与处理后尺寸，文本检查具名整数输入及实际 token 数。
真实 ORT、WordPiece tokenizer、HTTP probe 与既有规划器的联合回归还覆盖超限拒绝和自动尺度规划。
真实 HTTP 回归还验证异步等待、后台失败、超时及请求内不执行输出验证。
服务只发布 loopback 端口，不使用 privileged、Docker socket 挂载或硬件 runner；完成或失败后
清理本次容器。已有 `.github/workflows/hardware.yml` 仍仅手动触发。

两个预训练小模型可另行验证；第一步联网准备，第二步在同一无 Torch 环境断网执行。
以下目录需尚未存在或为空，重复检查请换新目录：

```bash
.venv/bin/python examples/onnxruntime/real_models.py prepare mnist \
  --directory internal-testing/pretrained-mnist
.venv/bin/python examples/onnxruntime/real_models.py prepare bert-tiny \
  --directory internal-testing/pretrained-bert-tiny
runtime_image=$(.venv/bin/python -c 'import json; print(json.load(open("internal-testing/onnx-runtime/runtime.json"))["image_id"])')
docker run --rm --network none --cpus 2 --memory 2g \
  -e ACPROF_RUNTIME_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD:/workspace:ro" -w /workspace "$runtime_image" \
  python examples/onnxruntime/real_models.py validate mnist \
  --directory internal-testing/pretrained-mnist
docker run --rm --network none --cpus 2 --memory 2g \
  -e ACPROF_RUNTIME_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD:/workspace:ro" -w /workspace "$runtime_image" \
  python examples/onnxruntime/real_models.py validate bert-tiny \
  --directory internal-testing/pretrained-bert-tiny
```

脚本保留来源、许可证说明、revision 和制品哈希，参考检查使用独立的 ONNX ReferenceEvaluator。
它验证一个样例的数值一致性，不执行训练数据准确率评测，也不输出正式性能结果。
真实 full 矩阵仍需相应硬件和权限；basic 成功不能替代 full 指标验收。

## TUI 与终端证据

终端兼容验收面向 Windows Terminal、PyCharm / JetBrains Terminal、VS Code Terminal、
Linux 原生终端、SSH 会话及浏览器 Web Terminal。SSH 只传输终端数据，记录结果时仍需注明
客户端终端、版本、字体和窗口列数 / 行数；每种环境分别标记通过、失败或未验证。
至少检查非浮层控件不重叠、边框可辨认、中文无乱码、`Tab` / `Input` / `Select` 可操作，
以及缩放后的布局和焦点可达性。字形能力有限的终端应检查基础 Unicode 线框和键盘路径，
不能依赖 `tall` 块状边框无缝拼接；纯 ASCII 终端不属于当前中文 TUI 的支持范围。

使用 pytest 原生 async 测试和 `pytest-asyncio`、Textual `run_test()` / `Pilot` 和临时 `settings_path`。
普通界面测试使用 `tests/tui_fixtures.py` 的就绪环境替身，保留启动状态机但不触发真实硬件探测；
启动 preflight 专项使用真实 `AcprofTui` 和 thread worker，只模拟 host diagnostics 边界。
快照与验证 runner 的测试同样隔离 host 探测；runner 实际启动子进程前等待启动检查，检查失败则保存失败画面并退出。
尺寸覆盖用户报告的场景，并按布局变更检查 `80×24`、`120×30`、`150×45` 及运行中 resize。
七个页面的标题或状态摘要和底部操作栏应保持可见；次要／导航动作在左下角，主要操作在右下角，
内容滚动、切换语言和隐藏快捷命令框后仍可点击。检查按钮标签完整、左右边缘未被容器裁切、
`Tab` / `Shift+Tab` 按左右分组顺序切换；实验页切入高级参数及监控页放大日志后仍满足该布局。
按钮颜色检查主题切换、悬停、聚焦、禁用与恢复；删除和终止的红色、补采的黄色不能在交互中丢失，服务就绪和保存成功使用绿色。
镜像树路径高亮检查最终屏幕的连接线颜色，覆盖点击、方向键、折叠、搜索、失焦和中英文/深浅主题切换，防止祖先线未重绘或其它分支误亮。
镜像详情检查摘要与诊断分离、默认折叠、长包清单的末项可达、点击/Enter 展开，以及换行选择、空筛选、语言切换和缩放时的折叠状态；这些操作不得触发额外 Docker 查询。
详情分隔条检查上下拖动的实际高度变化、拖出边界后的最小可见区域、按键调整与默认值恢复、缩窗后再放大的手动高度和阅读位置；切换页面、采集开始、失去捕获、缩放或按 `Esc` 后均须释放鼠标，三个镜像视图与中英文均应可用。
镜像自动刷新检查首次打开、定时更新、失败重试与有效勾选/浏览位置保留；验证后台页面、确认框、并发操作和测量窗口不启动扫描，以及任务结束后恢复。
退出阶段还需验证已排队的 timer 和 worker 回调：Textual 停止应用后、控件部分卸载而 `on_unmount` 尚未执行时，不再扫描或访问页面控件。
`test_tui_interaction.py` 在退出期间主动投递计时回调，检查耗时控件卸载后无异常且定时器释放；
同时保留运行时刷新、正式测量期间暂停和测量结束后恢复的断言，无需延长固定等待。
此边界依据锁定的 [Textual 8.2.8 退出生命周期](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/app.py)
和 [Timer.stop](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/timer.py)（MIT），
复用现有 API，不增加依赖或测量期刷新。
页面切换、挂载和布局更新后等待框架处理事件，再判断点击和焦点，不用堆叠固定 `sleep` 掩盖竞态。
设置持久化测试先进入实际设置页，按锁定版本的
[Textual 测试流程](https://github.com/Textualize/textual/blob/v8.2.8/docs/guide/testing.md)
等待初始化和控件变更事件完成，确认主题等偏好已应用后再保存并检查文件。
`scripts/run_tui_validation.py` 在子进程结束后，通过 `call_after_refresh` 等待监控页显示且日志控件
进入实际屏幕布局，再保存 `tui-finished.svg` 并返回退出码。排队的焦点事件若切回配置页，辅助入口
会重新选中监控页并等待刷新；10 秒内未就绪则明确报错。`test_tui_validation_runner.py` 覆盖真实子进程、
首次刷新前立即完成、排队的页面切换及启动失败；这些检查仅验证辅助入口的
headless 画面，不代表真实采集或用户终端已通过。
Textual 8.2.8 的 `Pilot.pause()` 可能在返回前的布局刷新中才排入 `Hide`；隐藏后的鼠标捕获断言需继续等待目标控件的 `wait_for_refresh()`，并设置超时，确保已排队的事件完成处理。
拖动中断的各个场景使用独立 `run_test()` 应用实例，避免某次断言失败遗留的页面或 busy 状态引发连带失败。
光标阶段与点击、编辑后恢复的测试只将光标闪烁 Timer 以 `pause=True` 创建，由测试显式推进阶段，
并等待输入框刷新完成后断言；重复点击同一位置也应恢复可见阶段。真实 0.5 秒闪烁间隔及无额外重绘
由独立的 idle-cursor 测试验证，避免较慢的 `Pilot` 操作跨过闪烁周期而误报失败。

Headless 能检查布局、键盘路径和输出状态；SVG、tmux 与真实 VS Code/SSH 终端是不同证据。
通知回归需显式使用 `run_test(notifications=True)`；默认测试模式不显示通知，无法发现浮层缺字。
隐藏快捷命令栏后让通知覆盖底部按钮，核对最终 compositor 输出和终端更新字节中的汉字完整性；
只断言通知原文或 Toast 自身的内容不能证明叠加后的显示正确。
颜色回归还需检查实际渲染器输出的 SGR 颜色序列：Textual 的 SVG 导出固定使用真彩色，
单看 SVG 无法发现终端输出被降级为 256 色的问题。终端颜色变更覆盖输入选中、下拉菜单、
确认按钮与深浅主题；PTY 输出仍不能代替用户客户端实际显示的验收。
原生光标、剪贴板、闪烁和宿主快捷键路由只能在相应终端确认。交付注明实际验证环境；
流程见 [TUI 回归 Skill](../.agents/skills/acprof-textual-regression/SKILL.md)，模块分工见[架构](Architecture.md#tui-与兼容维护)。

## 真实采集与实验隔离

`.github/workflows/hardware.yml` 只支持手动触发，在带 `acprof` 标签的专用 Linux x86_64 runner
上使用预先准备的 `.venv`、本机 Docker/cgroup v2、RAPL、perf 和抓包权限。不会由 PR 自动触发。
本地同一入口为：

```bash
.venv/bin/python scripts/check_hardware.py --model hf-internal-testing/tiny-random-bert \
  --task fill-mask --gpus off,on --output-dir internal-testing/hardware-smoke
```

默认每个设备 1 个 case、warmup=1、repeat=3、2 秒请求窗口及 2 秒 idle，属于短 smoke。
`command.json`、`run.log`、原始结果及 `audit.json` 一同保留；验收要求正式行为 `ok`、计划完整，
且两种延迟、CPU 能量、PMU instructions 和 GPU 模式下的 GPU 能量均为有效数值。
未通过不自动更改原参数或覆盖产物。`--compute-profile-tool` / `--execution-profile-tool`
可另测指定工具，工具字段仍需根据其计划和错误列验收，不能用主采集成功代替工具成功。

采样线程及 CLI/TUI 开销的独立对照入口和统计假设见[指标分析](Metrics.md#窗口置信区间与开销对照)。
`compare_ui.py --ui terminal` 继承当前终端，要求 stdout 为 TTY；自动化可用 `script` 分配 PTY
并保存会话。PTY、headless 和用户的 VS Code/SSH 终端须分别标明，不能互相替代。

镜像依赖变化后，主机 `.venv` 测试不能证明容器已更新；构建与复用契约见[运行兼容](Runtime_Compatibility.md#构建复用和验证)。
最小采集示例见[运行指南](Getting_Started.md#3-跑一个最小-smoke-test)。用独立输出目录运行验证，保留模型 revision、输入计划与日志。
`examples/` 下脚本是手动接口示例，不会自动运行，也不产生与正式 `acprof run` 等价的测量证据。

`internal-testing/` 用于本地临时验证和截图；原始实验结果留在对应结果目录。
普通推理成功不能证明 Torch/NCU/Massif/Nsys 都支持；每种设备、dtype 和工具分别报告实际覆盖范围。

## 参考实现与复用取舍

表单边框复用 [Textual 8.2.8 的 `solid` 字符集](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/_border.py)，
并根据 [Select 上游说明](https://github.com/Textualize/textual/discussions/4061)覆盖 `SelectCurrent`
及[源码中的展开菜单 `SelectOverlay`](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/widgets/_select.py)。
Textual 为 MIT 许可且由上游维护；这里只覆盖现有 TCSS，
保留控件尺寸和事件处理，不增加依赖、终端自动探测或测量期间的后台处理。

开发检查复用 [Ruff 官方 hook](https://github.com/astral-sh/ruff-pre-commit)
和 [pre-commit 官方基础 hooks](https://github.com/pre-commit/pre-commit-hooks)（MIT，持续维护，
所选版本支持 Python 3.10）。参考 [HTTPX 的工具配置](https://github.com/encode/httpx/blob/master/pyproject.toml)
把 Ruff 规则放在 `pyproject.toml`，通过 pytest 插件保存本项目的 evidence。
hooks 固定完整 commit SHA，CI 直接执行同一份配置，避免维护第二份检查清单；只新增开发依赖，
没有采集期间的后台进程或测量开销。

提交信息配置依据 [VS Code 官方 instructions 设置](https://code.visualstudio.com/docs/agent-customization/custom-instructions#specify-instructions-for-generated-content)
及 [Copilot Chat 的设置定义](https://github.com/microsoft/vscode-copilot-chat/blob/main/package.json)。
语言冲突依据本机 Copilot 0.67.0 和上游
[提交生成提示](https://github.com/microsoft/vscode-copilot-chat/blob/main/src/extension/prompts/node/git/gitCommitMessagePrompt.tsx)、
[ResponseTranslationRules](https://github.com/microsoft/vscode-copilot-chat/blob/main/src/extension/prompts/node/base/responseTranslationRules.tsx)
确认；生成提示还会引用近期提交风格，项目输出格式由仓库中的提交消息模板维护。
格式检查复用 [commitlint](https://github.com/conventional-changelog/commitlint) 和
[现成的 pre-commit adapter](https://github.com/alessandrojcm/commitlint-pre-commit-hook)（均为 MIT，
由上游维护），保留现有 pre-commit 管理方式；adapter 固定完整 commit SHA，CLI 和规则包固定
直接版本，不引入 Husky 或项目级 npm package。仅在提交和手动校验时运行。

结果原子发布采用 [CPython 的 tempfile](https://github.com/python/cpython/blob/main/Lib/tempfile.py)
和标准库文件同步、替换机制；实验身份参考 [ASV 的结果管理](https://github.com/airspeed-velocity/asv/blob/main/asv/results.py)。
两者的通用做法与现有 CSV/目录协议兼容，恢复仍按 AC-Prof 的 case 与测量窗口实现。
依赖解析复用持续维护的 [uv](https://github.com/astral-sh/uv)（MIT / Apache-2.0），只在更新锁时使用。
指标元数据参考 [Prometheus Python client](https://github.com/prometheus/client_python)（Apache-2.0）的类型与单位声明，
窗口区间参考 [SciPy bootstrap](https://github.com/scipy/scipy/blob/main/scipy/stats/_resampling.py)（BSD-3-Clause）的重采样方法。
本项目只需离线登记表和均值区间，使用标准库实现，无需在采集服务加入 exporter 或 SciPy 依赖。
profiler 调研了 [NVIDIA nsight-python](https://github.com/NVIDIA/nsight-python)（Apache-2.0）；其 kernel profiling 接口
不替代现有完整请求和旁路 probe 契约，因此保留 CLI/CSV 集成，提取纯解析与环境发现模块。
这些选择不增加正式测量窗口内的服务或网络调用，工具和环境验证均在采集前后进行。

生成文档的编码契约参考 [CPython 3.10 pathlib](https://github.com/python/cpython/blob/3.10/Lib/pathlib.py)
（PSF 许可）和 [PEP 597](https://github.com/python/peps/blob/main/peps/pep-0597.rst)，以及
[PyPA 对 README 显式使用 UTF-8 的修复](https://github.com/pypa/packaging.python.org/pull/682)。
采用标准库的显式编码、LF 写入与 `EncodingWarning` 回归保护；兼容 Python 3.10+，
不复制上游代码、不增加依赖，仅影响开发文档生成和检查。

主机分片参考 [pytest plugin hooks](https://github.com/pytest-dev/pytest/blob/8.4.x/src/_pytest/hookspec.py)
（MIT）和 [Textual 测试配置](https://github.com/Textualize/textual/blob/main/pyproject.toml)（MIT），
复用官方维护的 collection/report/session hooks 与 async fixture 生命周期；不复制源码。
[pytest-asyncio 清理讨论](https://github.com/pytest-dev/pytest-asyncio/issues/222)提示需核对
pending task、async generator 与 executor 的退出；锁定版本并保持 function loop scope。
项目独立实现 schema v1 聚合与排序分片，避免新增通用报告依赖。所有开销只在测试进程中，
不会进入正式测量窗口。涉及 `sys.modules` 隔离的测试，先用
`importlib.import_module` 获取真实目标再 `patch.object`，避免 Python 3.10 的字符串 patch
沿父包属性找到已脱离导入缓存的模块；请求完成、异常和超时断言保持原语义。

统计报告页复用 [Textual 官方 DataTable](https://github.com/Textualize/textual/blob/main/docs/widgets/data_table.md)
及 [Worker API](https://github.com/Textualize/textual/blob/main/docs/guide/workers.md)（MIT，官方持续维护），
已在项目使用的 Textual 8.2.8 中验证；不增加表格库或统计依赖。
窗口统计调用既有 CLI，JSON 读取在后台执行；只在用户操作和任务完成时更新表格，采集期间禁止启动，
避免给正式窗口增加轮询或统计计算。回归覆盖三种终端尺寸、中英文切换、失败恢复、原 CSV 不变及测量互斥。
报告保存参考 [pytest-benchmark 的文件存储](https://github.com/ionelmc/pytest-benchmark/blob/master/src/pytest_benchmark/storage/file.py)
（BSD-2-Clause）和 [Joblib 的稳定内容表示](https://github.com/joblib/joblib/blob/main/joblib/hashing.py)
（BSD-3-Clause）；仅借鉴思路，使用标准库比较完整 JSON、Linux 目录锁与既有原子写入，不增加依赖或测量开销。
`test_stats.py` 覆盖时间戳命名、旧 UUID 报告复用、格式无关的内容比较、参数/CSV/统计值变化、损坏文件、重名和并发发布；
TUI 回归核对重复计算时打开已有文件、显示中英文提示及控制恢复。

表头拖动复用 [Textual DataTable](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/widgets/_data_table.py)
的列元数据、渲染与鼠标捕获。沿用官方维护的 MIT 依赖，无额外包、后台轮询或测量窗口内的诊断。
当前 8.2.8 没有公开的列宽 setter；兼容逻辑集中在 `tui/table.py`，调整列宽时清理渲染缓存并更新虚拟尺寸。
升级 Textual 时需运行拖动回归，覆盖中英文 cell 宽度、固定列与横向滚动、表头点击排序、释放后的 Click、
拖出表格、禁用/隐藏/清空时的鼠标释放，以及三种终端尺寸下的会话列宽保留和采集锁定。
末列右沿在完整显示、横向滚动及单列表格中均不显示手柄或捕获拖动，普通表头点击仍有效。
报告页覆盖数据更新、报告重建和语言切换；镜像树另检查父子行的大小、未知值和容器数量与表头左对齐、横向滚动同步和折叠状态保留。

## 文档与 Skill 检查

检查新增文件也包括被 Git 忽略的文件；`git diff --check` 只覆盖已跟踪差异，不能替代完整文件清单。
迁移章节时核对原有锚点、相对链接、代码中的文档引用及字段表是否有遗漏；代码示例中的路径以注明的执行目录为准。
技能格式可用已安装 `skill-creator` 的 `scripts/quick_validate.py <skill-dir>` 检查；该工具是开发辅助，不是项目运行依赖。

交付说明实际执行的命令、结果、跳过原因及未验证范围。只改文档时，不宣称完成真实 Docker/GPU 或用户终端验证。

开销与负载契约测试包括 `test_overhead_contract.py`、`test_overhead_entrypoint.py`、
`test_execution_conditions.py`、`test_load_protocol.py`；外部 Docker／硬件边界模拟，内部编排真实执行。
负载测试使用本机临时 HTTP 服务证明并发与真实连接复用；它不等于目标模型服务支持 keep-alive。
跨实验统计使用 `test_independent_comparison.py`；CI 完整性使用 `test_report_aggregation.py`。
所有本地测试／真实 PCAP 复现均应避开另一个正式采集窗口，不绕过主机测量锁。

本轮配置与统计设计参考 [pyperf](https://github.com/psf/pyperf)（MIT），生命周期参考
[CPython ExitStack](https://github.com/python/cpython/blob/3.12/Lib/contextlib.py)（PSF），
负载计划参考 [MLCommons LoadGen](https://github.com/mlcommons/inference/blob/master/loadgen/test_settings.h)
（Apache-2.0）。仅借鉴方法，使用标准库和已有采集器，不复制框架或增加采集依赖。
CI 复用官方 [download-artifact v4](https://github.com/actions/download-artifact/tree/v4)（MIT），
与现有 upload-artifact v4 配对并固定 SHA；汇总离线运行，不增加采集开销。
