# 验证策略与 Python 修改工作流

[← 返回专题目录](testing.md)

## 验证范围

WSL2 是开发与 PARTIAL 采集平台。pytest 注册 `unit`、`wsl`、`native_linux`、`hardware`
markers；未标集成边界的测试归入 unit。WSL 默认执行 `.venv/bin/python -m pytest -m "not native_linux"`，
Native Linux 无需此平台筛选，具体测试范围按下表选择。仅显式平台集成测试按真实环境 skip，不因 WSL 跳过普通代码异常。
模拟 sysfs/NVML 的单元测试仍需执行；hardware marker 本身不隐藏失败。
环境检测、能力矩阵、历史 unknown、CSV 合并与比较隔离回归在 `test_environment_policy.py`、
`test_result_comparison.py`，详见 [WSL2 支持范围](../platforms/wsl2.md)。
修改 RAPL、PMU/perf、cgroup、NVML、CPU topology、affinity、cold start 或 energy 时必须报告
`Native validation: verified / required / not applicable` 中的一项；WSL/mock 通过不能替代 Native 证据。

| 改动 | 应取得的证据 |
| --- | --- |
| 文档、导航或链接迁移 | 本地文件与章节锚点可达、旧入口仍可跳转、代码块与差异格式正确；无需为措辞运行模型 |
| Skill | frontmatter、名称与描述匹配、相对路径、流程边界和可用验证器的格式检查 |
| 单个逻辑或失败路径 | 能观察目标行为的相关 pytest；修复缺陷时先复现，再验证修复 |
| 模块搬迁、依赖方向或兼容入口 | 当前入口成功、已删除入口拒绝、受影响调用链的 pytest、相关 CLI 帮助和编译；mock 放到函数实际查找依赖的模块 |
| 指标或产物协议 | 独立推导的期望值、当前 schema 成功与旧 schema 拒绝、缺失/失败/不适用字段和受影响消费者 |
| 模型、backend、依赖或 Dockerfile | 路由与离线加载测试、镜像构建、所声明设备的真实推理；profiler 分别验证 |
| TUI | 受影响的交互与尺寸检查；原生终端问题还需对应终端证据 |

局部模块搬迁先验证受影响的入口、调用者和消费者。改动涉及公共包初始化链、公共命令分发、
测试基础设施或多个任务族共用的执行路径，或定向验证后仍无法界定影响范围时，执行全套 pytest。
CI 的完整回归与分片验收要求保持不变，见 [CI 与环境测试](ci.md#ci-与环境测试)。

下载策略的定向回归包含 `test_download_network.py`、`test_model_store.py`、
`test_network_preflight.py`、`test_dependency_download_cache.py`、`test_lock_compiler.py`
与 `test_tui_downloads.py`。`test_hf_auto_download.py` 以真实 Hub SDK 加受控 HTTP transport
覆盖镜像 → Xet／区域 CDN、镜像 → 官方 Hub → CDN、未知第三方拒绝、鉴权隔离、HEAD probe、
部分缓存恢复或安全重下后的内容身份、流中断后的 Range 续传与完整内容校验、离线模式零请求、
模型错误分类，以及系统代理和透明上游不推断；不下载真实权重。
Host HF transport 初始化会禁用 SDK user-agent 遥测，避免 SDK 为识别 agent harness 而额外发出 GET，
保证轻量模型文件探测只发 HEAD 请求。
`test_modelscope_source.py` 覆盖仓库 commit 固定、文件 SHA256、source 隔离及旧设置迁移；
`test_tui_auto_download.py` 验证失败正文／折叠诊断和显式切换来源（含取消、子进程退出及独立结果目录）。
Model Store 覆盖完整命中零模型网络请求、默认 ref 与显式 revision 隔离、原始完整仓库上下文、缓存配置大小与 SHA256、空间、预算、独立 dependency refs 和活动 lease。TUI 候选与入门回归还验证切换来源时清除旧自动 revision、保留手动值，以及预设迁移到 `auto` 后保留模型来源、预算和路径。
依赖下载测试使用本地 HTTP 服务覆盖客户端标识、重定向后的 HEAD 方法、SHA256 校验和缓存命中，
无需外网；锁生成测试在 Python 3.10 验证明确拒绝且不修改原锁，在 Python 3.11+ 验证镜像解析及 hash 保护。
依赖、模型、Dockerfile 变更还需分别说明新构建、已有 runtime、小型 fixture、真实 checkpoint 的验证范围。

## Python 修改工作流

修改 Python 代码时优先使用 PyCharm MCP，覆盖 `acprof/`、`packaging/`、`scripts/` 和 `tests/`。
只使用本次任务涉及的工具；纯文档修改按[文档检查](documentation-checks.md#文档与-skill-检查)验证。

| 场景 | 工具与约束 |
| --- | --- |
| 定位程序符号 | 使用 `search_symbol`；`rg` 用于文件、普通文本和配置检索 |
| 分析调用或依赖 | 优先使用 `analyze_calls`，结合源码确认动态调用；不得仅凭文本搜索推断 Python 符号关系 |
| 重命名 Python 符号 | 优先使用 `rename_refactoring`，核对引用更新和实际差异 |
| 检查修改后的文件 | 使用 `lint_files` / `get_file_problems` 检查受影响文件的 IDE diagnostics，处理本次改动引入的问题 |
| 执行 IDE 测试或 smoke test | 用 `get_run_configurations` 选择相关的已有 Run Configuration，通过 `execute_run_configuration` 执行 |
| 核对最终改动 | 使用 `git_status` 并结合 diff，检查新增、被忽略文件，确认没有混入无关变更 |

按[验证范围](#验证范围)运行相关 pytest / evidence、Ruff 及真实 workload；
命令见[开发质量检查](quality.md#开发质量检查)与[自动化验证入口](test-commands.md#自动化验证入口)。
局部修改不默认跑完整测试集；只有新改动、失败或未解决问题才扩大或重复验证。

MCP 不可用、索引不完整或没有适用 Run Configuration 时，说明限制并用源码分析和项目 CLI 入口继续；
不把空调用树当作没有依赖。pytest 缺失时先安装开发锁；IDE、pytest 与 evidence 的边界见 [PyCharm MCP 的验证边界](devtools.md#pycharm-mcp-的验证边界)。
真实 workload 缺少 Docker、GPU、模型等运行条件时，明确标为未验证，不用 IDE diagnostics 或 smoke test 代替。

## Python 文件规模审查

使用 `python scripts/check_module_sizes.py` 汇总 `acprof/` 文件的物理行数及最长函数、
静态分支与 import 提示。大于 500 行列为软目标提醒，大于 800 行列为人工审查；
`data-review` 表示源码以大段静态字典或列表为主，**不是默认拆分任务**。
`--top 0` 列出全部待审查文件，`--json` 提供结构化结果。文件大小本身不作为 CI 拒绝条件；
脚本只能在扫描失败、编码错误或源码语法错误时失败。
CI 的 lint job 会运行该报告，pytest 覆盖阈值边界、数据目录识别、长函数和错误处理。
指标和拆分顺序的解释见[Python 文件规模与拆分原则](modules.md#python-文件规模与拆分原则)。
