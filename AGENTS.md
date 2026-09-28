# 项目地图与协作规则

AC-Prof 对 Docker 中的 Hugging Face 推理服务进行可复现分析，输出延迟、能耗、资源指标与实验产物。

## 项目来源

本仓库是原始 AC-Prof 项目的延续与大幅扩展。
涉及历史架构、测量语义、默认参数来源或遗留行为时，先阅读[项目来源与演进](docs/Project_Origin.md)。

## 仓库地图

| 位置 | 职责 |
| --- | --- |
| 根 CLI → `acprof/cli/` | 采集、探测、补采、绘图与 TUI 入口 |
| `acprof/host/`、`acprof/container/`、`acprof/workloads/` | 主机编排、容器推理、确定性输入 |
| `acprof/monitors/`、`acprof/packet/` | 原始测量、抓包与合并 |
| `acprof/analysis/`、`acprof/plotting/`、`acprof/tui/` | 结果分析、绘图、终端界面 |
| `dockerfiles/`、`tests/`、`examples/` | 镜像、回归测试、手动示例 |
| `docs/`、`.agents/skills/`、`internal-testing/` | 长期知识、可复用流程、临时验证 |

## 全局规则

- 使用简体中文回复；`AGENTS.md` 的标题与说明使用简体中文，保留技术标识。
- 修改 `README.md` 或 `docs/i18n/README_zh-CN.md` 时，必须在同一次改动中同步另一语言版本的对应内容、命令、链接和排版，保留各自语言及正确的相对路径。
- 不修改 `docs/Original_Project_Definition.md`；用户未明确要求时不提交 Git。
- 功能或结构改动前先检索 GitHub，评估兼容性、许可证、维护、依赖成本与测量开销，说明复用取舍。
- 使用已有 `.venv`、Python 3.10+；正式采集要求原生 Linux、本机 Docker Engine、cgroup v2。
- 不得通过扩大忽略规则掩盖新问题。
- 保持指标归因和可复现口径；界面活动、绘图、通知与额外诊断不进入正式测量窗口。
- 凭据放在被 Git 忽略的 `.env.local`，不得写入文档或提交密码、令牌、webhook。
- GPU/磁盘余量、Docker 状态、Git 分支和进程按需实时检查；临时计划不进入长期 Agent 文档。

## 按任务读取

进入目录前检查适用的局部 `AGENTS.md`。按下表的任务触发条件读取对应章节，长文档先用 `rg` 定位；不批量加载，已加载且未变化的规则无需重读。

| 任务 | 入口 |
| --- | --- |
| 修改 Python（含根脚本与测试） | [PyCharm MCP 工具与验证约定](docs/Testing.md#python-修改工作流) |
| 新建或修改 Skill、`AGENTS.md` | [编写规则](docs/README.md#skill-与-agent-文档编写) |
| 选择验证范围、命令或开发工具 | [验证范围](docs/Testing.md#验证范围)、[开发检查](docs/Testing.md#开发质量检查)、[自动化入口](docs/Testing.md#自动化验证入口) |
| 安装、运行、参数 | [快速开始](docs/i18n/README_zh-CN.md#快速开始)、[CLI 与设置](docs/CLI_Reference.md) |
| 架构或模块重构 | [代码架构](docs/Architecture.md) |
| 协议、冷启动、字段变更 | [采集协议](docs/Profiling_Protocol.md)、[指标](docs/Metrics.md) → [变更流程](.agents/skills/acprof-schema-change/SKILL.md) |
| 能耗、OOM、cgroup、结果异常 | [能耗](docs/Energy_Measurement.md)、[排障](docs/Troubleshooting.md) → [审计流程](.agents/skills/acprof-result-audit/SKILL.md) |
| 新增或修复模型/backend、依赖环境 | [运行兼容](docs/Runtime_Compatibility.md) → [适配流程](.agents/skills/acprof-model-adaptation/SKILL.md) |
| 模型集或 Hub 榜单兼容性评估 | [兼容性审计](.agents/skills/acprof-compatibility-audit/SKILL.md) |
| 镜像身份、重建原因与空间估算 | [Docker 审计](.agents/skills/acprof-docker-audit/SKILL.md) |
| GitHub Actions 失败定位与修复 | [CI 排障](.agents/skills/acprof-ci-triage/SKILL.md) |
| profiling、benchmark、GPU profiler | [实验流程](.agents/skills/acprof-profiling-workflow/SKILL.md)、[分析器](docs/Profilers.md) |
| TUI、焦点、日志、设置 | [交互说明](docs/i18n/README_zh-CN.md#交互式终端界面) → [回归流程](.agents/skills/acprof-textual-regression/SKILL.md) |
| 绘图、文档维护 | [结果分析](docs/Metrics.md#图表与延迟拟合产物)、[文档分工](docs/README.md) |

## 完成标准

- 在已授权范围内完成修改、相关验证和必要修复，无需逐步确认；验证范围与改动相称。
- 处理目标行为及相关失败路径，保持当前协议；已有替代实现的历史兼容代码删除，旧协议明确报错；按改动范围取得实际验证证据。
- 更新对应 `docs/` 权威专题及受影响的摘要、示例和链接；Skill 只维护执行流程。
- 检查完整变更清单（含新增、被忽略文件）；说明验证结果和未覆盖的 Docker/GPU 或终端范围。
