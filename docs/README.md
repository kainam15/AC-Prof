# 项目文档导航

AC-Prof 的长期知识在本目录按主题维护。先按任务选择一篇，再搜索相关标题或符号；链接是按需阅读入口，不表示需要加载整份文档。

## 按任务查阅

| 当前任务 | 权威文档与范围 |
| --- | --- |
| 安装、准备主机、跑通 CPU/GPU/ONNX 示例或资源矩阵 | [安装与运行](Getting_Started.md)：环境、认证、smoke test、最大输入探测与正式实验 |
| WSL2 开发、部分采集与环境比较边界 | [WSL2 平台](platforms/wsl2.md)：环境身份、能力矩阵、数据隔离与 Native validation |
| 使用终端界面、快捷键、日志、镜像页或设置 | [TUI 用户指南](TUI.md)：页面操作与自动记忆；设置文件协议见 CLI 参考 |
| 理解项目来源、历史架构、默认参数来源或遗留行为 | [项目来源与演进](Project_Origin.md)：原始仓库、扩展范围与历史参考边界 |
| 定位代码、重构模块、维护兼容入口 | [代码架构](Architecture.md)：职责、依赖方向与兼容设计 |
| 修改采集窗口、产物或冷启动 | [采集协议](Profiling_Protocol.md)：生命周期、文件与 schema、请求数、时间预算、冷启动边界 |
| 查字段、历史数据、绘图或延迟拟合 | [指标与结果分析](Metrics.md)：分析范围、单位、公式、归一化、图表与模型 |
| 查完整列协议、审计和置信区间 | [指标登记表](Metric_Reference.md)：由代码生成的单位、来源与窗口；[分析入口](Metrics.md#窗口置信区间与开销对照) |
| 解释功率、能耗、idle 或归因误差 | [能耗测量](Energy_Measurement.md)：RAPL、NVML、估算 vCPU 与适用限制 |
| 选择或排查 GPU/CPU profiler | [分析器](Profilers.md)：Torch、NCU、Massif、Nsys 的窗口、采样、成本与失败 |
| 排查 OOM、cgroup、抓包、空值或部分结果 | [运行排障](Troubleshooting.md)：证据分类、恢复入口与实时状态检查 |
| 新增模型/backend、改依赖或镜像 | [运行兼容](Runtime_Compatibility.md)：任务目录、加载接口、环境与构建契约 |
| 识别镜像类型、查看复用与空间释放规则 | [镜像管理与清理](Runtime_Compatibility.md#镜像管理与清理)：类型、共享层、构建缓存与删除范围 |
| 查参数、workload 清单、通知或 TUI 设置协议 | [CLI 与设置](CLI_Reference.md)：选项、输入规模、企业微信通知、持久化和历史兼容 |
| 修改 Python、配置开发检查、选择测试、做 TUI 回归 | [测试指南](Testing.md)：[PyCharm MCP 工具约定](Testing.md#python-修改工作流)、Ruff/pre-commit、硬件冒烟与验证边界 |
| 新建或修改 Skill、`AGENTS.md` | [编写规则](#skill-与-agent-文档编写)：触发条件、按需读取、完成与确认边界 |

根 [README.md](../README.md) 是英文正式主文档，提供项目介绍和首次运行路线；简体中文版位于 [docs/i18n/README_zh-CN.md](i18n/README_zh-CN.md)。
修改任一语言的 README 时，必须在同一次改动中同步另一版本的对应内容、命令、链接和排版，保留各自语言及正确的相对路径；详细专题目前以简体中文维护。完整安装与实验示例在[运行指南](Getting_Started.md)，界面操作在 [TUI 用户指南](TUI.md)。
实现与默认值用当前代码和 `--help` 核对；历史实验解释以当次产物的版本、计划、日志和来源为准。

## 文档、规则与流程的分工

[安装包、standalone 与发布](Distribution.md)维护 wheel 资源、工作目录、GitHub Release 和 GHCR 发布约定。

| 层级 | 内容 | 读取时机 |
| --- | --- | --- |
| 根 [AGENTS.md](../AGENTS.md) | 项目地图、全局约束、验证入口和完成标准 | 每次任务 |
| 目录级 `AGENTS.md` | 该目录独有的实现或产物约束 | 进入相关目录时 |
| 本目录专题 | 项目是什么、为什么这样设计、协议如何定义 | 任务涉及该主题时 |
| `.agents/skills/*/SKILL.md` | 可复用的多步骤执行流程 | 符合技能描述时 |
| 实验目录、命令输出 | 该次运行的事实与当前机器状态 | 实时、定向检查 |

一个主题只维护一处完整定义，其他文档保留必要摘要或链接；相关小主题使用章节，不为每个字段创建文件。
历史审计 `reviews/` 是带日期和输入指纹的证据快照，不是现行协议，也不证明当前机器状态。
临时任务计划和验证输出放在会话或 `internal-testing/`，不要写进长期 Agent 规则。

## Skill 与 Agent 文档编写

新建或修改 Skill、`AGENTS.md` 时，参考
[OpenAI：重新思考 GPT-6 Astra 的技能与提示词](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)。

- 描述简短、适用场景明确；根入口只保留项目地图、全局约束、任务导航和完成标准，详细资料按需读取。
- 规则聚焦任务或项目特有约束，避免重复指令、全量必读清单和不必要的固定流程。
- 目录独有规则放在局部 `AGENTS.md`；跨目录规则按任务从根入口链接到权威章节，避免遗漏根脚本与测试。
- 明确完成标准和需要确认的边界；在已授权范围内完成实现、相关验证和修复，验证范围与改动相称。
- 移动内容时修复链接和章节锚点，并按[文档与 Skill 检查](Testing.md#文档与-skill-检查)验证。

Codex 的 [AGENTS.md 发现实现](https://github.com/openai/codex/blob/1cc7e2361237ce7244430ee1d581c77f95c57ac8/codex-rs/core/src/agents_md.rs)
会沿项目根目录到工作目录收集指令；普通 Markdown 链接不是自动全文导入。
这里只借鉴按任务导航的组织方式，不复制上游代码，不增加依赖或测量期开销。

## 可复用流程

| 任务 | Skill |
| --- | --- |
| 新增指标或产物协议 | [acprof-schema-change](../.agents/skills/acprof-schema-change/SKILL.md) |
| 审计结果、OOM 或 profiler 异常 | [acprof-result-audit](../.agents/skills/acprof-result-audit/SKILL.md) |
| TUI 布局和交互验证 | [acprof-textual-regression](../.agents/skills/acprof-textual-regression/SKILL.md) |
| 新增模型、adapter 或 backend | [acprof-model-adaptation](../.agents/skills/acprof-model-adaptation/SKILL.md) |
| 完整 profiling 或 benchmark 实验 | [acprof-profiling-workflow](../.agents/skills/acprof-profiling-workflow/SKILL.md) |
| 镜像身份、复用、共享层与空间估算 | [acprof-docker-audit](../.agents/skills/acprof-docker-audit/SKILL.md) |
| GitHub Actions 失败取证与定向复现 | [acprof-ci-triage](../.agents/skills/acprof-ci-triage/SKILL.md) |
| 指定模型集或 Hub 榜单的兼容性矩阵 | [acprof-compatibility-audit](../.agents/skills/acprof-compatibility-audit/SKILL.md) |

OOM 排障复用结果审计流程；benchmark 与完整 profiling 共用实验流程，不再分别创建相近技能。
兼容性审计负责评估与分层验证，适配流程负责已授权的实现修改；镜像审计不默认执行清理。
各 Skill 的案例和报告模板按需读取，产出写到本次任务目录，协议与字段定义仍以 `docs/` 专题为准。

CI 流程参考公开维护的
[openai/skills gh-fix-ci](https://github.com/openai/skills/blob/main/skills/.curated/gh-fix-ci/SKILL.md)
（Apache-2.0），只借鉴日志取证方式，复用现有 GitHub 工具与项目 runner，不复制上游代码或增加依赖。

## Claude 与通用 Agent 指令

根 [CLAUDE.md](../CLAUDE.md) 只用 `@AGENTS.md` 引入通用规则。Claude 特有差异才写在导入后；
以后若添加目录级 `CLAUDE.md`，同样只导入对应 `AGENTS.md`。不要在根文件批量导入全部子目录、专题或 Skills。
Markdown 导航链接不做全文导入；Claude 的 `@` 导入会直接加载正文，语义见[官方说明](https://code.claude.com/docs/en/memory#agentsmd)。

这一组织方式参考 [AGENTS.md 项目](https://github.com/agentsmd/agents.md)和
[Skill 的分层披露说明](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md#progressive-disclosure)。
两者有公开维护的仓库，许可证分别为 MIT 和 [Apache-2.0](https://github.com/anthropics/skills/blob/main/skills/skill-creator/LICENSE.txt)。
这里只采用 Markdown 导航和按需读取的组织方式，沿用现有工具；不引入文档框架、自动同步依赖或测量期进程。
