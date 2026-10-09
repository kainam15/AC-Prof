# AC-Prof 文档导航

文档按主题目录维护。每个概念、命令或协议只在一个权威位置完整定义；其他页面只放导航和链接。

## 主题目录

| 分类 | 权威文档 |
| --- | --- |
| **getting_started/** · 入门 | [安装与主机准备](getting_started/installation.md) |
| **usage/** · 使用 | [实验指南](usage/experiments.md) · [TUI](usage/tui.md) · [CLI 参考](usage/cli.md) · [运行排障](usage/troubleshooting.md) |
| **models/** · 模型 | [运行兼容与适配](models/runtime.md) |
| **profiling/** · 采集 | [采集协议](profiling/protocol.md) · [能耗测量](profiling/energy.md) · [Profiler](profiling/profilers.md) |
| **results/** · 结果 | [指标与分析](results/metrics.md) · [自动生成字段登记表](results/metric_reference.md) |
| **development/** · 开发 | [架构](development/architecture.md) · [测试](development/testing.md) · [发行包](development/distribution.md) · [历史来源](development/history.md) |
| **platforms/** · 平台 | [WSL2](platforms/wsl2.md) |
| **reviews/** · 历史记录 | 审计快照，不代表当前实现 |
| **i18n/** · 翻译 | [简体中文 README](i18n/README_zh-CN.md) |

[项目首页](../README.md) 负责介绍和安装入口，不复制使用手册。

## 单一事实来源

- 教程写操作，CLI 参考定义参数，采集协议定义窗口，指标登记表定义字段。
- README、导航、Skills、AGENTS 和其他专题只做简短链接，不重复维护完整命令和字段表。
- 移动文档须同步修复源码、脚本、测试及 Markdown 的相对链接和章节锚点。
- 自动生成的字段登记表以项目指标注册表为权威来源，按原有生成脚本更新。
- reviews/ 是历史证据，不能被当成现行配置。

## 文档、规则与流程的分工

[安装包、standalone 与发布](development/distribution.md)维护 wheel 资源、工作目录、PyPI Trusted Publishing、GitHub Release 和 GHCR 发布约定。

| 层级 | 内容 | 读取时机 |
| --- | --- | --- |
| 根 [AGENTS.md](../AGENTS.md) | 项目地图、全局约束、验证入口和完成标准 | 每次任务 |
| 目录级 `AGENTS.md` | 该目录独有的实现或产物约束 | 进入相关目录时 |
| 本目录专题 | 项目是什么、为什么这样设计、协议如何定义 | 任务涉及该主题时 |
| `.agents/skills/*/SKILL.md` | 可复用的多步骤执行流程 | 符合技能描述时 |
| 实验目录、命令输出 | 该次运行的事实与当前机器状态 | 实时、定向检查 |

一个主题只维护一处完整定义，其他文档保留必要摘要或链接；相关小主题使用章节，不为每个字段创建文件。
历史审计 `reviews/` 是带日期和输入指纹的证据快照，不是现行协议，也不证明当前机器状态。
本轮 [Code Health 实施记录（2026-10-02）](reviews/2026-10-02-code-health.md) 汇总 private API、模块边界、输出和 coverage 的处理结果。
临时任务计划和验证输出放在会话或 `internal-testing/`，不要写进长期 Agent 规则。

## Skill 与 Agent 文档编写

新建或修改 Skill、`AGENTS.md` 时，参考
[OpenAI：重新思考 GPT-6 Astra 的技能与提示词](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)。

- 描述简短、适用场景明确；根入口只保留项目地图、全局约束、任务导航和完成标准，详细资料按需读取。
- 规则聚焦任务或项目特有约束，避免重复指令、全量必读清单和不必要的固定流程。
- 目录独有规则放在局部 `AGENTS.md`；跨目录规则按任务从根入口链接到权威章节，覆盖开发脚本、构建 hook 与测试。
- 明确完成标准和需要确认的边界；在已授权范围内完成实现、相关验证和修复，验证范围与改动相称。
- 移动内容时修复链接和章节锚点，并按[文档与 Skill 检查](development/documentation-checks.md#文档与-skill-检查)验证。

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
