# 开发 Skills 的范围与来源

[← 开发辅助工具](devtools.md)

项目 skills 放在 `.agents/skills/` 并随 Git 维护；个人通用 skills 放在 `$HOME/.agents/skills/`，
不成为项目运行依赖。已有 `$HOME/.codex/skills/` 内容保留，不通过复制制造同名入口。
目录发现规则见 [Codex Skills](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills)。
skill 被发现不代表其外部工具已经安装、连接或完成验收。

## 项目范围

已有的模型适配、兼容性审计、采集、结果、Schema、Docker、架构、TUI 与 CI triage skills
继续维护。下列入口补充静态调研和开发流程，按任务选择，不合并成每次必跑的总流程。

| Skill | 使用场景 | 边界 |
| --- | --- | --- |
| [acprof-html-report-regression](../../.agents/skills/acprof-html-report-regression/SKILL.md) | 离线 HTML 报告、浏览器交互与导出 | 复用 Playwright/Fixture，不覆盖 Textual 或真实模型验收 |
| [acprof-hub-research](../../.agents/skills/acprof-hub-research/SKILL.md) | Hub 元数据、任务与固定 revision 调研 | 静态证据不等于真实兼容；不下载权重或执行远程代码 |
| [acprof-service-metrics](../../.agents/skills/acprof-service-metrics/SKILL.md) | 服务直方图、Prometheus 消费与 SLO | 保留测量窗口和来源；不自动部署监控平台或发起 sweep |
| [acprof-github-actions-workflow](../../.agents/skills/acprof-github-actions-workflow/SKILL.md) | 工作流、矩阵、权限和发布门禁设计 | 失败 run 诊断仍交给 acprof-ci-triage；不自动推送发布 |

这些入口使用项目现有权威文档和验证工具。不会安装新的 MCP server、浏览器、模型或推理依赖，
也不会授权硬件实验。涉及多个入口时只加载本次任务需要的部分。

## 个人通用范围

以下是本机选择的通用能力，适用于其他 Python 或界面项目；新 checkout 不自动安装个人 skills。
它们不替代仓库对 worktree、依赖、验证范围和测量语义的规则。

| Skill | 用途与取舍 |
| --- | --- |
| `async-python-patterns` | 异步任务、取消、背压与有界等待 |
| `python-testing-patterns` | pytest、Fixture、Mock 和行为覆盖；不追求机械覆盖率 |
| `python-resource-management` | 资源所有权、部分初始化与异常清理 |
| `python-type-safety` | 边界校验、Protocol 和增量类型检查 |
| `python-error-handling` | 异常传播、分类和部分失败 |
| `python-configuration` | 配置来源、验证与快照；不强制引入 Pydantic |
| `python-packaging` | wheel/sdist、安装资源与入口；不自动上传发布 |
| `uv-package-manager` | 按项目版本、依赖组和锁管理环境 |
| `pr-review-toolkit` | 按需审查错误处理、行为测试和类型不变量；不自动委派或发布评论 |
| `ui-ux-pro-max` | 本地配色、图表、可访问性与交互检索；不用于 Textual |

八个 Python skills 保留上游参考与 MIT license，入口按项目契约优先的原则精简。
PR toolkit 将三个 Claude agents 的思路适配成一个 Codex skill，保留 Apache-2.0 license，
不依赖 Claude 的 `model`、`color` 或 Task 工具。
UI UX Pro Max 保留本地数据与脚本，并将 Claude plugin 路径改为独立 skill 路径；
搜索结果中的 CDN、在线字体或框架建议不覆盖离线 HTML 的约束。

现有 `systematic-debugging`、`test-driven-development`、`verification-before-completion`、
`frontend-design` 与相关插件保留。未重复安装 Superpowers、GitHub 或 Context7；
不导入禁止 Mock 的 scenario-testing、训练/云任务流程或与当前开发无关的 Office skills。
科研数据诚实与来源规则继续由项目测量协议和结果审计维护，不引入第二套规则。

## 上游版本与本地适配

| 来源 | 固定 revision | 复用方式 |
| --- | --- | --- |
| [wshobson/agents](https://github.com/wshobson/agents/tree/9c61ab243def98f72da12abd41cfcf5bcd91af9d) | `9c61ab243def98f72da12abd41cfcf5bcd91af9d` | Python 参考内容；服务指标与 Actions 设计思路 |
| [Anthropic PR Review Toolkit](https://github.com/anthropics/claude-plugins-official/tree/b5b01ac4f889fec0c4c6bb89f27ca7c8574473f6/plugins/pr-review-toolkit) | `b5b01ac4f889fec0c4c6bb89f27ca7c8574473f6` | 错误、测试与类型审查参考，改写为按需入口 |
| [Anthropic webapp-testing](https://github.com/anthropics/skills/tree/1780c5e5a552fd4d1c9d6a7b0cab26b0c2bef064/skills/webapp-testing) | `1780c5e5a552fd4d1c9d6a7b0cab26b0c2bef064` | 先观察再操作的流程，使用项目现有脚本 |
| [Hugging Face skills](https://github.com/huggingface/skills/tree/25910b3d0ba31a2ba102f901cdc9e008c278a69c) | `25910b3d0ba31a2ba102f901cdc9e008c278a69c` | Hub 静态发现与 CLI 自描述思路，不复制高版本命令表 |
| [UI UX Pro Max](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill/tree/e59ff376a41f7ca47b7541ad5c116495cbade3ee) | `e59ff376a41f7ca47b7541ad5c116495cbade3ee` | 本地搜索数据、脚本和参考；MIT license |

升级时先比较上游 diff，再保留本地入口、路径与项目边界；不覆盖已有同名 skill。
项目入口的来源记录在 frontmatter 与参考链接中，未复制上游运行代码。
个人 skills 的版本记录在各自 frontmatter；临时安装与验证证据放入 `internal-testing/`。

验证按[文档与 Skill 检查](documentation-checks.md#文档与-skill-检查)执行，并检查名称重复、
本地相对路径及 Codex `skills/list` 的发现结果。格式与发现通过不代表所有未来任务的行为都已验证；
有可执行辅助脚本时再做与改动相称的离线 smoke。
