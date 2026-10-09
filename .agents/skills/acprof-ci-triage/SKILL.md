---
name: acprof-ci-triage
description: 排查或修复 AC-Prof GitHub Actions 失败，定位真实断言并定向复现；普通本地测试不触发。
---

# AC-Prof CI 失败复现

## 入口与范围

从用户提供的 run、PR 或错误截图定位失败，依据当前[CI 与环境测试](../../../docs/development/testing.md#ci-与环境测试)选择本地入口。用户只要求诊断时保持只读；已要求修复时连续完成定位、修改与相关验证，不逐步确认。提交、推送、PR 和远端重跑遵循已有授权。

## 工作流程

1. 固定仓库、run ID、attempt、commit SHA、job、Python/依赖版本和失败时间。优先用已可用的 GitHub 工具；有 `gh` 时可复用 CLI，不为了读日志另装工具或重复登录。
2. 读取失败 step 的完整相关日志与已有 artifacts。界面显示的 step 名可能包含多条命令，定位实际返回非零的命令、测试 ID、断言和前置错误。分页取齐相关 jobs，不能把列表第一页当成全体。
3. 对照失败 SHA 与当前工作区。保护既有修改，必要时在隔离副本复现；使用项目 evidence runner 和同类环境定向运行，记录修复前失败。只在失败与环境差异需要时扩大验证。
4. 根据契约区分实现缺陷、过期 fixture/mock、生成文档或快照差异、竞态和基础设施故障。修正真正错误的一侧，保留有意义的断言；不扩大 ignore、放宽阈值或盲目更新快照。
5. 执行受影响测试与必需检查，保存完整输出及退出码。核对跳过项、分片完整性和测试归属；IDE 检查不能替代 evidence JSON。
6. 报告根因、改动、证据与剩余缺口。本地通过和同 SHA 的远端通过分别记录，未运行的 Docker/GPU、真实终端或远端重跑不能标为通过。

## 按需资源

- 常见 mock、文档、快照与测量锁问题见[失败模式](references/failure-patterns.md)。
- 交付多 job 排查结果时使用[报告模板](assets/triage-report.md)。

## 参考与复用

参考 [OpenAI gh-fix-ci](https://github.com/openai/skills/blob/main/skills/.curated/gh-fix-ci/SKILL.md) 的日志与 job 取证方式。上游为 Apache-2.0，未复制其代码；本流程增加 AC-Prof evidence、生成文档和分片边界，兼容无 `gh` 的连接器环境，不引入依赖或测量开销。
