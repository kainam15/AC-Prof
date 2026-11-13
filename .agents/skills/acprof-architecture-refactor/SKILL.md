---
name: acprof-architecture-refactor
description: 用于 AC-Prof 的模块职责拆分、依赖方向治理、复杂度与宽泛异常收窄、渐进类型收紧及行为保持重构；纯界面微调、新模型适配或只读结果审计不触发。
---

# AC-Prof 架构治理与行为保持重构

## 使用范围与边界

处理高复杂度函数拆分、大模块职责分离、导入方向修正、跨模块 private API 清理、`BLE001` 宽泛异常收窄及 `mypy` 局部严格门禁推进。遵循项目 [AGENTS.md](../../../AGENTS.md) 与 [Python 实现约束](../../../acprof/AGENTS.md)。

纯 TUI 样式或文案微调不触发本流程；指标与产物协议变更用[指标变更 Skill](../acprof-schema-change/SKILL.md)，模型适配用[模型适配 Skill](../acprof-model-adaptation/SKILL.md)，只读结果解释用[结果审计 Skill](../acprof-result-audit/SKILL.md)。

重构期间必须保持正式测量窗口、实验 provenance、恢复契约及错误/unknown 语义不变；禁止通过扩大 `ignore`、`skip`、静默 fallback 或放宽断言掩盖问题。

## 按需查阅

| 关注维度 | 权威入口 |
| --- | --- |
| 架构分层与目录职责 | [架构总览](../../../docs/development/architecture.md#ac-prof-代码架构)、[模块职责与拆分原则](../../../docs/development/modules.md#python-文件规模与拆分原则) |
| 依赖方向、private API 与 mypy | [类型与模块边界检查](../../../docs/development/static-checks.md#渐进类型检查与边界回归) |
| Ruff 门禁、Worktree 与提交 | [开发质量检查](../../../docs/development/quality.md#git-worktree-与合并) |
| 验证范围与修改工作流 | [验证策略](../../../docs/development/validation-scope.md#python-修改工作流) |
| 具体子系统责任归属 | [主机编排与测量](../../../docs/development/orchestration.md#主机编排与测量)、[结果分析与补采](../../../docs/development/consumers.md#结果分析与补采) |

## 执行流程

1. **建立隔离工作区与基线。** 核对已有 worktree 与未提交改动，按[Git 协作规则](../../../docs/development/quality.md#git-worktree-与合并)创建专属具名分支和 worktree，并准备独立 `.venv`。在改动前运行目标测试与静态检查脚本，固定初始基线。
2. **先调研调用链与边界。** 沿实际导入与调用关系定位全部生产者、消费者、CLI 惰性加载路径、子进程入口及测试 `mock` 查找位置；不凭文本搜索直接推断耦合或机械拆分 `data-review` 静态数据目录。需要借鉴外部模式时，只取轻量化设计思路并复用现有标准库与项目基础模块，不引入额外运行依赖。
3. **先冻结可观察行为。** 拆分或搬迁前确认已有回归覆盖正常执行、字段错误顺序、异常传播、资源清理与恢复路径；覆盖不足时先补齐基于公共入口或真实边界的回归用例。`TypedDict` 与 `TypeGuard` 必须承接真实运行时验证，不用 `cast`、`Any` 或 `assert` 放行未校验输入。
4. **分批重构并同步门禁。** 按单一职责抽出纯计算、输入校验或共享协议，保持 `cli`、`application`、`host`、`models`、`analysis`、`plotting` 与 `runtime` 的单向依赖；遵循[只保留当前协议](../../../docs/development/modules.md#只保留当前协议)删除已有替代的冗余兼容路径。模块移动后同步更新测试 patch 路径、`pyproject.toml` 中的 `[tool.importlinter]` / `[[tool.mypy.overrides]]`、`.pre-commit-config.yaml` 局部规则与 `tests/private_api_baseline.json`（只减不增），确保 `tests/quality/test_architecture.py` 目标校验一致。
5. **分层验证与交付收尾。** 按改动范围执行定向 pytest、模块导入隔离检查、`ruff`（含 `C901`/`BLE001`）、`mypy`、边界脚本与文档检查；审阅完整 `git diff` 与未跟踪文件。同步受影响专题文档及 [进度表](../../../docs/roadmap.md)，按约定提交、fast-forward 合入本地 `main` 并清理本任务 worktree 与分支。

## 核对清单与交付

具体门禁命令与收紧检查项见[门禁核对清单](references/gate-checklist.md)；交付汇总使用[重构交付模板](assets/refactor-report.md)，区分已冻结契约、门禁与测试证据、文档更新及未覆盖边界。
