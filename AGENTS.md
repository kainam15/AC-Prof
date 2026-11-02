# AC-Prof 协作规则

AC-Prof 是 Hugging Face / ModelScope 模型推理工作负载的运行时画像工具。开发前按需阅读[项目目标](docs/GOAL.md)和[进度表](docs/roadmap.md)。

## 工作原则

- 当前开发阶段优先功能正确、广泛兼容、采集可用、架构简洁和 TUI 可用性；遵循当前任务范围，不自行继续新的建设或实验任务。
- 对话使用简体中文；代码、CLI、Git commit message 及既有英文标识保持英文。
- 调研成熟的高 Star GitHub 实现，借鉴思路，不机械复制；保持单一事实来源、清晰模块边界，避免重复实现与过度抽象。
- 不得提交凭据或通过忽略、伪造、静默降级来掩盖真实错误。

## 开发流程

1. 核对当前代码、目标和影响范围；为独立任务创建专属 Git worktree 与具名分支。
2. 实现请求行为及相关失败路径，维护受影响的代码、测试和权威文档。
3. 按[验证策略](docs/development/validation-scope.md)做与改动相称的定向检查，审阅完整 diff、未跟踪文件和测试结果。
4. 按[Git 协作与提交规则](docs/development/quality.md#git-worktree-与合并)拆分提交；验证后默认 fast-forward 合入本地 `main`，再清理本任务工作树与分支，保护其他并行任务。
5. 汇报实现、验证范围、未验证项与合并清理情况；更新相关[进度表](docs/roadmap.md)，额外工作仅提出建议，不自动开展。

## 验证与执行边界

- 默认使用静态检查、Mock、Fixture、定向 pytest 和必要的轻量 smoke；控制耗时，具体预算与升级条件见[开发验证策略](docs/development/validation-scope.md)。
- 真实硬件采集、科研级精度及重复性实验由用户主导。未经用户**明确要求执行具体实验**，不得主动启动真实模型采集、完整 Profiler、长时间 Docker 构建、大规模下载或全量回归。
- 能耗基线噪声、负残差及可选 Profiler 缺口应如实报告，不得为消除告警而反复实测；开发验证通过不得冒充真实硬件验收，发布门禁另行遵守[CI 规则](docs/development/ci.md)。

## 测量与数据底线

- 遵守[测量协议不变量](docs/profiling/measurement.md#协议不变量)和[结果字段语义](docs/results/fields.md)；不得污染正式测量窗口、捏造不可用指标或无意改变已有结果、恢复和来源语义。
- 静态检查、startup probe、正式实验和 post-hoc profiling 保持语义分离。

## Git、环境与文档

- 禁止直接在 `main` 开发或提交，禁止以 Detached HEAD 作为任务分支；合并前保护已有未提交修改，遵守一任务一 worktree/分支。
- Python 项目环境用 uv 管理；主机依赖、每工作树 `.venv`、`uv.lock` 与 Docker 锁的细节以[开发环境](docs/development/distribution.md#本地开发环境)为准。
- 长期技术规则放入 `docs/` 的对应专题，临时证据放入 `internal-testing/`；遵守[文档维护约束](docs/AGENTS.md)。
- 默认不自动修改根目录 `AGENTS.md`、`docs/AGENTS.md` 或 `docs/GOAL.md`；后者另有禁止 Agent 自动修改的约束。
