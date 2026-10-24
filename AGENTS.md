# AC-Prof 协作规则

AC-Prof 用于对容器化 Hugging Face 推理工作负载进行可复现分析，记录延迟、吞吐、能耗、资源利用率及实验元数据。

修改项目时优先保证测量语义、结果可复现性和实验数据完整性。

## 工作原则

- 使用简体中文回复；代码、CLI、Git commit message 和已有英文技术标识保持英文。
- 只读取当前任务需要的上下文。不要因为一次局部修改而预读整个 `docs/`、全部 Skills 或完整仓库。
- 进入目录前检查是否存在更具体的 `AGENTS.md`，并同时遵循其局部规则。
- 非平凡代码改动涉及设计、兼容性、依赖、性能、资源管理或测量路径时，先调研成熟 GitHub 项目的官方源码、Issue 和 PR，再结合 AC-Prof 选择方案；借鉴设计思想，不机械复制，并考虑许可证、维护成本和测量开销。
- 优先保持单一事实来源和清晰边界。已有实现可以替代旧路径时，删除无必要的 alias、shim 和重复入口。
- 当前项目处于开发阶段，尚无用户。为降低未来重构成本，在已授权任务范围内，优先保证长期架构合理、模块边界清晰和维护成本可控；必要时可进行大规模重构，调整内部接口和模块结构，不以最小改动为默认目标。
- 重构应同步更新受影响的调用方、测试和文档，完成与改动相称的验证，并保护测量语义、结果可复现性和已有实验数据完整性。
- 除非涉及已有实验结果、持久化数据协议或用户明确要求，不为尚未正式发布的内部接口保留兼容层。
- 不得通过扩大 ignore、skip、fallback 或静默异常处理来掩盖真实问题。
- 凭据仅存放在 Git 忽略的 `.env.local` 等本地配置中；不得提交密码、Token、webhook 或其他秘密。

## 测量语义

- 正式测量窗口只包含目标工作负载及协议定义的采集行为。
- TUI、绘图、通知、额外诊断、环境检查和开发辅助逻辑不得污染正式测量窗口。
- 不支持或无法取得的硬件指标必须记录为 unavailable / unknown 等明确状态；禁止填 `0`、估算值或其他指标的替代值。
- 已有实验结果的字段定义、单位、采集边界、resume 语义和 provenance 不得因内部重构而无意改变。
- 静态检查、startup probe、正式实验和 post-hoc profiling 保持语义分离。

## 平台边界

Native Linux 是 FULL 测量的基准环境。

WSL2 可用于开发和 PARTIAL / basic 采集，但不得把 WSL 测量视为 Native Linux 测量，也不得为了兼容 WSL 修改 Native Linux 的测量语义。具体边界见 `docs/platforms/wsl2.md`

环境判断集中在 `acprof/platform.py`。

修改以下内容后，需要明确说明真实 Native Linux 验证状态：

- RAPL / energy
- PMU / perf
- cgroup
- NVML / GPU
- CPU topology / affinity
- cold start
- measurement window

交付时报告以下之一：

- `Native validation: verified`
- `Native validation: required`
- `Native validation: not applicable`

WSL、mock 和离线测试不能替代需要真实硬件证据的 Native validation。

## 按任务读取

只在对应任务出现时读取这些资料；长文档优先定位相关章节，不批量加载。

| 任务 | 资料 |
| --- | --- |
| Python 修改、测试范围、开发检查 | `docs/development/testing.md` |
| 架构或模块边界 | `docs/development/architecture.md` |
| CLI、参数、安装和运行 | `docs/usage/cli.md`、`docs/getting_started/installation.md` |
| 测量协议、结果字段、schema、resume | `docs/profiling/protocol.md`、`docs/results/metrics.md`、`.agents/skills/acprof-schema-change/SKILL.md` |
| 模型、backend、adapter、运行依赖 | `docs/models/runtime.md`、`.agents/skills/acprof-model-adaptation/SKILL.md` |
| 能耗、OOM、cgroup、结果异常 | `docs/profiling/energy.md`、`docs/usage/troubleshooting.md`、`.agents/skills/acprof-result-audit/SKILL.md` |
| 模型集或 Hub 兼容性审计 | `.agents/skills/acprof-compatibility-audit/SKILL.md` |
| Docker 镜像、重建和空间问题 | `.agents/skills/acprof-docker-audit/SKILL.md` |
| GitHub Actions 失败 | `.agents/skills/acprof-ci-triage/SKILL.md` |
| profiling、benchmark、GPU profiler | `.agents/skills/acprof-profiling-workflow/SKILL.md`、`docs/profiling/profilers.md` |
| TUI、焦点、日志和交互 | `docs/i18n/README_zh-CN.md`、`.agents/skills/acprof-textual-regression/SKILL.md` |
| 文档组织或 Agent / Skill 规则 | `docs/README.md` |

历史架构、旧默认值或项目来源不明确时，再读取 `docs/development/history.md`。

## Python 环境

- 主机发行依赖与开发依赖组统一声明在 `pyproject.toml`，项目环境只更新 `uv.lock`；使用 `uv sync` 准备环境、`uv run` 执行开发命令，CI 和固定版本验证使用 `--locked`。
- 每个 Git worktree 使用自己的 `.venv`；uv 安装在项目环境之外，不要求手动激活虚拟环境。依赖组与更新方式见 `docs/development/distribution.md#本地开发环境`。
- Docker 的目标平台 wheel、哈希与系统锁继续由 `dockerfiles/` 独立管理；项目 `uv.lock` 不替代推理环境身份。

## 验证与执行边界

在当前任务授权范围内，可以直接：

- 修改相关代码和文档；
- 运行与改动相称的本地、unit 和离线测试；
- 修复由本次改动引起的测试失败并重新验证；
- 运行 lint、type check、build 和必要的轻量 smoke check；
- 检查完整 Git diff、未跟踪文件和受影响测试。

正式实验、硬件测量或昂贵下载不是普通代码修改的默认验证手段。

## Git

完成开发后按逻辑主题拆分 commit，避免把无关修改混入同一个提交。

除非用户另有要求，任务完成并通过与改动相称的验证后，默认将本次任务的提交整合到本地 `main`，优先使用 `git merge --ff-only`；确认提交已纳入 `main` 后，清理本任务的临时工作树并删除已合并的任务分支，无需再次确认。清理前保留需要的验证记录，保护其他任务的工作树和未提交改动。

Git 提交信息统一遵循 `.github/commit-message.instructions.md`

## 文档

`README.md` 与 `docs/i18n/README_zh-CN.md` 中对应的用户可见内容必须保持同步，包括命令、链接、功能说明和排版。

长期事实放在对应 `docs/` 权威专题中。

Skill 维护特定工作流，不复制长期项目知识；一个 Skill 涉及多个流程时，根 `SKILL.md` 应保持为轻量路由，仅按需要指向详细资料和脚本。

临时计划、当前机器磁盘余量、Git 分支、运行中进程等易过期信息不得写入长期 Agent 文档。

## 完成标准

任务完成前：

1. 实现请求的目标行为及相关合理失败路径。
2. 运行与改动相称的验证，并处理由本次修改造成的失败。
3. 保持测量语义、结果协议和需要长期稳定的数据行为。
4. 更新受影响的权威文档、示例和链接。
5. 检查完整变更集，包括新增文件和意外生成文件。
6. 汇报已验证内容，以及未覆盖的 Docker、GPU、Native Linux 或终端范围。
