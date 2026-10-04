# AC-Prof 协作规则

AC-Prof 用于对容器化 Hugging Face 推理工作负载进行可复现分析，记录延迟、吞吐、能耗、资源利用率及实验元数据。

修改项目时优先保证测量语义、结果可复现性和实验数据完整性。

## 工作原则

- 使用简体中文回复；代码、CLI、Git commit message 和已有英文技术标识保持英文。
- 只读取当前任务需要的上下文。不要因为一次局部修改而预读整个 `docs/`、全部 Skills 或完整仓库。
- 进入目录前检查是否存在更具体的 `AGENTS.md`，并同时遵循其局部规则。
- 非平凡代码改动涉及设计、兼容性、依赖、性能、资源管理或测量路径时，先调研成熟 GitHub 项目的官方源码、Issue 和 PR，再结合 AC-Prof 选择方案；借鉴设计思想，不机械复制，并考虑许可证、维护成本和测量开销。
- 优先保持单一事实来源和清晰边界。已有实现可以替代旧路径时，删除无必要的 alias、shim 和重复入口。
- 当前项目仍处于开发阶段。除非涉及已有实验结果、持久化数据协议或用户明确要求，不为尚未正式发布的内部接口保留兼容层。
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

WSL2 可用于开发和 PARTIAL / basic 采集，但不得把 WSL 测量视为 Native Linux 测量，也不得为了兼容 WSL 修改 Native Linux 的测量语义。具体边界见 [WSL2](docs/platforms/wsl2.md)。

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
| Python 修改、测试范围、开发检查 | `docs/Testing.md` |
| 架构或模块边界 | `docs/Architecture.md` |
| CLI、参数、安装和运行 | `docs/CLI_Reference.md`、`docs/Getting_Started.md` |
| 测量协议、结果字段、schema、resume | `docs/Profiling_Protocol.md`、`docs/Metrics.md`、`.agents/skills/acprof-schema-change/SKILL.md` |
| 模型、backend、adapter、运行依赖 | `docs/Runtime_Compatibility.md`、`.agents/skills/acprof-model-adaptation/SKILL.md` |
| 能耗、OOM、cgroup、结果异常 | `docs/Energy_Measurement.md`、`docs/Troubleshooting.md`、`.agents/skills/acprof-result-audit/SKILL.md` |
| 模型集或 Hub 兼容性审计 | `.agents/skills/acprof-compatibility-audit/SKILL.md` |
| Docker 镜像、重建和空间问题 | `.agents/skills/acprof-docker-audit/SKILL.md` |
| GitHub Actions 失败 | `.agents/skills/acprof-ci-triage/SKILL.md` |
| profiling、benchmark、GPU profiler | `.agents/skills/acprof-profiling-workflow/SKILL.md`、`docs/Profilers.md` |
| TUI、焦点、日志和交互 | `docs/i18n/README_zh-CN.md`、`.agents/skills/acprof-textual-regression/SKILL.md` |
| 文档组织或 Agent / Skill 规则 | `docs/README.md` |

历史架构、旧默认值或项目来源不明确时，再读取 `docs/Project_Origin.md`。

## 验证与执行边界

在当前任务授权范围内，可以直接：

- 修改相关代码和文档；
- 运行与改动相称的本地、unit 和离线测试；
- 修复由本次改动引起的测试失败并重新验证；
- 运行 lint、type check、build 和必要的轻量 smoke check；
- 检查完整 Git diff、未跟踪文件和受影响测试。

不要仅因为完成了第一版实现就停止。若任务要求功能可用，应继续完成相关失败路径、验证、必要文档和最终 diff 检查，直到达到完成标准或遇到真实外部阻塞。

除非任务明确需要，不要自行：

- push、merge、rebase、force push 或改写远程历史；
- 执行 `reset --hard` 等破坏性 Git 操作；
- 启动正式大规模 profiling；
- 下载大型模型或镜像；
- 运行长时间 GPU profiler；
- 删除已有实验结果。

正式实验、硬件测量或昂贵下载不是普通代码修改的默认验证手段。

## Git

完成开发后按逻辑主题拆分 commit，避免把无关修改混入同一个提交。

Git 提交信息统一遵循 [`docs/commit-messages.md`](docs/commit-messages.md)。

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

除非存在需要用户做真实产品决策的歧义，否则不要在中间实现阶段停下来等待确认。
