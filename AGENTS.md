# AC-Prof 协作规则

AC-Prof 用于对来自 Hugging Face 和 ModelScope 的模型推理工作负载进行可复现的运行时画像分析，记录延迟、吞吐、能耗、资源利用率及实验元数据。

## 工作流程

1. **目标对齐** ：开发前阅读 [开发目标](docs/GOAL.md) 与 [进度表](docs/roadmap.md)，明确任务目标、优先级和验收标准。
2. 调研成熟和多stars的 GitHub 项目，借鉴设计思路
3. 实现对应代码
4. 实现请求的目标行为及相关合理失败路径。
5. 运行与改动相称的验证，并处理由本次修改造成的失败。
6. 检查完整变更集，包括新增文件和意外生成文件。
7. 更新受影响的权威文档、示例和链接（按主题分类)。
8. 维护可动态调整的 [进度表](docs/roadmap.md) ，根据新的发现更新进度表，并继续下一个高价值建设目标。
9. 根据项目当前成熟度，识别长期发展方向。



## 工作原则

- 对话使用简体中文回复；代码、CLI、Git commit message 和已有英文技术标识保持英文。
- 优先保持单一事实来源和清晰边界。已有实现可以替代旧路径时，删除无必要的 alias、shim 和重复入口。
- 文档优先放入docs/ ，而不是放到根目录README。
- 凭据仅存放在 Git 忽略的 `.env.local` 等本地配置中；不得提交密码、Token、webhook 或其他秘密。

## 测量语义

- 正式测量窗口只包含目标工作负载及协议定义的采集行为。
- TUI、绘图、通知、额外诊断、环境检查和开发辅助逻辑不得污染正式测量窗口。
- 不支持或无法取得的硬件指标必须记录为 unavailable / unknown 等明确状态；禁止填 `0`、估算值或其他指标的替代值。
- 已有实验结果的字段定义、单位、采集边界、resume 语义和 provenance 不得因内部重构而无意改变。
- 静态检查、startup probe、正式实验和 post-hoc profiling 保持语义分离。

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

默认不做全量回归，优先进行定向测试。
正式实验、硬件测量或昂贵下载不是普通代码修改的默认验证手段。

## Git

- 每个独立开发任务开始前，必须创建专属 Git Worktree 和对应的独立任务分支，两者一一对应。
- 禁止直接在 main 上开发、修改或提交代码。 main 仅用于集成已完成且通过验证的任务。
- 完成开发后按逻辑主题拆分 commit，避免把无关修改混入同一个提交。
- Git 提交信息统一遵循 `.github/commit-message.instructions.md`
- 除非用户另有要求，任务完成并通过与改动相称的验证后，默认将本次任务的提交整合到本地 `main`，优先使用 `git merge --ff-only`；确认提交已纳入 `main` 后，清理本任务的临时工作树并删除已合并的任务分支，无需再次确认。清理前保留需要的验证记录，保护其他任务的工作树和未提交改动，最后需要汇报分支清理情况。

## 文档

* 文档编写规则在 docs/AGENTS.md
* docs/roadmap.md 是项目的长期进度跟踪文档。
* 长期事实放在对应 `docs/` 权威专题中。
* 临时计划、维护记录、开发日志按规则写入 `internal-testing/`。
* 新建或修改 Skill、`AGENTS.md` 时，参考[OpenAI：重新思考 GPT-6 Astra 的技能与提示词](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)。
* 默认不自动修改 `AC-Prof/AGENTS.md` 、`docs/AGENTS.md` 、`docs/GOAL.md`
