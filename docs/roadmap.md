# 项目进度与下一步

本页只维护里程碑、尚未完成的工作和验收缺口；实现细节在对应专题，逐次修改通过 Git 历史和 [Release notes](https://github.com/kainam15/AC-Prof/releases) 查询。
开发优先级与执行边界以 [GOAL](GOAL.md) 和 [验证策略](development/validation-scope.md) 为准；离线回归不替代真实硬件、用户终端或同提交发布门禁。

## 里程碑

| 阶段 | 目标 | 状态与下一步 |
| --- | --- | --- |
| M0 · 能力盘点 | 区分已实现、可用和未验证的能力 | 已完成；新增能力继续明确证据范围。 |
| M1 · 采集闭环 | 从准备、采集到结果分析贯通 | 已完成；Process CPU/GPU basic 与具备条件的 Native Docker full 已建立闭环，代表性实测持续扩展。 |
| M2 · 通用适配 | 通过统一接口扩展模型与任务族 | 已完成既有六类真实 checkpoint CPU/GPU full 和 Structured CPU full 验收；新增适配仍按独立模型、设备与后端取得证据。 |
| M3 · Runtime / Profiler | 能力声明、资源边界、生命周期和失败证据一致 | 进行中；Torch/NCU/Nsys 的代表性双尺度后处理已验证，Massif 完整实测、同提交硬件工作流和恢复场景仍待验收。 |
| M4 · TUI 体验 | 配置、运行、诊断、恢复和结果查看完整可用 | 进行中；结构化共享视图与迁移文案清理已完成，结果概览、补采实验、实验配置采集计划、环境弹窗、任务选择弹窗和运行监控页头已逐页确认；A4 定向回归与 28 项 headless SVG 重新生成比较通过（23 张基准更新、5 张保留），12 项既有 VHS/xterm.js PNG 已完成重拍、逐图评审及基线更新；WSL 报告打开与复制路径已修复，Docker 下载状态、前端来源和静默计时已补齐；真实模型正式测量中断续跑、客户端终端和浏览器仍待验收。 |
| M5 · 科研结果 | 数据可解释、可复现、可比较 | 进行中；统计、审计、HTML 对比和流式 P0 已贯通，能耗稳定性、重复性及深入 LLM 资源画像待补。 |
| M6 · 交付使用 | 安装、文档、发行与故障诊断 | 已完成既有 wheel、uv tool、standalone 与安装后 smoke 验收；每次发行仍须通过对应版本的发布门禁。 |

## 工程维护优先序

| 优先级 | 工作 | 状态与完成条件 |
| --- | --- | --- |
| P0 | 共享 Workload 输入校验、明确依赖方向 | 本轮已完成六族公共字段复用与两个反向依赖修复，保留任务差异和错误语义；Import Linter 契约纳入门禁。见[模块边界](development/static-checks.md)与[Workload 输入校验](development/modules.md#workload-输入校验)。 |
| P1 | 按真实竞态改进 TUI 测试同步 | 本轮已合并三处条件等待，并以显式握手消除下载测试的一秒存活竞态；其余等待只在有具体问题时迁移。见[TUI 条件等待](development/validation-scope.md#tui-条件等待)。 |
| P1 | 结果存储与恢复边界 | A1–A5 已完成：持锁后先核对实验身份；pending 的读取、审计与恢复共用完整事务契约；manifest、初始化 receipt 与阶段化 run_state 在恢复副作用前验证。真实提交故障和合成畸形输入回归覆盖拒绝时证据不变、锁释放及合法恢复，保留 flat layout 与补采语义。真实测量中断验收仍按下表独立推进。见[实验产物结构与恢复](profiling/artifacts.md)。 |
| P1 | 核心数据契约增量强类型化 | 结果引用、层清单、pending/commit 与阶段化 run_state、attempt、case 已接入实际读写边界；运行时验证后再窄化类型，相关模块扩展增强 mypy 门禁，开放 metadata 与阶段可选字段保持真实语义。下载计划和检查点的受控生产者也已类型化并纳入增强门禁，缓存 JSON 保留真实验证边界；后续依实际调用链推进依赖预检报告和事件协议，不对全部临时 `dict` 一刀切。见[类型与边界检查](development/static-checks.md)。 |
| P1 | 宽泛异常与失败证据审计 | 网络包回填数值转换先收窄异常并增加 BLE001 门禁；后续逐模块核查 `monitors`、`host` 的失败、清理、unknown/NaN 语义，禁止把无法采集改成 0 或成功。 |
| P2 | 发布与运行输入边界 | 运行时镜像发布来源已限制为 main 手动触发或稳定标签 push；Docker 服务端口越界在准备前拒绝，保留串行端口复用和 Process 动态端口，地址占用保留清理语义并给出定位建议；plot 缺失输入已转换为单行用户错误，测量锁优先和未知异常传播保持。离线回归已覆盖；GitHub 平台保护、真实 Docker 绑定与同提交 CI 仍待完善或验收。 |
| P2 | 跨进程测量配置契约 | 所有 ClientConfig 字段已有独立编码样例和真实子进程表征；条件性环境继承、测量条件记录/重放、通知 webhook 隔离及代理保留已验证。后续在该边界下收敛重复 codec；不改变预热次数、结果身份或遗留中间产物路径覆盖。见[主机编排](development/orchestration.md)。 |
| P2 | 高复杂度职责的行为保持重构 | `plan_download` 已迁入 models 领域层并按真实布局拆分，冻结 JSON/哈希、未知大小、回退与原地校验语义，并补齐缓存/镜像源码指纹。下一步为 `RunConfig.validate` 冻结字段错误、报错顺序与归一化语义，再按校验领域抽取；`dependency_preflight`、输出/运行时验证按调用链逐个评估，不一刀切启用全仓 C901。 |
| P2 | 导入、格式和测试结构治理 | 只对可证明存在循环依赖的局部导入、审阅困难的超长行、真实竞态等待和过大测试场景定向处理；避免大规模 reformat 或机械拆分。 |
| P3 | 引导安装完整性 | setup.sh 的 uv 0.13.5 安装脚本已固定 SHA256 并在执行前校验，curl/wget、篡改、中断和缺少校验工具由离线 shell 夹具验证；真实首次安装与发行环境验收仍独立进行。见[源码安装](development/distribution.md)。 |
| P3 | Worktree 与实验产物安全收尾 | 先分别核对分支合并状态、未提交修改、并行任务、科研 evidence 与缓存可再生性；仅列候选，不自动删除历史工作树、结果或证据。 |

本轮治理采用离线定向回归与开发门禁验收；硬件、真实模型和用户终端的缺口仍按下表保留。
已主动隐藏的 TUI 区域继续遵循[维护边界](development/consumers.md#tui-与兼容维护)，共用后端保持维护。
历史结果、失败证据和恢复状态按[本地测试产物规范](development/internal-testing-layout.md)保留；清理默认先预览，不因目录体积而删除科研证据。

## 待验收与后续实验

| 范围 | 尚缺证据 | 执行与完成要求 |
| --- | --- | --- |
| 同提交 CI / Release gate | 修复提交对应的远端完整 CI、独立 checkout/虚拟环境硬件作业 | 按[CI 与发布验收](development/ci.md)执行；历史成功不能替代当前源码提交。 |
| Massif | BERT CPU 两档输入的完整报告 | 先前首尺度超时、第二尺度未执行，完成数为 0/2；用户按需组织新的有界实测。 |
| 能耗与重复性 | 基线稳定、足量独立样本、跨环境对照 | 保留负残差、告警和 unknown；小样本覆盖验收不能用于稳定性能排名。 |
| 正式测量中断与恢复 | 真实模型测量中断→清理→续跑→结果审计 | 当前离线 CLI Fixture 已覆盖；真实 CLI/Docker 准备取消与重试证据不能代替正式测量中断验收。 |
| Docker / Model Store 恢复 | 真实 WSL2 缺层修复、Engine 升级、崩溃与迟到容器创建后的恢复 | 现有故障注入和持久化所有权协议继续保持；真实 daemon 行为另行验收。 |
| Docker 下载统计 | 外网镜像层、完整运行镜像构建及用户实际终端 | 已用真实 WSL 日志验证前端静默及 HTTP ADD 无字节事件；中英文 80／120 列 PNG 为新建审阅基准。后续补发行与客户端验收，统计边界见[镜像下载统计](models/images.md#下载总量与传输统计)。 |
| 用户界面 | 实际 Windows/WSL/VS Code 终端、Windows/WSL/macOS 浏览器打开和剪贴板，以及 Edge/Chrome 的离线 HTML 操作 | 平台 Mock 和 headless 不代表用户客户端验收；最新打开行为见[打开报告](usage/tui.md#打开-html-报告)。 |
| 新模型与后端 | 新增真实 checkpoint、GPU/Process 回放、服务观测和 sweep | 按[模型适配](models/adaptation.md)和[流式测量](profiling/streaming.md)逐项核验；静态识别或 mock 不代表实际推理成功。 |

以上真实模型、GPU、Profiler、性能与重复性实验由用户明确选择后执行，不作为普通开发的默认阻塞项。

## 代表性矩阵扩展

既有本地验收覆盖六模型双尺度 Docker full、BERT/Iris 资源矩阵，以及 BERT CPU/GPU 与 Iris 的 Process basic。
历史扩展共 49 组配置、147 行，包含 49 个 warmup、98 个正式窗口；正式行 94 个 `ok`、4 个能耗 `warn`，无正式推理错误。
含告警的运行未满足正式行全 `ok` 的硬件验收条件；部分全 `ok` 运行仍有 idle baseline 诊断。
Process 的 10 个正式窗口均为 `ok`，资源为 inherited，不能等同于 Docker 数值限额。

BERT 后处理完成 Torch CPU/GPU 4/4、NCU GPU 2/2、Nsys GPU 2/2；Massif 仍为 0/2。
已成功工具的回填保留原 12 行及 201 个非 Profiler 列，不能把未完成工具标作通过。
复现边界与夹具见[代表模型与多尺度扩展验收](development/ci.md#代表模型与多尺度扩展验收)。
这些是已有实验记录，本轮工程治理没有增加真实采集证据。

## LLM 服务画像优先序

| 优先级 | 状态 | 已实现与下一步 |
| --- | --- | --- |
| P0 · 流式测量基础 | 已完成 | OpenAI Chat/Completions SSE、TTFT/TPOT/严格 ITL/E2E、分位数与 Token 来源已有确定性回归和 Qwen3-0.6B + vLLM 对照；见[流式测量](profiling/streaming.md)。 |
| P1 · 服务性能画像 | 进行中 | 并发、固定/Poisson 到达率、连接复用、SLO Goodput、单点/sweep HTML、服务身份和 Process 回放已有离线与本地 HTTP 证据；真实服务 sweep、GPU 回放及 vLLM/SGLang 观测待用户按需验收。 |
| P2 · 深入 LLM 画像 | 进行中 | `--server-metrics vllm` 已读取并展示 Queue/Prefill/Decode 等服务直方图差值，保留重置与缺失原因；阶段 GPU 资源、能耗、多后端对照及可选服务端时间戳尚未完成。见[服务端阶段统计](profiling/streaming.md#服务端阶段统计)。 |

共享服务直方图不是本次请求的精确阶段计时，共享 GPU 能耗不能直接解释为每请求的精确 Token 能耗。

## 历史与维护

整理前的完整进度记录保存在 Git 提交 `173058a01e2f164bea7901e1f0344084f7b97056`，可在本地查看：

```bash
git show 173058a01e2f164bea7901e1f0344084f7b97056:docs/roadmap.md
git log -- docs/roadmap.md
```

后续只更新受影响的里程碑、下一步和未验证项；长期事实放入对应专题，临时测试报告保存在 `internal-testing/`。
