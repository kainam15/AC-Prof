# 项目进度与下一步

本页只跟踪**阶段进度、下一步和未完成的验收**。实现细节以各专题文档为准，逐次改动见 [Git 历史](https://github.com/kainam15/AC-Prof/commits/main/docs/roadmap.md) 与 [Release notes](https://github.com/kainam15/AC-Prof/releases)。优先级及执行边界遵循 [GOAL](GOAL.md) 和 [验证策略](development/validation-scope.md)。

**证据原则：** 离线 Fixture / Mock、真实模型推理、正式硬件采集、用户终端验收和同提交远端 CI 是不同证据层级；不得以其中一种冒充另一种。新实验采用 Best-effort Collection，缺失、质量不足或口径不一致的指标按实际原因标记，不补造数值。

## 里程碑

| 阶段 | 状态 | 当前成果与边界 |
| --- | --- | --- |
| M0 · 能力盘点 | 已完成 | [逐字段覆盖矩阵](results/runtime-coverage.md)区分声明、实现、运行条件与硬件证据；新增能力沿用相同口径。 |
| M1 · 采集闭环 | 已完成既有范围 | Docker / Process 从预检、准备、测量到审计已贯通；既有 Process CPU/GPU basic、符合条件的 Native Docker full 有实测。新实验统一 Best-effort，额外硬件覆盖仍需独立验收。 |
| M2 · 通用适配 | 已完成既有范围 | 六类真实 checkpoint 的 CPU/GPU full、Structured CPU full 已验收；其他模型、任务和后端须分别提供兼容与运行证据。 |
| M3 · Runtime / Profiler | 进行中 | [统一画像语义](results/runtime-profile.md)、Process 条件性 cgroup / RAPL / DRAM / NVML / PMU / PCAP 和生命周期诊断已接入。同一委派单元内选择资源组及复核继承配额已有模拟 cgroup 验证。Torch、NCU、Nsys 的代表性后处理已验证。NCU 和 Massif 的失败报告隔离与断点恢复已离线验证；NCU 补采已覆盖原始计数核对及物理 GPU 不匹配时的结果保护。Massif 真实双尺度采集、配对开销及资源限制仍待验收。 |
| M4 · TUI 体验 | 进行中 | 四阶段界面、状态中心、自动预检（Docker 和 Process 的 GPU 选择均已与各自的 CLI 对照）、配置折叠、结果工作台（含报告导入取消）、运行环境修复引导及版本管理已通过定向离线／视觉检查。Linux `uv tool` 的 wheel 筛选、实际升级与版本不匹配回滚已有本地索引验证。用户终端和线上更新仍待验收。 |
| M5 · 科研结果 | 进行中 | 窗口身份和冻结输入、结果恢复、逐指标质量审计、统计、精确坐标比较、离线 HTML 与流式 P0 已贯通。安装包绘图已在两个受支持的 Pandas 版本验证历史空标记和文本运行身份。PNG 窗口均值、尾延迟、延迟／能量 Pareto 图及延迟拟合已按字段质量和采样条件筛选实测窗口。能耗稳定性、重复性、真实服务画像仍需补证。 |
| M6 · 交付使用 | 既有交付完成 | wheel、uv tool、standalone 及安装后 smoke 已验收，双语入口已对齐；预览版发布不等同于当前提交的完整 CI、硬件或正式版验收。 |

## 下一步（按优先级）

| 优先级 | 事项 | 完成条件／仍缺证据 |
| --- | --- | --- |
| P0 | **同提交 CI 与发布门禁** | 服务暂存源码可独立导入。本地 Python 3.12 和 3.13 已逐项复验[服务依赖与测试夹具回归](https://github.com/kainam15/AC-Prof/actions/runs/44981173369)，并覆盖[绘图缓存、负载重放与 TUI 回归](https://github.com/kainam15/AC-Prof/actions/runs/44989161951)。[Codex 编辑反馈](development/quality.md#codex-编辑反馈)按文件触发静态检查、CI 测试清单校验，以及 TUI 夹具、服务暂存导入、权限探测和 Decimal 上下文等离线回归。仍需核实**当前源码提交**的远端完整 CI、独立 checkout / 虚拟环境硬件作业及发行门禁。见 [CI 与发布验收](development/ci.md)。 |
| P1 | **Docker / Process 实机配对** | 在相同模型、CPU / 内存精确配额和输入尺度下核对两种 Runtime 的窗口数据、质量与开销；验证 Native Linux cgroup 委派和强制配额、RAPL / DRAM、PMU 事件运行比例及 I/O `io.stat`，再补 WSL2 场景。Process 的 inherited 资源不能当作 Docker 数值限额。见 [运行时覆盖](results/runtime-coverage.md) 与 [Linux Runtime](platforms/linux-runtimes.md)。 |
| P1 | **Profiler、能耗与重复性** | 完成 BERT CPU 两档 Massif（目前 **0/2**），验证正式窗口中断／续跑后的补采；以足量独立样本检查空闲基线、负残差、能耗与性能的稳定性。不得以少量 `ok` 行直接排名。见 [Profiler](profiling/profilers.md) 与 [测量协议](profiling/measurement.md)。 |
| P1 | **模型识别与真实适配** | 冻结 Hugging Face / ModelScope Top 100 原始名单、revision 和判定证据；按模型／设备／后端补齐真实 checkpoint、GPU 与 Process 验收。Kokoro 原生路线目前只有静态、Fixture / Mock 证据，不能宣称真实音频或 Docker 通过。见 [识别与覆盖](usage/cli-discovery.md) 和 [模型适配](models/adaptation.md)。 |
| P1 | **TUI 与安装的用户端验收** | 在 Windows / WSL / VS Code SSH 终端验证状态中心、Docker / Process 修复指引、配置自动预检、探测和结果操作；在浏览器检查离线 HTML、剪贴板及打开行为。版本管理自动更新目前聚焦 Linux `uv tool`，其他安装方式仍以手动指引为主。见 [TUI 操作](usage/tui.md)、[交互说明](usage/tui-interaction.md) 与 [版本管理](usage/version-management.md)。 |
| P1 | **LLM 服务实测** | 对真实 vLLM / SGLang 服务执行并发、到达率、SLO Goodput、sweep 和 GPU / Process 回放；进一步核验服务端阶段指标、逐请求时间戳与能耗归因。当前的 HTTP / 离线回归不是完整双后端实测。见 [流式与服务指标](profiling/streaming.md)。 |
| P2 | **工程治理的增量收尾** | 按实际调用链继续收敛依赖预检／事件协议类型、宽泛异常与高复杂度职责；逐问题补充并发、恢复和包级后处理测试，不做全仓机械重构。见 [模块边界](development/modules.md)、[静态检查](development/static-checks.md) 与 [编排](development/orchestration.md)。 |
| P2 | **发行与场景补证** | 补真实 Docker 镜像下载／恢复、无 Torch 的 ONNX 容器、全量 PNG 场景及不同安装环境验收；已有本地 Fixture、Pilot、SVG / VHS 仅证明相应离线行为。见 [镜像与下载](models/images.md) 和 [CI](development/ci.md)。 |

真实模型、GPU、Profiler、长时间采集、科研重复性及管理员权限实验均由用户明确选择后执行，不作为普通代码整理的自动步骤。

## 关键现状与证据边界

### 代表性矩阵扩展

已有六模型双尺度 Docker full、BERT / Iris 资源矩阵及 BERT CPU/GPU、Iris Process basic 的历史证据。扩展矩阵共 **49 组配置、147 行**（49 行 warmup、98 个正式窗口）；正式窗口 **94 `ok` / 4 能耗 `warn`**，无正式推理错误。Process 的 10 个正式窗口均为 `ok`，但配额是 inherited；部分全 `ok` 运行仍有 idle baseline 诊断，因此不能视作完整硬件验收。

BERT 后处理：Torch CPU/GPU **4/4**、NCU **2/2**、Nsys **2/2**，Massif **0/2**；已成功回填的工具保留历史原始证据，未完成工具不记为通过。详见 [代表模型与多尺度扩展验收](development/ci.md#代表模型与多尺度扩展验收)。

### 模型识别与冻结推理配置

ModelScope 自动识别依据固定版本的配置与模型声明选择任务和加载库。历史缓存中仓库任务或库标签变化时，BERT 与 Diffusers 路线已有离线对照。静态发现区分固定版本元数据读取失败与文件内容损坏；离线缓存缺少仓库索引时仍保留配置内容及来源，能区分配置结构错误、待选择任务与运行时未支持。静态报告保留固定配置中的权重精度及顶层或解码器量化声明。确认只有 GGUF 或 Core ML 且缺少兼容加载器时，检查会直接给出制品证据。类型化参数协议、CLI / TUI Auto 与手动审阅、预设选择、输入计划 SHA 冻结及恢复身份检查已接入；`static_meta.json.inference_configuration` 保留配置与来源证据。句向量、Whisper、Chronos、VitPose / SuperPoint 单图关键点检测、多模态和 Diffusers 等有定向离线覆盖，**不代表所有新 checkpoint 均可实测运行**。历史实验复用保留模型来源、已解析的仓库 ID 与固定 commit，严格续跑仍核对冻结请求；coverage 的参数预算保留有效但未命名 dtype 的数量。新一轮 GPU 预检失败后可修正配置并重试。Process 负载重放按保存的物理 GPU UUID 复核原设备，已覆盖默认和序号选择。中断续跑仍先核验未提交模型行的原清理状态。上述行为已有离线回归。Top 100 目前只有用户提供的 **41 / 25 / 25 / 4 / 4 / 1** 汇总，缺少可逐项复核的原始名单及固定 revision，不能计算真实通过率。[参数协议](models/inference-configuration.md) · [静态元数据](profiling/metadata.md) · [Kokoro 边界](models/adaptation.md#kokoro-原生适配)。

### 结果质量、恢复与服务口径

Docker / Process 的报告、审计、比较和离线 HTML 统一按**来源身份、精确窗口坐标、采样条件和指标质量**判断可比性；冻结计划及提交／恢复前核对原始证据，失效或缺失保留 `unknown` 或诊断，不静默改写。严格审计与 Profiler 补采已对照冻结输入计划的原始字节。独立实验比较按坐标核对指标证据，并识别指向同一份 CSV 的硬链接。同一 runtime 中，HTML 与 CLI 已对齐缺包窗口的描述性差值，质量不足的有限测量仍保留拒绝比较的原因。HTML 配置按 CPU / GPU 归属质量检查，历史 CSV 审计按表头统计未记录的指标。已有事务、取消、崩溃及兼容性恢复的离线回归；真实测量中断、Docker daemon / Model Store 恢复仍待现场证据。[结果字段](results/fields.md) · [实验产物与恢复](profiling/artifacts.md) · [实验比较](results/comparisons.md)。

OpenAI Chat / Completions SSE 的成功率和 SLO Goodput 已覆盖服务端中止后仍发送 `[DONE]` 的情况。TTFT、TPOT、ITL、E2E 包含 Qwen3-0.6B + vLLM 对照；推理文本的时延和部分 Token 计数、并发负载、sweep 逐点请求错误统计与部分请求已观测的输出 Token、vLLM / SGLang exporter 与阶段直方图有离线或本地 HTTP 验证。**共享直方图不是逐请求精确计时，共享 GPU 能耗不能直接解释为每请求的精确 Token 能耗。**[服务画像](profiling/streaming.md)。

## 文档维护

本页不再累积逐次 CI 修复号、UI 控件细节、测试断言和文件级重构日志：长期规则写入对应的 `docs/` 专题，临时测试与验收记录写入 `internal-testing/`。历史详情通过 `git log -- docs/roadmap.md` 查询；更早的完整进度记录可从提交 `173058a01e2f164bea7901e1f0344084f7b97056` 查看。新增进展只更新相关里程碑、未完成事项和证据边界。
