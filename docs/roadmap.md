# 项目进度与下一步

本页只维护里程碑、尚未完成的工作和验收缺口；实现细节在对应专题，逐次修改通过 Git 历史和 [Release notes](https://github.com/kainam15/AC-Prof/releases) 查询。
开发优先级与执行边界以 [GOAL](GOAL.md) 和 [验证策略](development/validation-scope.md) 为准；离线回归不替代真实硬件、用户终端或同提交发布门禁。

## 里程碑

Docker / Process 统一画像：已建立 [逐字段覆盖矩阵](results/runtime-coverage.md) 与
[统一语义契约](results/runtime-profile.md)，区分登记、实现、运行条件及硬件证据。
共同进程树 CPU/RSS/边界计数、Process 独立 cgroup 与 GPU 遥测、准备期 CPU 拓扑已接入；
逐指标质量已贯通审计、统计、比较与离线 HTML，开销脚本支持两种 Runtime 的配对诊断。
两种 Runtime 已从同一次 HTTP 请求记录请求及解码后的响应正文大小，历史字段集按实验冻结。
新实验统一使用 Best-effort Collection，常规能耗采集沿用 5 秒冷却、20 秒空闲基线。逐项预检、可选失败、全局状态中心和指标级可比性已接入。
Process 在 Native Linux 按条件使用 RAPL/DRAM、整卡 NVML、PID PMU 与回环 PCAP。
同版本 Docker/Process 配对开销、真实硬件和 WSL2 实机验收仍待独立执行。
Process 深度 Profiler 和 CPU 时间份额能耗归因见[接入边界](profiling/collector-capabilities.md)。

| 阶段 | 目标 | 状态与下一步 |
| --- | --- | --- |
| M0 · 能力盘点 | 区分已实现、可用和未验证的能力 | 已完成；新增能力继续明确证据范围。 |
| M1 · 采集闭环 | 从准备、采集到结果分析贯通 | 已完成；Process CPU/GPU basic 与具备条件的 Native Docker full 已建立闭环，代表性实测持续扩展。 |
| M2 · 通用适配 | 通过统一接口扩展模型与任务族 | 已完成既有六类真实 checkpoint CPU/GPU full 和 Structured CPU full 验收；新增适配仍按独立模型、设备与后端取得证据。 |
| M3 · Runtime / Profiler | 能力声明、资源边界、生命周期和失败证据一致 | 进行中；Torch/NCU/Nsys 的代表性双尺度后处理已验证，Massif 完整实测、同提交硬件工作流和恢复场景仍待验收。 |
| M4 · TUI 体验 | 配置、运行、诊断、恢复和结果查看完整可用 | 进行中；已完成四阶段 TUI 交互与界面体验升级（实验配置首屏聚焦与内联阻断跳转、点击式全局状态中心、浮动快捷命令面板、显式折叠侧栏、监控暂停横幅与上下文下一步动作、实验结果与补采统一工作台、向导步骤条与 Auto 直接编辑、选择器宽屏分栏与快捷筛选、镜像一键选无标签与自动保存界面偏好）；离线定向回归、双语检查、catalog/ruff/mypy/import-linter 通过；视觉基准（SVG/PNG）更新、真实终端与真机实验仍待验收。 |
| M5 · 科研结果 | 数据可解释、可复现、可比较 | 进行中；统计、审计、HTML 对比和流式 P0 已贯通；统计缓存的精度诊断已保留至 HTML 展示与导出，浏览器重算清除过期诊断。能耗稳定性、重复性及深入 LLM 资源画像待补。 |
| M6 · 交付使用 | 安装、文档、发行与故障诊断 | 已完成既有 wheel、uv tool、standalone 与安装后 smoke 验收；中英文 README 的章节、安装指令和文档入口已对齐；`0.4.0a8` 沿用预览版发行门禁，完整 CI、硬件与依赖风险审计独立记录，不以预览发布替代正式版验收。 |

## 工程维护优先序

开发辅助能力已按项目与个人范围拆分：新增 HTML 报告回归、Hub 静态调研、服务指标与 Actions 维护入口，
通用 Python、定向审查和 UI 检索放在个人 skills；来源与边界见[开发 Skills](development/agent-skills.md)。
这些入口不改变下表的硬件、真实模型及客户端验收缺口。

| 优先级 | 工作 | 状态与完成条件 |
| --- | --- | --- |
| P0 | 共享 Workload 输入校验、明确依赖方向 | 本轮已完成六族公共字段复用与两个反向依赖修复，保留任务差异和错误语义；Import Linter 契约纳入门禁。见[模块边界](development/static-checks.md)与[Workload 输入校验](development/modules.md#workload-输入校验)。 |
| P1 | 按真实竞态改进 TUI 测试同步 | 已按操作状态等待，并以显式握手消除下载测试的一秒存活竞态；元数据概览等待自身渲染，推理配置流程覆盖参数提交、摘要确认后才继续运行；表单修正等待对应错误提示实际清除。结果／补采工作流共用切页和输入辅助，来源编辑等待本次读取结束，取消后等待新实验概览。其余等待只在有具体问题时迁移。见[TUI 条件等待](development/validation-scope.md#tui-条件等待)。 |
| P0 / P1 | TUI 全局状态统一 | 状态栏、环境清单和详情共享具体原因文案。未启用的可选 Profiler 保留检查结果，启用后按根因计入警告。详情列出对应字段和修复建议，沿用原有重新检查和启动门禁。见[环境检查](usage/tui-interaction.md#环境检查)。 |
| P0 / P1 / P2 | Docker 安装与修复引导 | 全局状态内细分 CLI 缺失／PATH、daemon、Socket 权限、Context 与 Desktop 问题；提供按系统匹配的复制命令、官方文档、风险提示及重新检查。Process 保留非阻断警告，显式 Docker 失败阻断；沿用测量锁，不执行修复命令。定向离线、双语 Pilot 与 SVG 验证见任务证据；真实安装、用户终端剪贴板与硬件采集仍未验证。见[安装与修复引导](usage/tui-interaction.md#docker-安装与修复引导)。 |
| P1 | 配置编辑后自动预检 | 已接入即时字段校验、500ms 防抖、最新配置检查与启动门禁；自动预检复用整机清单，手动检查仍可刷新外部环境。定向离线 Worker／Pilot 回归覆盖连续编辑、过期结果、失败重试和测量互斥；真实硬件与用户终端仍待验收。见[自动配置检查](usage/tui-interaction.md#自动配置检查)。 |
| P1 | 存储空间查询与反馈 | 已隔离磁盘、Docker 统计和镜像释放估算的失败，增加固定顶部诊断、刷新受阻原因与同环境历史时间标记；Fixture、Pilot 及中英文 SVG 覆盖加载、失败、重试、关闭和窄窗。真实 Docker 服务故障与用户终端仍待现场验收。见[查询与删除](models/images.md#查询与删除)。 |
| P1 | TUI 内置版本管理 | 已实现 PEP 440 双渠道筛选、Linux 标准 `uv tool` 的确认更新／降级、独立安装与验证回退、阶段日志和重启。隔离工具环境与 Pilot／SVG 验证见本地验收记录；公共 PyPI 发行及用户终端验收未执行。`pip` / `uv pip`、`pipx` 与 Standalone 目前识别并提供手动指引，自动更新仍待按各自环境事务扩展。见[版本管理与恢复](usage/version-management.md)。 |
| P1 | 结果存储与恢复边界 | A1–A5 已完成：持锁后先核对实验身份；pending 的读取、审计与恢复共用完整事务契约；manifest、初始化 receipt 与阶段化 run_state 在恢复副作用前验证。真实提交故障和合成畸形输入回归覆盖拒绝时证据不变、锁释放及合法恢复，保留 flat layout 与补采语义。历史来源恢复夹具已通过公开状态写入器补齐信封，原样保留模型决策，并独立验证缺少 run_id 时拒绝且不写入。真实测量中断验收仍按下表独立推进。见[实验产物结构与恢复](profiling/artifacts.md)。 |
| P1 | 核心数据契约增量强类型化 | 结果引用、层清单、pending/commit 与阶段化 run_state、attempt、case 已接入实际读写边界；运行时验证后再窄化类型，相关模块扩展增强 mypy 门禁，开放 metadata 与阶段可选字段保持真实语义。下载计划和检查点的受控生产者也已类型化并纳入增强门禁，缓存 JSON 保留真实验证边界；后续依实际调用链推进依赖预检报告和事件协议，不对全部临时 `dict` 一刀切。见[类型与边界检查](development/static-checks.md)。 |
| P1 | 宽泛异常与失败证据审计 | 网络包回填数值转换已收窄异常并增加 BLE001 门禁；monitor 收尾独立于指标汇总，线程有界停止后才释放 NVML，perf 回收输出管道；Process 确认退出后 lease／资源范围独立清理，保留原始故障与取消。故障注入和轻量子进程已覆盖，Native 硬件仍待验收；其余异常只按具体失败继续核查。见[主机编排](development/orchestration.md)。 |
| P2 | 发布与运行输入边界 | 运行时镜像发布来源已限制为 main 手动触发或稳定标签 push；Docker 服务端口越界在准备前拒绝，保留串行端口复用和 Process 动态端口，地址占用保留清理语义并给出定位建议；plot 缺失输入已转换为单行用户错误，测量锁优先和未知异常传播保持；report 的参数声明不再提前加载分析模型，帮助和锁忙路径的独立进程隔离回归已补齐。离线回归已覆盖；GitHub 平台保护、真实 Docker 绑定与同提交 CI 仍待完善或验收。 |
| P2 | 跨进程测量配置契约 | 所有 ClientConfig 字段已有独立编码样例和真实子进程表征；条件性环境继承、测量条件记录/重放、通知 webhook 隔离及代理保留已验证。双向 codec 已收敛为只读字段声明表，默认值保持单一来源，替换两项已被边界测试覆盖的重复往返用例；不改变预热次数、结果身份或遗留中间产物路径覆盖。见[主机编排](development/orchestration.md)。 |
| P2 | 运行阶段契约 | 公开 RunApplication.run 回归覆盖正常执行、已完成恢复、多任务、重建、准备/采集/报告/收尾失败；真实状态锁和结果提交继续参与测试。runtime 解析返回独立执行参数，原请求不再被原位改写，Docker 默认轴集中声明，阶段顺序与统一收尾保持。见[主机编排](development/orchestration.md)。 |
| P2 | 包级后处理协议 | parser、正常/resume 和历史分段已共用命令与 schema v2 校验，合并只接受本次临时目录的新产物，拒绝误用旧 `.merged`。真实 TShark 合成 pcap、公开 case 恢复、部分错误行和无副作用拒绝路径已覆盖；保持原始证据、字段与测量窗口。见[主机编排](development/orchestration.md)。 |
| P2 | 高复杂度职责的行为保持重构 | `plan_download` 已迁入 models 领域层并按真实布局拆分，冻结 JSON/哈希、未知大小、回退与原地校验语义，并补齐缓存/镜像源码指纹。`RunConfig.validate` 已按校验领域抽取同模块辅助方法，公开入口回归固定跨字段错误顺序、消息、归一化、自动预检与路径异常语义；配置模块纳入局部 C901 门禁。`dependency_preflight`、输出/运行时验证仍按调用链逐个评估，不一刀切启用全仓 C901。见[主机编排](development/orchestration.md)。 |
| P2 | 导入、格式和测试结构治理 | 只对可证明存在循环依赖的局部导入、审阅困难的超长行、真实竞态等待和过大测试场景定向处理；避免大规模 reformat 或机械拆分。 |
| P3 | 模型适配器身份单一来源（C3） | 配置捕获后的 adapter 已贯通 registry、声明、execution、所有 handler 与加载策略，并统一 model probe 的配置和签名检查；共享加载入口在模型加载前拒绝 profile 身份或远程代码权限冲突，构建 manifest 与制品核验保持。离线调用链、四入口、交错会话、缓存覆盖、失败重试及原有依赖锁/异常诊断回归已通过；真实模型、Docker/GPU 与同提交 CI 仍未验证。见[主机编排](development/orchestration.md)。 |
| P3 | 测试与 QA 后续（D1–D3） | 待推进测试私有 API 的逐依赖递减基线、复核关键路径覆盖范围，以及依赖新增风险的报告与有限阻断；本轮不改 CI、覆盖率门槛或依赖审计策略。GitHub 平台侧分支、标签、Environment 与 GHCR 保护另行核对。 |
| P3 | CI 与视觉回归契约同步 | 已对齐验证脚本回调、版本检查与预检等待、Windows 平台模拟、模型身份与运行时注册数量，并补齐全局状态和版本管理的视觉场景目录；启动错误按文件日志验收。CI #259 的控件页面、状态断言和异步等待已修正，语言切换的结构化重绘统一延后到测量结束；翻译 hook 补齐遗漏检查，统计页双语 PNG 已定向审阅复核。CI #263 已修复跨模块私有引用和比较时的缺失值扫描，界面回归已对齐来源分组、Docker 诊断和预检完成状态。PNG 工具安装设有独立超时，Linux perf 阻断场景已按来源分组重新审阅。CI #264 的 SVG 场景已改为等待预检结果回调。PNG #75 的夹具在导航就绪后填入数据，VHS 等待场景文字显示后录制。快速测试现覆盖全部 12 个 PNG 场景的等待文字，单独修改 `options.py` 中的默认值也会触发 main 推送和 PR 的 PNG 工作流。按终端字符宽度复拍后，错误弹窗基线覆盖必需 DRAM 能耗权限阻断，宽屏表单反映自动采集布局，Process 配置的 Docker 修复引导显示“未请求”。CI #265 修复了验证程序遗漏预检回调参数导致的挂起，以及指标计算导入监控器的问题。回归用例覆盖自动采集计划、冻结配置恢复、独立 Profiler 选择和编辑期间的预检回调。同提交远端 CI 与 PNG 工作流仍待验收。 |
| P3 | 引导安装完整性 | setup.sh 的 uv 0.13.5 安装脚本已固定 SHA256 并在执行前校验，curl/wget、篡改、中断和缺少校验工具由离线 shell 夹具验证；真实首次安装与发行环境验收仍独立进行。见[源码安装](development/distribution.md)。 |
| P3 | Worktree 与实验产物安全收尾 | 先分别核对分支合并状态、未提交修改、并行任务、科研 evidence 与缓存可再生性；仅列候选，不自动删除历史工作树、结果或证据。 |

本轮治理采用离线定向回归与开发门禁验收；硬件、真实模型和用户终端的缺口仍按下表保留。
已主动隐藏的 TUI 区域继续遵循[维护边界](development/consumers.md#tui-与兼容维护)，共用后端保持维护。
历史结果、失败证据和恢复状态按[本地测试产物规范](development/internal-testing-layout.md)保留；清理默认先预览，不因目录体积而删除科研证据。

## 模型识别与动态推理配置

本轮以 `main @ 3e4bde93` 为起点，先收紧静态识别的判定与证据，再共同定义配置协议。能力和限制继续复用 Extension Manifest、RuntimeProfile、BaseHandler 与现有任务配置。

| 范围 | 当前实现与剩余验收 |
| --- | --- |
| P0 · 任务选择与失败分类 | `inspect --task` 选择接口，`--tasks` 选择推理组；逐组检查并保存结果。缺少运行时、任务未选定、缺少配置、证据冲突与不兼容制品分别报告原因和下一步。 |
| P0 · 制品与元数据 | 依据实际文件清单、被选中的索引及组件声明校验加载布局；无证据保留 unknown。Diffusers 任务提示、组件和根目录布局能力已共用声明，已知但未审查布局单独说明，不扩大真实兼容范围。非有限数值、非法 JSON 与过深嵌套转为带文件、字段路径和原始 SHA256 的诊断。 |
| P0 · 证据级别 | 静态 candidate、兼容性、接口验证和正式测量分别展示；未知架构或制品不计为静态支持。失败后修正配置可以重新预检，上游非法元数据证据继续阻断。 |
| P0-00 / P2-05 · 固定回归 | 已有 [17 场景离线种子集](usage/cli-discovery.md#离线元数据回归)，复用真实识别／coverage 链路，校验固定输入与预期，按原因排序并比较旧报告。种子场景均为合成，T5 复用固定 revision 的作者原始文件。用户提供的 Top 100 汇总为 41 / 25 / 25 / 4 / 4 / 1；原始 100 项附件及逐项 revision 尚未取得，不能宣称完成 Top 100 整集冻结或真实模型通过率比较。 |
| P1-01～05 · 声明与静态发现基础 | [统一参数协议](models/inference-configuration.md)已接入 ExtensionCatalog、inspect 和配置解析；类型、nullable、依赖冲突、消费者和来源绑定统一声明。除 NLP 外，新增 Whisper 语言、Chronos 预测长度、多模态生成控制和 SDXL 参数／Scheduler，限定模型／Pipeline 范围。默认候选保留文件／指针／revision／摘要，模型属性只读，unknown 不冒充可用。未声明路线仍需逐项适配。 |
| P1-06～15 · 表单与严格执行 | 类型化 TUI、逐字段 Auto、CLI／TUI 最终审阅与 SHA 冻结已贯通。新增作者任务预设和 AC-Prof Benchmark v1，保留来源与版本；分组后的多模态生成长度审阅已修复，custom／auto／benchmark 冻结与传递由离线回归覆盖；NLP 加载观察及四类新增任务的实际调用边界均有定向离线回归，不一致／无消费记录时阻止采集。作者预设要求固定原始来源且无冲突；新增路线的真实 checkpoint 验收未执行。 |
| P1-16～21 · 元数据与实验身份 | v8 `static_meta.json.inference_configuration` 保存冻结值、请求／Auto／预设决策、固定模型与依赖／代码身份；新任务参数及预设采用规则 v2，旧 v1 原样可读且不混入新控制。输入计划、恢复及比较绑定同一 SHA。Torch eager Profiler 使用独立派生身份，观察不改写冻结配置。跨设备真实执行和完整可复现性验收仍待用户选择实验。 |
| P2 / P3 · 覆盖与扩展 | 继续按标准模型、任务专属能力、独立官方库的顺序推进；未知官方代码只作为候选，不直接执行或声称兼容。 |

本轮验证限于离线 Fixture、Mock 与定向检查；没有新增真实模型推理、GPU、完整 Profiler 或用户终端验收证据。Top 100 的冻结集须保留原始 metadata、模型 revision、解析规则版本、原始判定和新的原因，不以放宽检查提高通过率。

## Kokoro 原生适配

固定 `hexgrad/Kokoro-82M` revision 的精确路由、专用 Handler、CPU FP32 依赖锁、Model Store
文件计划和音素尺度已接入现有扩展契约；语言／voice／speed 使用既有冻结与消费验证。
Docker 启动失败保留完整日志，TUI 展示分类原因与建议。静态检查和 Fixture/Mock 回归覆盖路由、
分段完成、音频协议、输入规划及错误路径，现有 Transformers TTS 保持独立。
真实 checkpoint CPU/GPU 推理、Docker 构建、音质、正式采集和 Profiler 均未验证；
第一阶段不开放 CUDA／其它 dtype／Profiler，独立实测由用户明确授权后开展。
完整边界见[Kokoro 原生适配](models/adaptation.md#kokoro-原生适配)。

## TUI 优化分支整合

旧 `feat/tui-optimization` 的事件路由、繁忙控件缓存、结果分析与只读锁职责拆分，
以及选择器排序／列宽、快捷键说明、日志匹配索引和结构化弹窗，已按当前架构选择性整合。
全局状态与诊断仍由现行入口管理；运行方式自动预检、推理参数确认、版本渠道／安装／重启保留。
旧分支的第二套配置摘要废弃，继续使用 `RunPlanPanel`；存储分项失败、历史读数与 unknown 语义保持。
模块与验证边界见 [TUI 架构](development/tui-architecture.md)；本轮不新增真实模型、Profiler、
客户端终端或远端 CI 验收，视觉基准由最终界面重新生成并单独比较。

## Process 资源权限与修复引导

运行方式与画像模式已迁入首页基本配置，画像模式位于运行方式之后，复用原有配置字段、预检失效和命令／计划刷新。
运行方式仅保留 Docker / Process 下拉菜单，后台默认 Auto；显示自动选择结果不会覆盖请求，手动确认后固定。
保存与重启保留选择来源，元数据继续区分 requested 与 selected；Runtime 状态统一进入右上角状态中心。
成功 Runtime 同步显示于采集计划，固定 CPU 集合保留在高级参数；权限问题从状态中心主动进入专用诊断。
运行预设下拉框及其标签同步已移除，表单直接编辑实验参数；已保存配置、CLI 预设和 `/smoke`、`/main` 快捷命令继续沿用。
CPU、内存、GPU 和输入规模归入默认折叠的“资源矩阵”，沿用现有参数与错误字段定位。
Warmup、Repeat 等九项数值设置归入基本配置中默认折叠的“测量参数”，沿用原有单位、校验和保存逻辑。
“深度性能分析”及四个 Profiler 选项也迁入基本配置，默认折叠，保留说明和勾选／保存语义。
选择、页面往返和保存后重启由离线 Pilot 覆盖，中英文及窄终端布局通过 headless 快照检查；
用户实际终端仍待验收，未改变 Runtime 策略或采集语义。见[TUI 操作](usage/tui.md)。

已复用 Process 资源管理、PreflightController、ConfigField 和 Carbon Callout，实现 cgroup
版本/控制器/委派/父级配额诊断、已有权限内的修复、配额回读与短生命周期迁移验证。
TUI 提供详细原因、手动管理员指引、重新检测及确认后的继承限制；错误不自动切换页面或修改配额。
状态中心与采集按钮共用 `GlobalStatus`，配置错误参与根因计数，未验证或过期结果不能通过。
整机能力即使未启用采集也保留实际检查结果；Process 运行环境、CPU／内存读取和各 Profiler
保留可用性与具体原因，当前实验未请求的采集工具缺失归入“未请求”。
perf / tcpdump 定位具体权限选项并在操作后复检；Docker / RAPL 使用对应诊断，页面返回恢复原操作上下文。
未改变测量字段或结果格式，也未增加测量窗口内的轮询与探测。
定向 Fixture 与 Pilot 覆盖中英文、三种终端尺寸和恢复交互；真实模型、管理员授权后的 cgroup
强制配额、SSH 客户端与 systemd 用户服务的现场验收仍未执行。见[Linux Runtime](platforms/linux-runtimes.md#资源与指标口径)。

## 待验收与后续实验

| 范围 | 尚缺证据 | 执行与完成要求 |
| --- | --- | --- |
| 同提交 CI / Release gate | 推理配置类型检查及旧回归 Fixture 已本地修复；CI #251 的结果路径迟到消息已补齐退出保护与确定性回归。CI #253 的下载来源 Fixture 已随后续适配补全，监控清理与环境警告旧断言已对齐现行契约，表单错误清除改为状态等待。修复提交对应的远端完整 CI、独立 checkout/虚拟环境硬件作业仍待验证 | 按[CI 与发布验收](development/ci.md)执行；历史成功不能替代当前源码提交。 |
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
