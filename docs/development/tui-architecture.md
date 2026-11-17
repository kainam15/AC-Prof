# TUI 模块边界与运行会话所有权

AC-Prof 的 TUI 使用 **Textual 应用外壳 + 页面组合 + 执行协调器 + 业务功能模块**。目录边界反映实际职责，不把所有业务对象强制收归 `AcprofTui`。

## 实现结构

```text
acprof/tui/
├── app.py                  # Textual 应用外壳、导航、互斥和 worker 宿主
├── event_routes.py         # 当前 App 的声明式事件转发，无独立状态或 Worker
├── busy_state.py           # 默认页面控件引用缓存及 OperationState 投影
├── commands.py             # 无 Textual 的命令/操作策略
├── i18n.py                 # 翻译目录，导入时不加载 Textual
├── catalog/                # 翻译文案的静态词典
├── experiments/            # 实验历史与选择器
├── images/                 # 镜像列表、操作与存储
│   ├── presentation.py     # 树、表格、工作区及局部行刷新
│   ├── detail_panel.py     # 单份镜像摘要、详情及高度调整
│   └── formatting.py       # 镜像文案纯函数
├── logging/                # 日志保留、显示、导出
├── models/                 # 模型候选和 Store 确认
├── results/                # 历史结果、统计、报告与 profiler
│   ├── controller.py       # 实验选择、独立元数据/数据 Worker 与操作控件副作用
│   ├── analysis_controller.py # JSON 统计、比较及导出状态的唯一拥有者
│   ├── html_controller.py  # HTML 生成回执、完成状态及打开/导出操作
│   ├── offline_reads.py    # 重叠只读任务共用的测量排斥租约
│   └── report_actions.py   # 不依赖 Textual 的按钮可用性/显示纯计算
├── run/                    # 执行、预检、恢复、进度与命令呈现
│   ├── execution.py        # 唯一执行会话/身份和进程所有者
│   ├── process.py          # 进程/进程组启动、停止和回收
│   ├── process_controller.py # 工作线程读取、事件、延迟回收
│   ├── presenter.py        # 当前运行信息的 Textual 呈现，无执行所有权
│   ├── monitor_view.py     # 进度快照的纯结构化页头投影
│   ├── monitor_panel.py    # 共享组件组合与无计时器的 Case 进度
│   └── results.py          # 只读归属检查与纯完成判定
├── settings/               # 表单配置、语言和本地首选项
└── ui/
    ├── dialogs.py          # 对话框及操作栏方向键导航
    ├── shortcuts.py        # F1 / ? 快捷键说明，无宿主 I/O
    ├── pages/              # Run / Monitor / Results(在 results/) / Backfill / Images / Settings
    └── ...                 # 公共控件、布局及主题
```

## 所有权与依赖

- `ExecutionCoordinator` 是当前执行尝试的唯一拥有者。它持有 `ExecutionSession`（执行种类、命令、Token、Snapshot、停止标记与归属结果）和唯一的 `ProcessLifecycle`。新尝试必须取得新 Token，旧结果和取消回调不能进入新会话。
- `ProcessLifecycle` 只能在目标进程确认退出后释放。清理超时保留进程占用状态，禁止新实验开始；停止必须携带期望的进程对象，防止迟到的 SIGINT 误伤替换进程。
- `run/process_controller.py` 使用唯一 App Worker Manager 调度日志读取和失败后的回收。Textual 窗口卸载不等于采集进程结束；UI 回调必须在 App 存活且 Token 匹配时投递。
- `RunPresenter` 仅修改现有控件、焦点、通知和计时器。它通过 App 的明确接口读取 `ExecutionCoordinator`，不缓存执行身份，不创建第二个 Worker Manager，也不修改测量协议。
- `ResultController` 组合 `AnalysisReportController` 与 `HtmlReportController`，分别持有统计/比较和 HTML 回执状态，通过同一个 App 调度文件 Worker；属性和公共回调只转发，不保存第二份报告状态。`offline_reads.py` 使重叠读取共用测量排斥租约，直到最后一个读取结束才释放；取消旧读取不阻塞新选择。`report_actions.py` 只根据 `OperationState`、结果可用性和执行种类计算控件规则。
- `AcprofTui` 仅是 Textual 宿主：事件路由、组合页面、全局互斥、Worker/Timer 和卸载回收桥接。少量公共 callback 方法委派给 `RunPresenter`，供 Textual 和线程安全回调使用；这些委派不是独立状态副本。
- `AppEventRoutes` 只转发当前事件，包括版本渠道、安装/重启与诊断定位。`BusyControlCache` 只缓存默认页面控件引用，在卸载时清除；忙闲仍即时读取 `OperationState`，并保留更新控制、自动预检暂停/恢复及待完成的表单校验。测量窗口继续禁止镜像刷新和表格缩放。
- `commands.py`、`i18n.py` 和 `acprof.tui` 包导入不能提前加载 Textual。重依赖仍在使用时才导入。
- `updates.py` 持有版本管理的页面状态与 App Worker；`acprof/updates/` 负责候选、安装身份与事务。
  安装是独立的维护进程，不属于采集执行会话。它及安装器继承 `MeasurementLock` 的两种内核保护，
  页面关闭不释放锁；`OperationState` 在更新期间及等待重启时阻止采集。恢复脚本在替换前复制，
  仅使用标准库与环境外解释器；用户配置与更新渠道状态分开保存。

## 行为不变量

1. 正式测量开始/结束由协议进度决定；输出解析中的 `measurement_active` 门控不得改变。下载进度、普通日志、动画、镜像轮询不能进入正式测量窗口。
2. `ProcessLifecycle.release` 只有匹配且确认退出的进程才能成功；归属判定不能把历史 CSV 误当作当前运行结果。
3. `ExecutionCoordinator.finish` 只允许匹配的执行种类、已回收的进程。成功/取消/失败的 UI 状态均源于纯 `evaluate_process_completion`。
4. CLI 进程仍是实际采集者；TUI 只是其外部协调器，不新增 TUI 到正式测量路径的工作。
5. Textual Worker 归属仍由 App 宿主管理；不因页面切换失去进程所有权。

## 结构化展示

结果概览、补采实验、采集计划、环境详情、任务选择和监控页头共用以下数据与组件契约。
状态映射、值格式化和页面投影由纯函数测试验证；Pilot 覆盖焦点、键盘、折叠、语言切换和既有操作路由，测量期门控另以事件与计时器行为核验。

`ui/view_models.py` 提供不依赖 Textual 的 `StatusSummary`、`ViewSection`、`FactRow` 与动作/提示数据。
状态码统一映射为文字、固定符号和语义等级；未知保持未知，零值不被当作缺失。原因码按权威类型映射，
原始异常只进入详情，不靠分析异常文字猜测业务状态。时间按现有实验选择器语义转本地秒，无偏移历史时间不推断为 UTC。

`ui/structured.py` 的 `StatusBanner`、`FactGrid`、`Callout` 只呈现传入数据并上抛稳定 action ID。
复用 Textual 的焦点、Grid、Collapsible 和语义 TCSS；完整原值与按 cell 宽度截断的显示分开。
普通字符串值不做翻译替换，结构化 Message 在显示时翻译；查看全文和复制有键盘入口，tooltip 只是补充。
语言切换更新已挂载控件，不重建相同形状的分组。数据改变行数时保留折叠选择和仍可识别的行焦点；原行删除后回到对应分组标题，不抢占其他控件的焦点。
分组异步重建前后均检查控件仍在 DOM 且未进入卸载；退出或移除控件时清除待重建标志，停止后续更新与焦点恢复。
简短弹窗可使用不折叠的 `FactGrid`，无动作的 Banner/Callout 不挂载空操作栏。搜索选择器复用
`ResizableDataTable`，表头排序保留所选对象与手动列宽；关闭后丢弃排队的过滤、详情与读取回调。
局部设置保存、Model Store、准备失败、恢复与 HTML 报告只展示各自操作结果，不能改变全局诊断或启动资格。
准备失败沿用当前类型化原因、异常类型及日志证据；推理参数确认保留现有控件，不把待确认参数标为已验证。
存储空间保留当前环境身份、分项错误与历史读数，不将旧读数和新释放估算混算；镜像详情不重复显示同一组事实。
实验配置继续只有 `RunPlanPanel` 一份摘要，快捷键帮助不额外维护一套配置差异状态。

首个迁移页面为 `results/overview.py::overview_view()` → `results/overview_panel.py`。
纯函数只组合既有元数据 Worker 与数据 Worker 的结果，区分产物存在、有效正式窗口和未核验状态。
展示层不重新读取 CSV，也不改变统计、恢复或采集门禁；恢复按钮仍调用 CatalogController / RecoveryController。
第二个迁移页面为 `results/backfill_view.py::backfill_view()` → `results/backfill_panel.py`。
纯模型使用既有 `ProfilerOverview`、读取阶段与 `OperationState`，不推断新的采集资格；工具总数和各工具的 CSV 记录数保留原口径。
检查失败、取消或过期时以 Callout 给出恢复操作，完整路径和异常保留在详情；只有数据读取器已确认的独立 CSV 才显示对应限制，不从异常文本猜测来源。
`ProfileController` 路由选择和重新检查动作，并在触发刷新前检查操作锁，避免过期事件清空已核验状态。相同视图数据不重复更新控件。
第三个迁移页面为 `run/planning.py::run_plan_view()` → `run/plan_panel.py::RunPlanPanel`。
`PlanView` 以同一个 `ViewSection` 合并原先上下两段配置摘要；估算沿用资源矩阵与输入规模口径，详细参数、预算、耗时假设和路径进入可展开行。
`ConfigurationController` 提供已选 runtime、预算和缓存路径；纯模型不读环境、文件或硬件，也不重复校验配置。
`invalid_plan_view()` 清空有效计划事实，配置问题交给状态中心和字段标识，计划区隐藏重复错误段落；controller 同时清除旧输入单位和规模提示，保留身份仍匹配时可复用的已解析计划。既有字段定位与采集门禁不变。准备期即使 busy 仍可更新输入计划，`measurement_active` 时不投影。
第四个迁移页面为 `run/environment_view.py` → `ui/environment.py::EnvironmentPopover`。
`run/global_status.py::build_status()` 以当前配置校验及 `PreflightController` 缓存为输入，生成不可变 `GlobalStatus`。
稳定根因 ID 去重，当前实验预检覆盖同 ID 的整机证据；失效、检查中、未请求分别分类。
配置错误使用 `config_<field>`，Process 配置校验失败不能冒充 `process_resources` 权限失败。
顶栏、问题列表、`preflight_run_reason()` 与“开始采集”消费同一份展示快照；未验证的必需检查、空检查结果也会阻止启动。
`allow_collection()` 只更新阻断状态，不主动导航。稳定检查 ID 映射字段、Process / Docker / RAPL 诊断或 perf / tcpdump 权限选项；无映射时不提供跳转。
整机清单保留已检查能力的可用性，失败仅是警告；未请求的预检占位项不覆盖该证据。清理失败持续阻断直到显式重检确认。
模型不读宿主、不启动探测、不保存第二份可变状态，正式执行仍保留原有权限与安全校验。
`environment_view.py` 将条目投影为 `EnvironmentCheckView`，问题用 `Callout`，正常、待确认和未请求用 `FactGrid`。
问题提供名称、原因、当前影响和下一步，通过显式 `status-open` 定位现有配置控件或权限诊断；Docker 异常项点击或按 Enter / Space 直接进入修复引导，悬停与聚焦仍只解释。
`NavigationContext` 只保存返回页面、焦点、滚动和最大化控件，不保存业务状态；专用 `IssueDiagnosticScreen` 从同一个快照读取诊断。
权限配置继续使用原有预览、确认、sudo 与测量锁。`PreflightController` 先使权限证据失效，再按稳定 ID 合并实际复检结果；
不以勾选或命令退出码判定权限通过，不覆盖过期配置的必需检查，保留清理失败的阻断语义。
Process 继承限制须确认且配置未被修改后才应用，绝不自动替换数值配额。
参考 [VS Code MarkerService](https://github.com/microsoft/vscode/blob/main/src/vs/platform/markers/common/markerService.ts)
的统一诊断统计与 [Carbon 状态指示](https://www.carbondesignsystem.com/building-blocks/core/patterns/status-indicators)
的符号、文字及颜色语义；仅复用设计思路，无新依赖。
`EnvironmentPopover` 保留分类独立固定、悬停预览、键盘选择和离开恢复；无选择时解释首个优先项，不主动夺取焦点。按稳定检查 ID 保留仍存在的选择，分类清单与详情各自可滚动。
技术详情由展示层翻译结构化消息、拼入原始状态与来源后按既有规则脱敏，不截断异常或路径。检查背景默认折叠，使用共享本地秒级时间格式化，列出平台、测量锁证据和检查范围。
重检使用稳定 action ID `environment-retry`，继续交给 `PreflightController.quick_check()` 和既有 `OperationState`；`retry_button` 提供当前唯一可见重检按钮，不依赖 Callout 内部控件 ID。正式测量时可交互查看缓存，但重检仍被拒绝。
环境浮层沿用原有 180 ms 悬停关闭定时器；本次不新增探测、周期刷新、轮询或缓存过期 TTL，也不因时间经过推导新的已过期状态。
Docker 诊断由 `run/diagnostics.py` 调用 `run/docker_diagnostics.py`，将 OS、CLI、有效 Context、端点、
产品类型及分类错误保存在同一个 `PreflightCheck.docker_diagnostic` 中；复用 host preflight 的本机 Socket / Desktop 判定。
Auto 回退到 Process 时保留该证据作为警告；显式 Docker 的失败继续阻断。Process 不依赖 Docker，但整机 Docker 故障仍提供可操作的警告。
`run/docker_guide.py` 纯投影安装与修复步骤，`run/docker_repair.py::DockerRepairScreen` 复用诊断弹窗生命周期和主题，
不新增设置页面、宿主探测、管理员执行器或状态所有者。重检仍调用 `PreflightController.quick_check()`，复用 Worker、操作门禁与 `offline_analysis_lock()`。
参考 [docker/cli Context 解析](https://github.com/docker/cli/blob/master/cli/command/cli.go)、
[Context 文档 PR](https://github.com/docker/cli/pull/7415) 与 [Docker 官方安装说明](https://docs.docker.com/engine/install/)；
采用环境变量优先级、失败不回退及官方软件源思路，不复制上游执行器，不引入新依赖。
第五个迁移页面为 `models/task_choice_view.py` → `models/task_choices.py::TaskChoices`。
`task_selection_view()` 纯投影既有 `TaskSelection`：共同配置使用 `ViewSection`，各组只呈现已计算的差异，次要行及采集组 ID、代表输入和全部语义任务 ID 进入详情；不重新分组、推断参数或修改执行签名。
原 JSON 路径、字段是否存在及带类型的值保留在详情中；缺失与 null、false、整数零和浮点零保持区别，未知路径和值不翻译或解释为 markup。
`selection_summary()` 只对有效的稳定组 ID 去重计数，以中性状态说明待选择或已选；“已选”不代表加载、推理或输出已验证。
`task_context_view()` 将任务选择请求中的模型信息放到列表后默认折叠的网格；普通 review 请求仍保留自己的字段展示，不改变其他准备弹窗。
`TaskChoice` 的勾选入口与网格详情、值查看和复制分离；`review_answers()` 仍只回传选中的稳定采集组 ID。空选择、全选、取消、单次确认及原有准备门禁继续由既有流程负责，不增加轮询或测量窗口内工作。
第六个迁移页面为 `run/monitor_view.py::monitor_header_view()` → `run/monitor_panel.py::MonitorHeader` → `ui/pages/monitor.py` 的监控页头。
`MonitorHeaderView` 只投影权威 `ProgressSnapshot`：顶部使用 `StatusSummary`，Case、资源及阶段适用的验证项使用 `ViewSection`，原始详情与非零警告/错误进入默认折叠网格。
Case 的“已处理”沿用 `completed_cases`，包含失败或跳过，不重新计算有效窗口；只有总数已知且已处理数在范围内才绘制进度。CPU、内存、GPU 的展示词汇与原始资源值分开，全文仍保留在详情。
监控进度使用静态 Rich 条形，由已有快照事件驱动；不引入 Textual 不定进度或周期动画。`RunPresenter` 继续拥有展示更新，既有 `status-elapsed` 时钟、阶段耗时、日志暂停与停止路径保持各自职责。
`measurement_active` 时不新增定时器、轮询或下载/阶段耗时刷新；测量边界仍由执行快照控制。普通日志不触发新的结构化投影，结束或失败后恢复既有交互。
测量期间推迟的页头语言更新，在既有的首个非测量进度事件中补刷一次；补刷使用当时的快照，不创建定时任务或第二份执行状态。

新组件没有轮询或周期定时器。结果概览的 case ProgressBar 只使用已知总数并关闭 ETA；未知总数隐藏条形，
不进入 Textual 的不定进度自动刷新模式。`measurement_active` 时不会投影/刷新新的概览、补采数据或采集计划。
这些约束须以事件/计时器测试验证，不能仅凭动画视觉上不可见判断。

设计参考 [Posting 2.10.0 的统一键值复制交互](https://github.com/darrenburns/posting/releases/tag/2.10.0)、
[Textual 的键盘焦点与 tooltip 边界](https://textual.textualize.io/guide/widgets/#tooltips)、
[Collapsible](https://textual.textualize.io/widgets/collapsible/) 与
[ProgressBar](https://textual.textualize.io/widgets/progress_bar/)；借鉴交互组织，不复制代码或增加依赖。

## 验收与设计来源

- `uv run pytest tests/tui/workflows/test_tui_execution_coordinator.py tests/tui/workflows/test_tui_process_lifecycle.py tests/tui/workflows/test_tui_process_completion.py`：所有权、异常与完成证据。
- `tests/tui/`：交互/导航/布局/日志/设置；`tests/visual/`：headless SVG；`tests/visual_png/`：VHS/xterm.js PNG 呈现验证（可选）。
- `uv run ruff check acprof/tui`、`uv run mypy acprof/tui`，以及 CLI 惰性导入与非 TUI 入口验证。

SVG 与 VHS/xterm.js PNG 验证界面呈现，不代表真实客户端终端、浏览器或剪贴板验收。
PNG 基线更新后复用同批捕获进行比较，只核对写入对应性；再次生成截图的独立回归证据单独记录。

参考 [Textual 的 Screen/Widget 职责](https://github.com/Textualize/frogmouth/tree/main/frogmouth)、
[Toolong 的进程/查看器分工](https://github.com/Textualize/toolong/tree/main/src/toolong) 和
[Textual Worker 生命周期讨论](https://github.com/Textualize/textual/issues/2593)。
只借鉴独立生命周期、数据流和展示职责，不复制源代码，也不依赖新框架。

实验选择保留目录，命令准备统一使用 `results.layout.resolve_result_file` 的单实验解析；多任务目录需显式选择。`results/overview.py` 只读取目录元数据，`results/data_status.py` 在后台让质量审计与 `inspect_profile_result` 借用一次结果快照，不调用统计聚合。两个 Worker 复用 App 的管理器、取消令牌和共享只读测量锁，返回时检查当前选择身份。HTML 来源校验只在后台进行，旧报告没有来源证据时保留 unverified；补采在用户确认后再次检查来源及互斥状态。

Results 与 Backfill 的路径变更消息在访问控件前检查 `is_running`、`form_ready` 和 `ui_closing`；
退出后不再同步选择或调度读取。[Textual 8.2.8 的退出流程](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/app.py)
先停止应用、卸载控件，最后才分发 App 的 `Unmount`，因此只在 `on_unmount` 清除 `form_ready` 不足以保护排队消息。
回归在请求退出后及控件部分卸载时重放真实 `Input.Changed`，复用现有生命周期状态，不增加等待或吞掉 `NoMatches`。
统计导出和 HTML 文件操作结束时先释放读取 token，再检查 App 是否存活；搜索弹窗检查附着、卸载与取消状态后才访问子控件。
Model Store 的容量、完成、清理及确认回调在卸载后停止，不能再访问已删除的事实表或启动新的清理。

快捷键说明参考 [Posting 的 HelpScreen](https://github.com/darrenburns/posting/blob/main/src/posting/help_screen.py)，
沿用 Textual 的模态焦点与关闭方式，不引入额外框架。
[Textual Screen 回调设计](https://github.com/Textualize/textual/pull/2321)及
[延迟关闭回调问题](https://github.com/Textualize/textual/pull/6637)用于审查生命周期风险；
项目仍使用锁定版本，以本地关闭/退出回归验证行为，不假定未安装的上游修复已经生效。

线程取消与 UI 回调核验参考 [Textual Workers 指南及示例](https://github.com/Textualize/textual/blob/main/docs/guide/workers.md)。复用已锁定的 Textual，不增加依赖；同一取消令牌传递给现有结果解析器，避免仅取消 Worker 而遗留正在执行的读取。
