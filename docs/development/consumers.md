# 结果分析与 TUI 模块

[← 返回专题目录](architecture.md)

## 结果分析与补采

`analysis/latency_model.py` 负责拟合、预测与验证，`latency_report.py` 负责报告和残差数据。
`build_latency_model_report()` 只计算并返回 `LatencyModelReport`，不创建目录或写文件；
CPU/GPU 拟合、尺度基函数说明和残差整理各有独立函数。`write_latency_model_report()`
负责发布既有 schema v2 JSON 与 CSV，包括跳过拟合时清空旧残差的行为。
`plotting` 内的 `config`、`data`、`styles` 分别管理图表声明、CSV 整理和样式；
`metrics`、`diagnostics`、`latency` 分别渲染常规指标、诊断图和模型图。

`analysis/model.py` 将历史或当前 CSV 转成统一长表、配置汇总和 Summary，不导入绘图库，
不回写采集产物。`metric_registry.py` 同时维护采集单位与展示方向、分组、聚合等语义；
展示信息直接声明在各 `Metric` 上；行组装字段由内部 `row_group` / `row_order` 和工具元数据派生，
不再在 client 中重复列出整组名称。内部分组不写入 `Metric.to_dict()`，保持 CSV 顺序及已有指标元数据。
分析专用派生量独立于 `CSV_FIELDS`。`cli/report.py` 负责参数与测量锁，
`plotting/report.py` 按需加载 Plotly 并内嵌 HTML/CSS/JavaScript；`view_model.js` 集中处理
baseline、颜色、Pareto 和 Scaling 分组，`report.js` 处理交互。该入口复用现有 CSV 和布局协议，
不引入 Web 服务或 TUI 依赖，详见[分析模型](../results/reports.md#统一分析模型与精简汇总)。

绘图函数从 `acprof.plotting.data`、`metrics`、`latency` 等模块导入，数值报告从
`acprof.analysis.latency_report` 导入。`acprof.cli.plot` 只解析参数和调度；已移除
旧函数包装、`SHOW_PLOTS` / `AGG_FUNC` 全局转发和重复指标清单。

补采包采用以下依赖关系：

```text
context ← backfill / plans / storage ← service ← CLI
```

- `context`：类型、常量、结果与 plan 读取、基础数值转换。
- `backfill`：结果行、静态元数据及 collection history 的更新计算。
- `plans`：工具适用性、完整性判断、计划复用、采集与合并。
- `storage`：活跃进程检查、锁、备份、临时文件发布与回滚。
- `service`：连接上述步骤的 `run_posthoc` 流程。

dry-run、已有数据完整性判断、计划复用、备份和发布顺序沿用既有语义。
项目根目录由 `context` 统一定位，避免更深的包目录影响 Dockerfile 和结果路径解析。

采集准备按 `resolve → interface validation → prepare runtime → runtime validation → matrix measurement` 执行。
接口检查不接触权重；完整运行验证使用每个选中设备的最小输入，全部成功才允许正式矩阵。
两者分别保存证据，不产生测量行，详见[运行兼容](../models/contracts.md#自动生成模型契约m1m6)。

## TUI 与兼容维护

`AcprofTui` 只继承终端光标适配 `BarCursorApp`；业务 controller 通过明确属性组合，
不继承 Textual `MessagePump`，也不把业务方法复制到 App 命名空间。
App 保留进程启动、停止及退出清理，正式测量快照、全局操作互斥、页面导航和 Textual 事件入口。
配置、镜像、预检、结果、恢复和选择器各自拥有本职责的状态；共享进程及 `read_jobs` 只有 App 一个拥有者。
`process.ProcessLifecycle` 统一拥有子进程和停止策略，`process_controller.ProcessController`
负责输出读取、跨线程 UI 回调、产物关联和未退出进程的延迟回收。
controller 通过 App 的 `project_dir` 与 `python_executable` 读取当前路径；不缓存或复制进程所有权。
正式测量窗口内延迟重要日志、抑制常规输出及定时器恢复沿用原语义。
`views` 以已注解的 `AcprofTui` 组合页面，通过 `app.configuration` 与 `app.localization` 读取相应能力。
`localization_controller.LocalizationController` 拥有 UI 文案登记、翻译和语言切换状态；
App 的 `tr/notify` 是 Textual 与小部件使用的展示接口，不翻译原始日志、CLI 参数或实验事件。
`configuration_controller` 管理表单、字段校验、设置保存、最近路径和预览；
`experiment_controller` 负责预设、采集及探测确认，实际 `launch` 仍由 App 执行。
`preflight_controller` 管理启动检测与模型准备弹窗，`result_controller` 管理摘要/报告读取及请求身份，
`profile_controller` 管理补采，`slash_controller` 分发快捷命令。
HTML 报告复用 `PendingLaunch` 与 `acprof report` 子进程；`result_controller` 只管理输出路径、
完成回执和展示，不导入分析或绘图库。`report_files` 负责已有 HTML 的发现、主动浏览器打开和独占创建导出文件；
文件操作仍交给 App worker manager，导出使用既有 MeasurementLock。`platform.can_open_local_browser`
集中判断 SSH / 图形桌面前提，浏览器失败不改变生成结果。没有新增运行依赖或测量窗口内的工作。
设计参考 [Textual worker 生命周期](https://github.com/Textualize/textual/blob/main/docs/guide/workers.md)、
[线程取消限制的 PR](https://github.com/Textualize/textual/pull/6514) 和
[Plotly HTML renderer](https://github.com/plotly/plotly.py/blob/main/plotly/io/_html.py)（均为 MIT，官方维护）；
使用已有调度和内嵌 Plotly.js，只借鉴边界设计，不复制上游源码或添加 Web 服务。
`log_controller` 管理日志批量展示、搜索、筛选和保留原文导出；`log_records` 保留有界原文并生成可逆的显示投影，
`SelectableLog` 继续负责 TextArea 选择、滚动及样式。采用 [TextArea.get_line](https://textual.textualize.io/widgets/text_area/#textual.widgets.TextArea.get_line)
扩展点（MIT），不往原文写 Rich markup，不引入独立日志框架。
`log_export_screen` 提供保存表单和逐级目录浏览，`log_export` 负责路径校验、目录读取及独占创建文件；
它们不调用实验的 `ArtifactLayout`。后台读写经 App 的 worker manager 调度，controller 持有 `read_jobs`
直到线程实际返回，取消弹窗不会提前释放读取任务。成功导出后复用配置 controller 的原子保存路径，
只更新 `last_log_export_dir`；读取配置不扫描或创建导出目录。
交互借鉴 [textual-fspicker 的 FileSave](https://github.com/davep/textual-fspicker/blob/main/src/textual_fspicker/file_save.py)
和 [Textual ModalScreen/Worker](https://github.com/Textualize/textual/blob/v8.2.8/docs/guide/workers.md)（MIT），
复用已锁定组件，不新增依赖。锁定版本的 [DirectoryTree](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/widgets/_directory_tree.py)
会忽略部分读取错误，因此浏览器使用 OptionList 显式展示错误，并独立跟踪取消后尚未结束的读线程；
相关边界参考上游 [权限错误 Issue #2564](https://github.com/Textualize/textual/issues/2564)
与 [异步目录加载测试 PR #3498](https://github.com/Textualize/textual/pull/3498)。所有目录 I/O 都在任务空闲时执行。
`host.download_progress` 通过 `huggingface_hub` 的 `tqdm_class` 扩展采集准备期进度，
`download_events` 是与正式事件独立的有界消息协议，`preflight_controller` 将其展示为独立下载区域。
`progress_widgets` 只渲染阶段与下载状态，复用主题语义色和 [Rich 的静态进度条](https://github.com/Textualize/rich/blob/v15.0.0/rich/progress_bar.py)（MIT）。
按锁定版本的接口使用确定进度；未知值保留文字，不使用 [Textual 的不确定进度动画](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/widgets/_progress_bar.py)，不新增依赖、计时器或测量期工作。
最低 Hub 版本 1.28 包含 [重试计数修复](https://github.com/huggingface/huggingface_hub/pull/4209)、
[Xet 双流进度](https://github.com/huggingface/huggingface_hub/pull/4400) 和聚合回调；依赖继续采用 Apache-2.0 许可。
仅借鉴与调用上游扩展点，不复制下载器、全局 monkey patch 或轮询缓存目录；Docker 运行锁不随主机 SDK 变更。
所有 `@on` 入口在 App 中按原事件顺序显式调用所属 controller；快捷键保留 Textual 的 `action_*` 入口。
业务调用使用具体类型的 `app.configuration.collect_config()` 等路径；没有字符串方法转发、
全量 App Protocol、新增 cast 或 `assert isinstance(self, AcprofTui)` 类型断言。
普通 controller 的线程任务用 `App.run_worker(partial(...))` 提交；保留原 group、thread、exclusive
和 exit_on_error 配置，所有 worker 仍归属同一个 App 生命周期。
这遵循 [Textual 8.2.8 的 work 实现](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/_work_decorator.py)
及 [组件数据流指南](https://textual.textualize.io/guide/widgets/#data-flow)（MIT）；只使用既有 API，
不新增调度器、后台事件循环或运行依赖。弹窗仍使用原生类型化结果和已有消息契约。
环境检测由 `platform` 提供；预检、设置保存、摘要与报告读取在实际 controller 模块中导入。
测试按实际查找位置替换依赖；不保留 App 的旧静态代理或尚未发布的 mixin 兼容入口。
本次边界调整不更改进度事件、CSV、resume、进程树所有权或测量窗口语义。
`run_form` 负责 RunConfig 字段映射、验证及 preset 匹配，不导入 Textual、不访问 widget。
`field_validation` 按稳定字段 ID 将共享校验错误呈现在现有控件旁；App 负责页面切换、展开和焦点。
`preparation` 用同一弹窗呈现等待、字段确认与失败重试。`review_inputs` 提供 Select、路径及 JSON 输入；
答案回到 resolver 重新解析，必填字段 resolved 后才允许确认，不在 TUI 改写最终 contract。
下载计划以已解析的 review 携带原始 bytes 报告，`downloads` 提供共享分区数据，供确认框和文本摘要使用，
`preparation` 显示“下载确认”并直接回传 confirm / cancel，不借用未决字段表单。
`host.model_errors` 保留模型查找失败的类型、身份和诊断；`detect` 抛出业务异常，公共 CLI dispatcher
将其转换为退出码。准备弹窗翻译结构化提示，原始诊断放入折叠详情。
`run_planning` 只计算准备阶段的配置、窗口和假设耗时摘要；实际档位由既有 host 输入计划经有界准备消息传入，不轮询产物。
`commands` 复用原来的命令构造函数，另持有 `PendingLaunch`、结果路径与 plot/stats/compare/profile 启动参数准备；
`app` 继续持有控件、busy/measurement 状态和进程生命周期；controller 通过 App 的准入检查操作控件，并管理各自的请求 token。
`commands.OperationState` 统一操作可用性；系统忙碌与存在可停止的子进程分别判断。
`run_results` 在子进程退出后关联现有 `run_state.run_id/attempts/pid` 与启动前快照，并调用 `audit_result`；
manifest 只用于路径路由，CSV 存在或退出码 0 都不单独构成当前运行成功证据，不另写结果协议。
摘要与报告线程保留独立请求身份，切换路径后旧结果只释放任务占用，不更新页面；退出阶段先检查应用生命周期。
后台读取尚未返回时继续禁止新的采集，避免取消 UI 请求后仍有读取干扰正式窗口。
`host.collection_workflow` 在原采集进程内执行准备阶段和用户裁决；`tui.preparation` 只呈现未决项／错误，
通过有界、带请求 ID 的 stdin 回复继续同一进程。`preparation_events` 定义独立的版本化准备消息，
不复用正式测量边界事件，不依赖日志错误字符串决定是否询问。普通 CLI 保持非交互失败行为。
任务选择由 `host.task_selection` 传递结构化推理组；`task_presentation` 只从规范化的
`execution_profile.signature` 提取共同／差异字段，并从 `semantic_tasks` 构建用途和代表输入数据。
它不重新分组、不推断默认值，不导入 Textual。`tui.task_labels` 负责统一数据的中英文格式化，
`tui.task_choices` 负责可换行多选项和键盘操作；回复只包含已有稳定组 ID。
完整组件、IO、声明与指纹保存在实验元数据，普通选择界面展示用途、推理操作及有效计算差异。
唯一接口直接继续；配置冲突由同一准备流程请求缺失信息，不另列底层架构或重复的任务选择控件。
配置展示借鉴 [Transformers GenerationConfig](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/generation/configuration_utils.py)
（Apache-2.0）的完整配置与展示差异分离；[PR #47953](https://github.com/huggingface/transformers/pull/47953)
说明了默认值与显式覆盖的优先级风险，因此界面复用后端已生效的签名。
布局依据 [Textual 8.2.8 SelectionList](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/widgets/_selection_list.py)
的单行限制与 [Issue #5418](https://github.com/Textualize/textual/issues/5418)，用现有 Checkbox、Static 和滚动容器组合，
不覆盖框架内部绘制方法。只借鉴官方维护项目的分工，不复制源码或增加依赖；处理只在准备阶段发生。
确认缓存保存于结果根目录之外的 `.model-contracts/`；静态身份变化使其失效，运行证据始终独立验证。
重试只回到失败准备阶段；显式重建环境会连带重做输入与验证。退出和取消沿用原进程组清理机制。
设计借鉴 [Transformers Pipeline 的唯一候选选择](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/pipelines/__init__.py)
（Apache-2.0）和 [Textual 的弹窗结果回传](https://github.com/Textualize/textual/discussions/2559)（项目为 MIT）。
只复用接口思路，沿用当前 Textual 与标准库，不引入依赖或复制 loader；消息只在准备阶段发送，维护和测量成本局限在现有边界内。
七页底栏共用 `.action-bar`，内部由 `.action-secondary` 和 `.action-primary` 两个 `Horizontal`
分别承载左侧次要／导航动作与右侧主操作；间距由容器分配，按钮宽度随标签变化。
`acprof.experiment` 定义共享 `RunConfig`、`ConfigIssue`、`RunConfigError`、校验与 `build_run_command`，
CLI 参数、TUI 表单和硬件验证使用同一契约；该模块不导入 TUI。
`RunConfigError.issues` 保留字段名与可翻译原因，CLI 文本附带字段名，TUI 不解析错误字符串。
字段错误元数据与候选控件参考 [Textual validation](https://github.com/Textualize/textual/blob/main/src/textual/validation.py)
和 [Input 文档](https://github.com/Textualize/textual/blob/main/docs/widgets/input.md)（MIT）。
沿用现有 Textual 与标准库；共享校验不引入 UI 依赖，界面校验只在编辑或提交阶段执行，不进入正式测量窗口。
`acprof.messages` 保存可翻译的结构化消息；`tui.translation_catalog` 为 927 条已审校显示文案
固定 `ui.NNNN` 标识及中文源文案、英文翻译，键只追加不重新编号。
`tui.i18n.message_id(key, *values)` 构造保留 msgid 的消息。TUI 内静态 catalog `message(...)` 和直接 `tr(...)` 文案必须通过稳定 ID 引用；`test_tui_i18n.py` 以 AST 约束防止退化为中文键。
`translate` 在展示边界按标识查找，
允许文本变化而不影响身份；已有 `message(中文源文本)` 及少量直接字符串调用仍可从同一目录生成
`ENGLISH` 源文案索引逐步迁移，避免改变验证错误文本、CLI 与结果协议。
缺失 ID 直接抛出 `KeyError`，参数位置／格式必须与两种语言模板一致，禁止回退成任意字符串。
`commands` 保留 probe／统计／绘图等界面命令，`progress` 仅从版本化控制事件更新进度，
`stages.Stage` 是进度与最终进程状态的稳定机器标识；其中文源标签通过 `Stage.label`
在 `app`、命令回显等展示边界翻译，状态判断、状态样式和测试只比较枚举身份。
修改标签或英语翻译不会改变事件解析与状态转移。该状态不写入实验 CSV、resume、progress event
等持久化协议，不为此引入兼容映射。
`progress_events` 的有界 v1 消息覆盖矩阵计划、case 选择／跳过、测量开始／停止、
结果发布以及最大尺度探测。资源编号、结果路径与探测耗时有类型校验，普通日志只提供
诊断及警告计数，不再通过正则表达式改变阶段、case 身份、测量边界或结果路径。
准备消息继续使用独立的 `preparation_events` 协议，包含计算／执行 profiler 阶段。
生产者仅在窗口外发送低频事件；最大尺度请求的事件位于计时之前和请求结束之后。
`run_finished(status=ok)` 只进入产物核验状态，子进程退出后仍按持久化 run/attempt 身份及
审计结果决定最终成功。TUI 继续以子进程运行 CLI，不把测量循环搬入 Textual 事件循环。
最大尺度探测同样要求完成事件与 summary 路径；退出后读取既有 schema v3 summary，
核对当前模型、probe 输出目录中的进程 PID、文件时间以及资源／输入／耗时与事件的一致性。
缺失、旧文件、无效 schema 或事件与文件不一致都不能以退出码 0 标为探测成功。
`diagnostics` 负责提示性预检和结果摘要。TUI 提示性检查与 CLI 权威检查保留各自用途。
启动检查直接复用 `diagnostics.quick_preflight` 及其 host 探测，不新增环境检测器。`app` 在首屏
刷新后调用现有 thread worker，以测量锁隔离诊断，保存请求身份、检查配置和结果；UI 回调只接受
当前请求，退出后的结果不再访问控件。检查不锁定表单，但采集需等待结果且不能使用失效配置。
正常结果不产生 UI 通知；`warn` 只呈现对应能力限制，`fail` 与检查异常阻断采集，最终运行仍须通过
CLI 的权威预检。`views.EnvironmentPreflightScreen` 只展示缓存问题和返回重试意图。
线程生命周期与主线程更新参考 [Textual 8.2.8 worker](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/worker.py)
和 [上游线程安全说明](https://github.com/Textualize/textual/discussions/2853)（MIT）。复用项目已锁定且
上游维护的 API，不复制框架源码、不新增依赖；诊断仅在启动或显式重试时运行，不进入测量窗口。
`presentation` 统一数值输入格式、十进制 B/KB/MB/GB 大小显示与不适用、计算中、未知的显示标记。
下载、镜像与存储页面共用大小格式化函数；诊断 JSON 的展示副本不修改原始 bytes 或保存结果。
确认交互参考 [Textual 的 ModalScreen 示例](https://github.com/Textualize/textual/blob/v8.2.8/docs/examples/guide/screens/modal02.py)，
单位切换参考 [humanize 的十进制格式化](https://github.com/python-humanize/humanize/blob/main/src/humanize/filesize.py)（均为 MIT）；
仅借鉴展示方式，沿用现有 Textual 和标准库，不新增依赖或测量期开销。
`reports` 用标准库校验已有统计/对照 JSON，并提供带单位和口径的表格数据；不加载 Textual 或采集依赖。
统计页通过 `commands.build_stats_command` 启动既有 `acprof stats`，沿用 App 的进程互斥、停止和日志流程；
完成后在后台读取一次报告并更新表格。读取期间锁定启动入口，允许切换读取目标和退出；不定时扫描 CSV 或自动运行开销实验。
绘图页摘要复用 `analysis.uncertainty.summarize_windows` 的分组、过滤和窗口均值，只关闭 bootstrap。
借鉴 [Textual 8.2.8 thread worker 示例](https://github.com/Textualize/textual/blob/v8.2.8/docs/examples/guide/workers/weather05.py)
及 [Worker 实现](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/worker.py) 的取消与 UI 回传边界（MIT，当前已有依赖）；
请求身份在主线程再次核对，取消线程任务不等于底层工作已停止。沿用当前版本，不复制线程框架或增加依赖；所有结果读取都在测量窗口之外。
`experiment_catalog` 只扫描显式已知结果根目录，以 `run_id` 合并路径副本，按元数据、枚举与重复内容预算限制读取；
`catalog_controller` 将这些记录接入已有路径框，`experiment_picker` 的取消线程在实际返回后才释放采集互斥。
`recovery` 在启动前读取原实验，复用 `host.run_state.validate_resume` 检查身份与产物；
`recovery_controller` 统一承接开始采集、历史选择和失败监控页，提供准备重试、续跑或自动新建实验。
确认后重新核验记录；需要切换采集模式时复用现有环境预检，配置变化或过期 worker 不得启动进程。
准备重试的证据归档由持有目录锁的 `RunState` 完成，TUI 的评估及目录分配始终只读。
流程借鉴 [Ray Tune 的恢复与重启区分](https://github.com/ray-project/ray/blob/master/python/ray/tune/tuner.py)
和 [Accelerate 的恢复目录管理](https://github.com/huggingface/accelerate/pull/1741)（均为 Apache-2.0 项目）；
只复用设计思想，保留 AC-Prof 的 JSON 与严格测量身份协议，不复制训练状态机制、不新增依赖或测量窗口开销。
选择器在同一个空闲 worker 中检查搜索目录，显示有效路径摘要，并在悬停提示中保留完整路径及跳过原因；
父子路径合并只用于摘要，不改变有深度上限和实验目录边界的实际扫描入口。借鉴
[pytest 的 collection 路径规范化](https://github.com/pytest-dev/pytest/blob/main/src/_pytest/main.py)（MIT）思路，
使用标准库 `pathlib` 与已有 Textual worker 实现，不复制其扫描策略、不新增依赖或测量期活动。
根级 `acprof.run_options.RunOptions` 是共享配置字段、默认值、CLI 元数据和 run/probe 适用性的唯一声明；
`RunConfig` 继承这些字段并负责验证与预设，两个 parser 和命令构造器消费同一 schema。
命令参数及 TUI 命令预览按共享字段的声明顺序生成；参数含义与历史配置格式保持一致。
`acprof.run_args` 保留高级参数及冻结参数序列化；CLI 的 `download_args` 仅负责执行时应用环境变量。
新增共享字段无需再同步表单配置、run/probe parser 和命令字段名单。
该方式参考 [Transformers HfArgumentParser](https://github.com/huggingface/transformers/blob/main/src/transformers/hf_argparser.py)
的 dataclass 字段与 metadata 设计，使用标准库实现，不引入 Transformers 主机依赖。
`run_args.arguments_from_options` 从同一 parser 定义序列化冻结参数；`RunConfig.extra_options` 仅保留未显式映射的公共选项，
复用不丢失 seed、SLO 等条件，续跑继续交给既有 RunState 校验。索引不作为新的身份或权威数据库。
设计参考 [MLflow RunInfo](https://github.com/mlflow/mlflow/blob/master/mlflow/entities/run_info.py) 对 run_id 与 artifact_uri 的分离
（Apache-2.0），以及 [Textual Input suggester](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/suggester.py)
和原生 ModalScreen/DataTable（MIT）。只借鉴稳定身份、局部搜索与原生事件模式，不引入 MLflow、数据库或新依赖。
`model_candidates` 复用 runtime profile 路由、依赖锁与 Model Store entry 验证；`model_controller` 把候选回填原 ModelInput，
runtime/host/device/资源条件不完整或变化时降为需要重新验证。扫描与至多一次平台探测只在空闲 worker 中执行，
不读取权重内容、不自动下载，也不凭历史成功跳过正式准备验证。
`images` 提供镜像树、筛选、摘要与折叠详情、层引用和可滚动的删除确认；`ImageDetailPanel` 按镜像/层身份维护展开状态，将用户信息、完整依赖和诊断依据分组。`views` 构建三个视图，`image_controller.ImageController` 收纳镜像页事件、渲染及 Docker worker。
`ImageController` 拥有清单、选择、计时器和存储弹窗状态，App 显式路由事件；Docker worker 通过 App 调度，配置模块仍不提前加载 Textual。
`storage.StorageSpaceScreen` 展示可滚动的存储空间弹窗；`host.images.inventory.read_storage` 读取固定 Docker 连接的分类汇总和可核验的本机数据目录文件系统。
存储 worker 由 `ImageController` 管理，打开和手动刷新时才查询；关闭弹窗不取消尚在执行的 Docker 子进程，查询完成前保持与采集互斥。
`ImageWorkspace` 按可用空间分配列表和详情高度；`ImageDetailResizeHandle` 使用 Textual 鼠标捕获和屏幕坐标处理上下拖动，也支持聚焦后按键调整。
两侧各保留至少三行，手动高度仅存于控件的本次会话，窗口缩小不覆盖偏好。拖动只触发布局更新；禁用、隐藏、窗口缩放、失去捕获或按 `Esc` 时释放鼠标，沿用镜像控件的任务互斥，不增加后台扫描或定时器。
`table.ResizableDataTable` 为统计报告和镜像管理的表格提供统一表头边界拖动，按稳定 column key 在控件内保留本次会话的手动列宽。
拖动边界只存在于相邻列之间；末列右沿不绘制手柄，也不参与拖动命中。
`images.ImageTreeHeader` 复用该控件，更新树节点的列宽，并同步表头与树的横向滚动；拖动不重建树节点或改变折叠状态。
镜像列表通过原生 `fixed_columns=1` 只固定勾选列，“环境 / 模型”与其余数据列一起横向滚动。
列重建复用手动值，未调整的列仍采用页面默认宽度；设置文件不保存列宽。拖动只更新列缓存及滚动范围，禁用、隐藏或任务开始时释放鼠标。
Docker 访问由标准库模块 `host.images.inventory` 执行，固定连接并复核 daemon ID、镜像 ID 和全部标签。
打开镜像页自动读取清单，空闲时每轮完成后 5 秒更新；切换筛选和语言只操作内存中的清单。
离开页面或运行任务时暂停计时器；确认框和鼠标拖动期间不启动扫描。查询期间禁止启动实验，但保留镜像浏览交互。
自动更新保留有效选择和浏览状态；查询失败保留上次清单并自动重试。删除仍按用户确认的快照复核。

settings、i18n、themes、input、log、scrollbar 各自管理设置、语言、主题和控件。
`environment.EnvironmentSettingsScreen` 管理连接表单与显式权限配置；`host.env_utils` 负责
白名单字段校验、私有原子保存和环境更新，`host.permissions` 生成并复核固定的系统授权计划。
权限配置通过 Textual 的原生 `App.suspend()` 交给系统 sudo 终端，测量、doctor 和 preflight
不会调用该安装路径；无新增 Python 依赖。复用现有 MIT 许可的 Textual，终端交接参考
[上游 suspend 文档](https://github.com/Textualize/textual/blob/main/docs/guide/app.md#suspending-your-app)。
env 文件保留非目标行、原子替换和引号处理参考 BSD-3-Clause 许可的
[python-dotenv](https://github.com/theskumar/python-dotenv/blob/main/src/dotenv/main.py) 思路，
使用标准库实现项目所需子集，不引入 shell 展开或完整 dotenv 语法。
日志抑制保留错误/警告的整段续行，测量窗口结束后统一显示。
`rendering.CjkCompositor` 用于主屏幕和确认屏幕，合并同一控件可见的连续片段，避免被遮挡控件的边界
拆散中文宽字符；局部刷新按实际片段宽度输出，并完整重画与脏区域相交的片段，避免只刷新半个汉字。
这是针对 [Textual #6357](https://github.com/Textualize/textual/issues/6357) 的应用内适配，参考其
[修复讨论](https://github.com/0x7c13/textual/pull/1) 的合并思路，保留原生遮挡、样式与点击信息。
不修改 Textual 全局类，不增加依赖、定时器或刷新次数；升级 Textual 时需重新核对私有 compositor API
及 `test_tui_cjk_rendering.py` 的完整帧、局部输出和浮层交互回归。
CSS 路径相对 App 文件明确定位；设置文件位置、版本、项目隔离算法和恢复优先级保持一致。
TUI 应用从 `acprof.tui.app` 导入；共享配置与运行命令从 `acprof.experiment` 导入，
界面专用命令从 `acprof.tui.commands` 导入；旧 `acprof.cli.tui_*` 模块已删除。

测试覆盖当前实现与旧入口拒绝行为；不为历史调用增加转导出或参数别名。
测试选择、终端证据与验证范围统一见[测试指南](testing.md)。
