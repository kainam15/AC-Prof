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

`app` 保留顶层 UI 状态、子进程启动/回收、测量窗口状态和退出清理；`process.ProcessLifecycle` 统一拥有子进程及停止策略。`views` 使用页面构建函数输出 TabPane 子树。
`localization_actions.LocalizationActions` 只处理 UI 文案登记、翻译和语言切换，
通过现有 Textual MRO 提供 `tr/notify` 等操作；`app` 仍拥有首屏、业务状态和页面生命周期，
不把日志原文、命令参数或实验事件交给翻译模块。
`configuration_actions` 管理表单、字段校验、保存偏好与预览，`experiment_actions` 负责预设、采集/探测确认与最近路径记忆，实际 `_launch` 仍在 App。
`preflight_actions` 管理启动检测与模型准备弹窗，`result_actions` 管理异步摘要/报告读取、取消 token 和只读展示，`profile_actions` 管理补采操作，`slash_actions` 分发快捷命令。
各 mixin 继承 Textual `MessagePump`，由 `AcprofTui` 的 MRO 集成事件处理，用户可见消息和工作线程的身份检查沿用原契约。
对原先依赖 `acprof.tui.app` 的测试/可替换依赖（环境检测、预检、设置保存、摘要与报告读取）由 App 的窄方法转交；工作目录/解释器由 App 属性读取，不在 mixin 中缓存路径。
职责拆分不会更改原始进度、CSV、resume、进程树所有权或测量窗口行为；可参考 [Textual MessagePump 分发实现](https://github.com/Textualize/textual/blob/main/src/textual/message_pump.py)（MIT），只借鉴框架契约，不复制实现。
`run_form` 负责 RunConfig 字段映射、验证及 preset 匹配，不导入 Textual、不访问 widget。
`field_validation` 按稳定字段 ID 将共享校验错误呈现在现有控件旁；App 负责页面切换、展开和焦点。
`preparation` 用同一弹窗呈现等待、字段确认与失败重试。`review_inputs` 提供 Select、路径及 JSON 输入；
答案回到 resolver 重新解析，必填字段 resolved 后才允许确认，不在 TUI 改写最终 contract。
下载计划以已解析的 review 携带原始 bytes 报告，`downloads` 在 TUI 中生成摘要与详情，
`preparation` 显示“下载确认”并直接回传 confirm / cancel，不借用未决字段表单。
`host.model_errors` 保留模型查找失败的类型、身份和诊断；`detect` 抛出业务异常，公共 CLI dispatcher
将其转换为退出码。准备弹窗翻译结构化提示，原始诊断放入折叠详情。
`run_planning` 只计算准备阶段的配置、窗口和假设耗时摘要；实际档位由既有 host 输入计划经有界准备消息传入，不轮询产物。
`commands` 复用原来的命令构造函数，另持有 `PendingLaunch`、结果路径与 plot/stats/compare/profile 启动参数准备；
`app` 继续持有控件、busy/measurement 状态和进程生命周期；各 action mixin 操作 App 持有的控件与请求 token，不另起 controller 或私有状态副本。
`commands.OperationState` 统一操作可用性；系统忙碌与存在可停止的子进程分别判断。
`run_results` 在子进程退出后关联现有 `run_state.run_id/attempts/pid` 与启动前快照，并调用 `audit_result`；
manifest 只用于路径路由，CSV 存在或退出码 0 都不单独构成当前运行成功证据，不另写结果协议。
摘要与报告线程保留独立请求身份，切换路径后旧结果只释放任务占用，不更新页面；退出阶段先检查应用生命周期。
后台读取尚未返回时继续禁止新的采集，避免取消 UI 请求后仍有读取干扰正式窗口。
`host.collection_workflow` 在原采集进程内执行准备阶段和用户裁决；`tui.preparation` 只呈现未决项／错误，
通过有界、带请求 ID 的 stdin 回复继续同一进程。`preparation_events` 定义独立的版本化准备消息，
不复用正式测量边界事件，不依赖日志错误字符串决定是否询问。普通 CLI 保持非交互失败行为。
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
`acprof.messages` 保存可翻译的结构化消息，翻译表仍属于 `tui.i18n`。
`commands` 保留 probe／统计／绘图等界面命令，`progress` 解析运行日志，
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
`catalog_actions` 将这些记录接入已有路径框，`experiment_picker` 的取消线程在实际返回后才释放采集互斥。
`recovery` 在启动前读取原实验，复用 `host.run_state.validate_resume` 检查身份与产物；
`recovery_actions` 统一承接开始采集、历史选择和失败监控页，提供准备重试、续跑或自动新建实验。
确认后重新核验记录；需要切换采集模式时复用现有环境预检，配置变化或过期 worker 不得启动进程。
准备重试的证据归档由持有目录锁的 `RunState` 完成，TUI 的评估及目录分配始终只读。
流程借鉴 [Ray Tune 的恢复与重启区分](https://github.com/ray-project/ray/blob/master/python/ray/tune/tuner.py)
和 [Accelerate 的恢复目录管理](https://github.com/huggingface/accelerate/pull/1741)（均为 Apache-2.0 项目）；
只复用设计思想，保留 AC-Prof 的 JSON 与严格测量身份协议，不复制训练状态机制、不新增依赖或测量窗口开销。
选择器在同一个空闲 worker 中检查搜索目录，显示有效路径摘要，并在悬停提示中保留完整路径及跳过原因；
父子路径合并只用于摘要，不改变有深度上限和实验目录边界的实际扫描入口。借鉴
[pytest 的 collection 路径规范化](https://github.com/pytest-dev/pytest/blob/main/src/_pytest/main.py)（MIT）思路，
使用标准库 `pathlib` 与已有 Textual worker 实现，不复制其扫描策略、不新增依赖或测量期活动。
根级 `acprof.run_args` 集中维护 CLI、共享配置与 TUI 使用的纯参数声明、下载参数和冻结参数序列化；
它只依赖低层常量与 argparse，不反向导入 CLI。CLI 的 `download_args` 仅负责执行时应用环境变量。
`run_args.arguments_from_options` 从同一 parser 定义序列化冻结参数；`RunConfig.extra_options` 仅保留未显式映射的公共选项，
复用不丢失 seed、SLO 等条件，续跑继续交给既有 RunState 校验。索引不作为新的身份或权威数据库。
设计参考 [MLflow RunInfo](https://github.com/mlflow/mlflow/blob/master/mlflow/entities/run_info.py) 对 run_id 与 artifact_uri 的分离
（Apache-2.0），以及 [Textual Input suggester](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/suggester.py)
和原生 ModalScreen/DataTable（MIT）。只借鉴稳定身份、局部搜索与原生事件模式，不引入 MLflow、数据库或新依赖。
`model_candidates` 复用 runtime profile 路由、依赖锁与 Model Store entry 验证；`model_actions` 把候选回填原 ModelInput，
runtime/host/device/资源条件不完整或变化时降为需要重新验证。扫描与至多一次平台探测只在空闲 worker 中执行，
不读取权重内容、不自动下载，也不凭历史成功跳过正式准备验证。
`images` 提供镜像树、筛选、摘要与折叠详情、层引用和可滚动的删除确认；`ImageDetailPanel` 按镜像/层身份维护展开状态，将用户信息、完整依赖和诊断依据分组。`views` 构建三个视图，`image_actions.ImageActions` 收纳镜像页事件、渲染及 Docker worker。
`ImageActions` 继承 Textual 的 `MessagePump`，通过原生事件继承和 `@work` 保留调度；`AcprofTui` 持有状态、计时器和进程管理器，配置模块仍不提前加载 Textual。
`storage.StorageSpaceScreen` 展示可滚动的存储空间弹窗；`host.image_management.read_storage` 读取固定 Docker 连接的分类汇总和可核验的本机数据目录文件系统。
存储 worker 由 `ImageActions` 管理，打开和手动刷新时才查询；关闭弹窗不取消尚在执行的 Docker 子进程，查询完成前保持与采集互斥。
`ImageWorkspace` 按可用空间分配列表和详情高度；`ImageDetailResizeHandle` 使用 Textual 鼠标捕获和屏幕坐标处理上下拖动，也支持聚焦后按键调整。
两侧各保留至少三行，手动高度仅存于控件的本次会话，窗口缩小不覆盖偏好。拖动只触发布局更新；禁用、隐藏、窗口缩放、失去捕获或按 `Esc` 时释放鼠标，沿用镜像控件的任务互斥，不增加后台扫描或定时器。
`table.ResizableDataTable` 为统计报告和镜像管理的表格提供统一表头边界拖动，按稳定 column key 在控件内保留本次会话的手动列宽。
拖动边界只存在于相邻列之间；末列右沿不绘制手柄，也不参与拖动命中。
`images.ImageTreeHeader` 复用该控件，更新树节点的列宽，并同步表头与树的横向滚动；拖动不重建树节点或改变折叠状态。
镜像列表通过原生 `fixed_columns=1` 只固定勾选列，“环境 / 模型”与其余数据列一起横向滚动。
列重建复用手动值，未调整的列仍采用页面默认宽度；设置文件不保存列宽。拖动只更新列缓存及滚动范围，禁用、隐藏或任务开始时释放鼠标。
Docker 访问由标准库模块 `host.image_management` 执行，固定连接并复核 daemon ID、镜像 ID 和全部标签。
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
