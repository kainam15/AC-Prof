# TUI 模块边界与运行会话所有权

AC-Prof 的 TUI 使用 **Textual 应用外壳 + 页面组合 + 执行协调器 + 业务功能模块**。目录边界反映实际职责，不把所有业务对象强制收归 `AcprofTui`。

## 实现结构

```text
acprof/tui/
├── app.py                  # Textual 应用外壳、导航、事件、互斥和 worker 宿主
├── commands.py             # 无 Textual 的命令/操作策略
├── i18n.py                 # 翻译目录，导入时不加载 Textual
├── catalog/                # 翻译文案的静态词典
├── experiments/            # 实验历史与选择器
├── images/                 # 镜像列表、操作与存储
├── logging/                # 日志保留、显示、导出
├── models/                 # 模型候选和 Store 确认
├── results/                # 历史结果、统计、报告与 profiler
│   ├── controller.py       # 实验选择、独立元数据/数据 Worker 与操作控件副作用
│   ├── html_controller.py  # HTML 生成回执、完成状态及打开/导出操作
│   └── report_actions.py   # 不依赖 Textual 的按钮可用性/显示纯计算
├── run/                    # 执行、预检、恢复、进度与命令呈现
│   ├── execution.py        # 唯一执行会话/身份和进程所有者
│   ├── process.py          # 进程/进程组启动、停止和回收
│   ├── process_controller.py # 工作线程读取、事件、延迟回收
│   ├── presenter.py        # 当前运行信息的 Textual 呈现，无执行所有权
│   └── results.py          # 只读归属检查与纯完成判定
├── settings/               # 表单配置、语言和本地首选项
└── ui/
    ├── dialogs.py          # 对话框
    ├── pages/              # Run / Monitor / Results(在 results/) / Backfill / Images / Settings
    └── ...                 # 公共控件、布局及主题
```

## 所有权与依赖

- `ExecutionCoordinator` 是当前执行尝试的唯一拥有者。它持有 `ExecutionSession`（执行种类、命令、Token、Snapshot、停止标记与归属结果）和唯一的 `ProcessLifecycle`。新尝试必须取得新 Token，旧结果和取消回调不能进入新会话。
- `ProcessLifecycle` 只能在目标进程确认退出后释放。清理超时保留进程占用状态，禁止新实验开始；停止必须携带期望的进程对象，防止迟到的 SIGINT 误伤替换进程。
- `run/process_controller.py` 使用唯一 App Worker Manager 调度日志读取和失败后的回收。Textual 窗口卸载不等于采集进程结束；UI 回调必须在 App 存活且 Token 匹配时投递。
- `RunPresenter` 仅修改现有控件、焦点、通知和计时器。它通过 App 的明确接口读取 `ExecutionCoordinator`，不缓存执行身份，不创建第二个 Worker Manager，也不修改测量协议。
- `ResultController` 组合 `HtmlReportController`，后者持有 HTML 生成回执和完成状态，并通过 App 调度文件 Worker；二者都不持有执行会话或进程。`report_actions.py` 只根据 `OperationState`、结果可用性和执行种类计算控件规则，测量互斥仍由 App 的同一个 `OperationState` 提供。
- `AcprofTui` 仅是 Textual 宿主：事件路由、组合页面、全局互斥、Worker/Timer 和卸载回收桥接。少量公共 callback 方法委派给 `RunPresenter`，供 Textual 和线程安全回调使用；这些委派不是独立状态副本。
- `commands.py`、`i18n.py` 和 `acprof.tui` 包导入不能提前加载 Textual。重依赖仍在使用时才导入。

## 行为不变量

1. 正式测量开始/结束由协议进度决定；输出解析中的 `measurement_active` 门控不得改变。下载进度、普通日志、动画、镜像轮询不能进入正式测量窗口。
2. `ProcessLifecycle.release` 只有匹配且确认退出的进程才能成功；归属判定不能把历史 CSV 误当作当前运行结果。
3. `ExecutionCoordinator.finish` 只允许匹配的执行种类、已回收的进程。成功/取消/失败的 UI 状态均源于纯 `evaluate_process_completion`。
4. CLI 进程仍是实际采集者；TUI 只是其外部协调器，不新增 TUI 到正式测量路径的工作。
5. Textual Worker 归属仍由 App 宿主管理；不因页面切换失去进程所有权。

## 结构化展示

`ui/view_models.py` 提供不依赖 Textual 的 `StatusSummary`、`ViewSection`、`FactRow` 与动作/提示数据。
状态码统一映射为文字、固定符号和语义等级；未知保持未知，零值不被当作缺失。原因码按权威类型映射，
原始异常只进入详情，不靠分析异常文字猜测业务状态。时间按现有实验选择器语义转本地秒，无偏移历史时间不推断为 UTC。

`ui/structured.py` 的 `StatusBanner`、`FactGrid`、`Callout` 只呈现传入数据并上抛稳定 action ID。
复用 Textual 的焦点、Grid、Collapsible 和语义 TCSS；完整原值与按 cell 宽度截断的显示分开。
普通字符串值不做翻译替换，结构化 Message 在显示时翻译；查看全文和复制有键盘入口，tooltip 只是补充。
语言切换更新已挂载控件，不重建相同形状的分组、不丢弃焦点或折叠选择。

首个迁移页面为 `results/overview.py::overview_view()` → `results/overview_panel.py`。
纯函数只组合既有元数据 Worker 与数据 Worker 的结果，区分产物存在、有效正式窗口和未核验状态。
展示层不重新读取 CSV，也不改变统计、恢复或采集门禁；恢复按钮仍调用 CatalogController / RecoveryController。
其他页面按截图评审逐页迁移，不通过本次改动提前替换运行监控等页面。

新组件没有轮询或周期定时器。case ProgressBar 只使用已知总数并关闭 ETA；未知总数隐藏条形，
不进入 Textual 的不定进度自动刷新模式。`measurement_active` 时不会投影/刷新新的概览数据。
这些约束须以事件/计时器测试验证，不能仅凭动画视觉上不可见判断。

设计参考 [Posting 2.10.0 的统一键值复制交互](https://github.com/darrenburns/posting/releases/tag/2.10.0)、
[Textual 的键盘焦点与 tooltip 边界](https://textual.textualize.io/guide/widgets/#tooltips)、
[Collapsible](https://textual.textualize.io/widgets/collapsible/) 与
[ProgressBar](https://textual.textualize.io/widgets/progress_bar/)；借鉴交互组织，不复制代码或增加依赖。

## 验收与设计来源

- `uv run pytest tests/tui/workflows/test_tui_execution_coordinator.py tests/tui/workflows/test_tui_process_lifecycle.py tests/tui/workflows/test_tui_process_completion.py`：所有权、异常与完成证据。
- `tests/tui/`：交互/导航/布局/日志/设置；`tests/visual/`：SVG；`tests/visual_png/`：真实终端 PNG 验证（可选）。
- `uv run ruff check acprof/tui`、`uv run mypy acprof/tui`，以及 CLI 惰性导入与非 TUI 入口验证。

参考 [Textual 的 Screen/Widget 职责](https://github.com/Textualize/frogmouth/tree/main/frogmouth)、
[Toolong 的进程/查看器分工](https://github.com/Textualize/toolong/tree/main/src/toolong) 和
[Textual Worker 生命周期讨论](https://github.com/Textualize/textual/issues/2593)。
只借鉴独立生命周期、数据流和展示职责，不复制源代码，也不依赖新框架。

实验选择保留目录，命令准备统一使用 `results.layout.resolve_result_file` 的单实验解析；多任务目录需显式选择。`results/overview.py` 只读取目录元数据，`results/data_status.py` 在后台让质量审计与 `inspect_profile_result` 借用一次结果快照，不调用统计聚合。两个 Worker 复用 App 的管理器、取消令牌和共享只读测量锁，返回时检查当前选择身份。HTML 来源校验只在后台进行，旧报告没有来源证据时保留 unverified；补采在用户确认后再次检查来源及互斥状态。

线程取消与 UI 回调核验参考 [Textual Workers 指南及示例](https://github.com/Textualize/textual/blob/main/docs/guide/workers.md)。复用已锁定的 Textual，不增加依赖；同一取消令牌传递给现有结果解析器，避免仅取消 Worker 而遗留正在执行的读取。
