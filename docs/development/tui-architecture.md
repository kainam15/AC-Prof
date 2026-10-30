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
├── run/                    # 执行、预检、恢复、进度与命令呈现
│   ├── execution.py        # 唯一执行会话/身份和进程所有者
│   ├── process.py          # 进程/进程组启动、停止和回收
│   ├── process_controller.py # 工作线程读取、事件、延迟回收
│   ├── presenter.py        # 当前运行信息的 Textual 呈现，无执行所有权
│   └── results.py          # 只读归属检查与纯完成判定
├── settings/               # 表单配置、语言和本地首选项
└── ui/
    ├── dialogs.py          # 对话框
    ├── pages/              # Run / Monitor / Results(在 results/) / Images / Settings
    └── ...                 # 公共控件、布局及主题
```

## 所有权与依赖

- `ExecutionCoordinator` 是当前执行尝试的唯一拥有者。它持有 `ExecutionSession`（执行种类、命令、Token、Snapshot、停止标记与归属结果）和唯一的 `ProcessLifecycle`。新尝试必须取得新 Token，旧结果和取消回调不能进入新会话。
- `ProcessLifecycle` 只能在目标进程确认退出后释放。清理超时保留进程占用状态，禁止新实验开始；停止必须携带期望的进程对象，防止迟到的 SIGINT 误伤替换进程。
- `run/process_controller.py` 使用唯一 App Worker Manager 调度日志读取和失败后的回收。Textual 窗口卸载不等于采集进程结束；UI 回调必须在 App 存活且 Token 匹配时投递。
- `RunPresenter` 仅修改现有控件、焦点、通知和计时器。它通过 App 的明确接口读取 `ExecutionCoordinator`，不缓存执行身份，不创建第二个 Worker Manager，也不修改测量协议。
- `AcprofTui` 仅是 Textual 宿主：事件路由、组合页面、全局互斥、Worker/Timer 和卸载回收桥接。少量公共 callback 方法委派给 `RunPresenter`，供 Textual 和线程安全回调使用；这些委派不是独立状态副本。
- `commands.py`、`i18n.py` 和 `acprof.tui` 包导入不能提前加载 Textual。重依赖仍在使用时才导入。

## 行为不变量

1. 正式测量开始/结束由协议进度决定；输出解析中的 `measurement_active` 门控不得改变。下载进度、普通日志、动画、镜像轮询不能进入正式测量窗口。
2. `ProcessLifecycle.release` 只有匹配且确认退出的进程才能成功；归属判定不能把历史 CSV 误当作当前运行结果。
3. `ExecutionCoordinator.finish` 只允许匹配的执行种类、已回收的进程。成功/取消/失败的 UI 状态均源于纯 `evaluate_process_completion`。
4. CLI 进程仍是实际采集者；TUI 只是其外部协调器，不新增 TUI 到正式测量路径的工作。
5. Textual Worker 归属仍由 App 宿主管理；不因页面切换失去进程所有权。

## 验收与设计来源

- `uv run pytest tests/tui/workflows/test_tui_execution_coordinator.py tests/tui/workflows/test_tui_process_lifecycle.py tests/tui/workflows/test_tui_process_completion.py`：所有权、异常与完成证据。
- `tests/tui/`：交互/导航/布局/日志/设置；`tests/visual/`：SVG；`tests/visual_png/`：真实终端 PNG 验证（可选）。
- `uv run ruff check acprof/tui`、`uv run mypy acprof/tui`，以及 CLI 惰性导入与非 TUI 入口验证。

参考 [Textual 的 Screen/Widget 职责](https://github.com/Textualize/frogmouth/tree/main/frogmouth)、
[Toolong 的进程/查看器分工](https://github.com/Textualize/toolong/tree/main/src/toolong) 和
[Textual Worker 生命周期讨论](https://github.com/Textualize/textual/issues/2593)。
只借鉴独立生命周期、数据流和展示职责，不复制源代码，也不依赖新框架。
