# TUI 实现约束

- 架构以[运行会话和模块所有权](../../docs/development/tui-architecture.md)为准。`AcprofTui` 是 Textual 事件、导航、Widget/Worker 宿主，不强制承担采集状态及业务方法。旧版“`app.py` 必须拥有进程/测量生命周期”的限制已被独立 `run/execution.py` 替代。
- `run/execution.py` 的 `ExecutionCoordinator` 唯一持有执行会话与 `ProcessLifecycle`。启动、停止、结果和迟到回调必须检查执行 Token/进程身份；未退出的子进程保持占用，不能释放或重启。不得存在第二个采集进程所有者、Worker Manager 或状态副本。
- `run/presenter.py` 负责运行期 UI 副作用，`run/process_controller.py` 负责输出读取与跨线程事件；业务模块按 `experiments/`、`models/`、`images/`、`results/`、`logging/`、`settings/` 管理，通用组件归 `ui/`。页面可拆分，但不可将正式采集绑定到可卸载的页面 Worker。
- `commands.py`、`i18n.py` 和包初始化不能提前导入 Textual；`app.py` 在启动 TUI 时加载 Textual。新文案经 `i18n.py` 管理，语言切换不改变原始日志、CLI 参数和进度协议。
- 所有 UI 工作者复用 App 的 Worker Manager；保留 group、thread、exclusive、取消和退出行为。计时器、日志刷新、下载/镜像轮询必须遵守 `ProgressSnapshot.measurement_active`，不得在测量窗口中新增工作。
- 修改进程与生命周期时，验证“停止请求误伤替换进程”“忽略信号后继续拥有未回收进程”“延迟回调”“取消/失败/正确产物归属”等边界；交互及终端尺寸验证见[测试指南](../../docs/development/terminal-validation.md#tui-与终端证据)和[TUI 回归 Skill](../../.agents/skills/acprof-textual-regression/SKILL.md)。
