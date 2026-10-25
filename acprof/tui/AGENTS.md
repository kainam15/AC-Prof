# TUI 实现约束

- `app.py` 保留 Textual 事件/快捷键入口、页面导航、全局互斥、进程启动/停止/回收和正式测量生命周期；业务使用具体类型的 `*_controller` 对象组合，各自初始化并拥有配置、镜像、预检、结果等状态。跨职责调用写明 controller 接收方，不恢复业务 Mixin、App 私有方法代理、巨型 Protocol 或类型断言。所有 worker 经 App 的同一个 worker manager 调度；不要复制进程状态或启动第二个测量拥有者。
- 页面、命令、进度、设置按[架构分工](../../docs/development/consumers.md#tui-与兼容维护)维护，调用方直接导入 `acprof.tui` 下的所属模块，不恢复旧 `acprof.cli.tui_*` 导出。
- 新文案经 `i18n.py` 管理；语言切换不改变原始日志、CLI 参数和进度协议。设置变更遵循[持久化契约](../../docs/usage/configuration.md#tui-本地设置)。
- 涉及定时器、日志刷新或动画时核对 `ProgressSnapshot.measurement_active` 的进入、退出和失败恢复，不绕过已有抑制机制。
- 交互测试使用临时设置并模拟采集边界；终端尺寸和真实焦点/按键验证见[测试指南](../../docs/development/terminal-validation.md#tui-与终端证据)，执行流程见[TUI 回归 Skill](../../.agents/skills/acprof-textual-regression/SKILL.md)。
