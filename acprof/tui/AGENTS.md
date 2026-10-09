# TUI 实现约束

- `app.py` 保留状态、进程启动/停止/回收、正式测量边界和生命周期；`configuration_actions`、`experiment_actions`、`preflight_actions`、`result_actions`、`profile_actions`、`slash_actions` 负责 UI 输入及只读流程，并从 `AcprofTui` 通过 Textual MRO 调用。不要复制 App 状态或偷偷启动第二个测量拥有者。
- 页面、命令、进度、设置按[架构分工](../../docs/development/consumers.md#tui-与兼容维护)维护，调用方直接导入 `acprof.tui` 下的所属模块，不恢复旧 `acprof.cli.tui_*` 导出。
- 新文案经 `i18n.py` 管理；语言切换不改变原始日志、CLI 参数和进度协议。设置变更遵循[持久化契约](../../docs/usage/configuration.md#tui-本地设置)。
- 涉及定时器、日志刷新或动画时核对 `ProgressSnapshot.measurement_active` 的进入、退出和失败恢复，不绕过已有抑制机制。
- 交互测试使用临时设置并模拟采集边界；终端尺寸和真实焦点/按键验证见[测试指南](../../docs/development/terminal-validation.md#tui-与终端证据)，执行流程见[TUI 回归 Skill](../../.agents/skills/acprof-textual-regression/SKILL.md)。
