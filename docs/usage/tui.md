# 终端界面使用指南

TUI 提供实验配置、运行监控、绘图、统计、补采、镜像管理和设置。
先完成[安装与主机准备](../getting_started/installation.md#快速开始)，再启动界面。
设置文件格式与颜色参数见 [CLI 与设置](configuration.md#tui-本地设置)。

顶栏始终显示 `Native Linux / FULL` 或 `WSL2 / PARTIAL`（未支持环境显示 unknown）。
采集确认页列出会采、语义降级与缺失指标；WSL2 的 full 或未开放 profiler 请求会在启动前拒绝。
平台策略见 [WSL2](../platforms/wsl2.md)，这些界面显示不进入正式测量窗口。

[文档导航](../README.md)

## 专题索引

- [TUI 页面与功能](tui-pages.md)：启动、实验页面、统计和镜像。
- [TUI 交互与输入](tui-interaction.md)：快捷键、环境检查、输入确认及界面偏好。
- [TUI 配置与测量行为](tui-experiments.md)：配置记忆、日志、运行中行为和磁盘预检。
