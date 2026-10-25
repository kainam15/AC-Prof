# 终端界面使用指南

TUI 提供实验配置、运行监控、绘图、统计、补采、镜像管理和设置。
在“统计报告”页选择实验后可生成、打开和导出离线 HTML，完整步骤见[交互式 HTML 报告](tui-pages.md#交互式-html-报告)。
先完成[安装与主机准备](../getting_started/installation.md#快速开始)，再启动界面。
设置文件格式与颜色参数见 [CLI 与设置](configuration.md#tui-本地设置)。

顶栏显示当前 Linux、VM、容器或 WSL2 环境及能力等级；完整 Native 硬件条件仍需预检确认。
采集确认页列出会采、语义降级与缺失指标；WSL2 的 full 或未开放 profiler 请求会在启动前拒绝。
平台策略见 [WSL2](../platforms/wsl2.md)，这些界面显示不进入正式测量窗口。

无 Docker 环境选择 Smoke/basic，在高级参数选择 Process 并点击“使用 Process 继承限制”。
预检显示 Runtime 选择、可用采集器和权限缺项；计划、结果及统计保留继承限制的标识。
完整操作及已验证/未验证边界见 [Linux Runtime](../platforms/linux-runtimes.md)。

[文档导航](../README.md)

## 专题索引

- [TUI 页面与功能](tui-pages.md)：启动、实验页面、恢复与重试、统计和镜像。
- [TUI 交互与输入](tui-interaction.md)：快捷键、环境检查、输入确认及界面偏好。
- [TUI 配置与测量行为](tui-experiments.md)：配置记忆、日志、运行中行为和磁盘预检。
