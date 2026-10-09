# AC-Prof 代码架构

AC-Prof 的命令入口负责参数和调度，业务模块按输入规划、运行时采集、结果分析与界面组织。
唯一公开入口是 `acprof <command>`；Python 调用直接引用职责所属模块。

`acprof/platform.py` 是环境识别和平台能力策略的唯一入口，保持标准库依赖；
`capabilities.py` 分开维护平台 support 与采集 evidence。`host/platform_metadata.py` 只在准备阶段
采集版本信息，analysis/plotting 只读取保存的环境身份，不探测当前主机来解释历史结果。
WSL2 PARTIAL 与 Native Linux FULL 的边界见 [WSL2](../platforms/wsl2.md)。

源码开发、editable、wheel 和 standalone 均通过 `acprof.cli.main` 惰性分发 `acprof <command>`。
`installation.py` 区分只读构建资源和用户工作目录，并生成 Python/standalone 子进程命令。
资源、安装与发布边界见[发行包说明](distribution.md)。

修改模块边界、依赖方向或兼容入口时按下列专题查阅。开发安装与操作说明见 [开发安装](distribution.md#本地开发环境)和[测试指南](quality.md#开发质量检查)，
字段与测量口径见 [指标与结果分析](../results/fields.md#分层指标字段解释完整宽表导出)，运行环境扩展见[模型运行环境与适配器](../models/runtime.md)。

## 专题索引

- [目录职责与依赖边界](modules.md)：文件规模、模块地图、入口和兼容边界。
- [主机编排与测量模块](orchestration.md)：Host、运行时、测量和 diagnostics。
- [结果分析与 TUI 模块](consumers.md)：分析/补采与 TUI 控制器的依赖。
