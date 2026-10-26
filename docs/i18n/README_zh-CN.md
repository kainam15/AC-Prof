<p align="center">
  <img alt="AC-Prof" src="../assets/logos/acprof-logo-light.svg#gh-light-mode-only" width="55%">
  <img alt="AC-Prof" src="../assets/logos/acprof-logo-dark.svg#gh-dark-mode-only" width="55%">
</p>

[English](../../README.md) · **简体中文**

AC-Prof 用于分析 AI 模型在不同 CPU、内存、GPU 配置和输入规模下的推理表现。它自动准备 Docker 或隔离的 Process 运行环境、复用固定版本的模型权重，并生成可复现的测量和分析产物。兼容的模型无需修改模型代码即可采集。

## AC-Prof 能做什么

| 能力 | 说明 |
| --- | --- |
| 性能与资源 | 应用层延迟、吞吐量、CPU 和内存占用，以及可用时的 GPU 指标 |
| 完整画像 | 能耗、抓包计时、CPU 硬件计数器，以及按需启用的独立 Profiler |
| 可复现实验 | 输入与环境元数据、分层 CSV 结果、图表和报告 |

支持文本、视觉、音频、时间序列、Diffusion、多模态与结构化数据任务，但模型仍需满足[任务接口要求](../models/supported-tasks.md#任务支持范围)，并非所有同类 checkpoint 都保证可运行。

## 快速开始

使用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 从 [PyPI](https://pypi.org/project/acprof/) 安装最新正式版本，启动交互式终端界面：

```bash
uv tool install acprof
acprof --version
acprof tui
```

通过 PyPI 安装的用户可使用以下命令升级：

```bash
uv tool upgrade acprof
```

如需从 GitHub Release 下载 Wheel 或 Linux standalone，可查看[最新正式版](https://github.com/kainam15/AC-Prof/releases/latest)，并按[安装说明](../getting_started/installation.md#github-release-fallback)校验文件哈希。

Native Linux x86_64、本机 Docker 和 cgroup v2 仍是 FULL 测量的基准。当前源码也提供无需 Docker 或 root 的 Linux Process basic 采集。Process CPU/RSS、继承限制和共享设备 GPU 指标保留独立证据及比较规则；云 GPU 环境仍需对应平台实测。详见 [Linux Runtime](../platforms/linux-runtimes.md)和[主机要求](../getting_started/installation.md#1-检查主机环境)。

### 开发者快速开始

参与开发或使用最新源码时，克隆仓库，由 uv 准备 editable 项目和开发工具：

```bash
git clone https://github.com/kainam15/AC-Prof.git
cd AC-Prof
uv sync
uv run acprof tui
```

`uv sync` 自动创建和同步当前 checkout 的 `.venv`，无需手动激活。每个 Git worktree 使用独立环境，并共享 uv 下载缓存。普通发行用户仍使用 `uv tool install acprof` 和 `acprof tui`。

完整流程见[开发环境](../development/distribution.md#本地开发环境)和[测试指南](../development/testing.md)。

Docker Runtime 默认复用通过校验的模型镜像，缺失时自动构建。需要单次重建时，使用 `--rebuild-image` 或 TUI 的“镜像高级操作”；仍可使用 Docker 缓存。Process 在 TUI 中选择 basic 和继承 CPU/内存限制。详见[镜像准备](../models/images.md#构建复用和验证)和 [Runtime 选择](../platforms/linux-runtimes.md#选择与准备)。

TUI 通过结构化事件显示采集与探测进度。采集进程退出后，会核对已保存的实验身份并审计结果，再判断是否成功；仅有正常退出码不足以证明采集成功。

监控页支持日志级别配色、普通文字搜索、详细视图，以及复制或导出保留原文。准备阶段与下载状态使用颜色区分，独立进度条和对齐的指标区分缓存数据与 SDK 网络 payload。保留上限和测量期间的行为见[日志与下载进度](../usage/tui-experiments.md#日志查看与复制)。

结果目录通过校验并初始化后，每次准备都会保留经脱敏的主进程 Python 日志和结构化阶段／失败摘要。恢复保留历史尝试，界面显示前的两个日志队列也有明确容量上限。保存范围和失败处理见[准备阶段诊断](../profiling/artifacts.md#准备阶段诊断)。

下载确认按概览、运行环境与来源、存储空间、网络预检分区展示，所有信息直接展开。确认前会进行总计至多 256 KiB 的限量模型 GET 探测，计入下载预算；零预算跳过 GET。完整下载只有确认后才开始。预检通过不保证完整传输成功，详见[下载与磁盘预检](../usage/tui-experiments.md#下载与磁盘预检)。

模型开始推理前，完整 CSV 表头与结果目录已经存在。每个完成的测量窗口同步追加到根目录模块 CSV，实验完成时冻结不可变快照。恢复和后处理期间，使用 AC-Prof 读取器获取一致的已提交数据。详见[实时结果与恢复](../profiling/artifacts.md#实时提交与后处理)。

### 生成交互式报告

运行 `acprof tui` → **统计报告** → **选择实验** → **生成交互式报告**。页面会显示完整输出路径、窗口数量与数据质量限制。本地图形桌面可点击 **打开 HTML**，也可展开 **导出 / 远程获取** → **导出 HTML**，复制为新文件。每次生成保留旧报告；实验结束不会自动生成 HTML。

HTML 单文件包含 **Comparison Matrix**、**Pareto / Trade-off** 和 **Scaling**，无需网络或 Web 服务。在 dor-server 等 SSH 服务器上，用 SFTP 或远程 IDE 下载页面显示的文件，再用本机浏览器打开。多个输入和输出目录见 [TUI 报告流程](../usage/tui-pages.md#交互式-html-报告)。

CLI 仍可独立使用：

```bash
acprof report results/run-a/ results/run-b/ --output comparison.html
```

## 文档导航

| 想做什么 | 阅读入口 |
| --- | --- |
| 准备主机、跑通首个 CPU 实验或查看结果 | [安装与运行](../getting_started/installation.md) |
| 了解终端界面 | [TUI 用户指南](../usage/tui.md) |
| 使用命令行选项与工具 | [CLI 参考](../usage/cli.md) |
| 理解指标、分层 CSV 和报告 | [指标与结果分析](../results/metrics.md) |
| 了解支持的模型与运行环境 | [运行兼容](../models/runtime.md) |
| 开发或贡献 AC-Prof | [开发安装](../development/distribution.md#本地开发环境)、[代码架构](../development/architecture.md)、[测试指南](../development/testing.md) |

[完整文档索引](../README.md) · [运行排障](../usage/troubleshooting.md)

## 贡献与安全

参与开发请参阅 [CONTRIBUTING.md](../../CONTRIBUTING.md)；漏洞报告与第三方模型代码的信任边界请参阅 [SECURITY.md](../../SECURITY.md)。

## 许可与来源

项目代码采用 [Apache-2.0](../../LICENSE)。内置 LibriSpeech 音频保留 [CC-BY-4.0](../../licenses/CC-BY-4.0.txt)，模型代码和权重遵循各自许可。AC-Prof 属于 JNU DISTINT 的 DOR 项目；贡献记录及来源见 [NOTICE](../../NOTICE) 与[项目来源](../development/history.md)。
