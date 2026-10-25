<p align="center">
  <img alt="AC-Prof" src="../assets/logos/acprof-logo-light.svg#gh-light-mode-only" width="55%">
  <img alt="AC-Prof" src="../assets/logos/acprof-logo-dark.svg#gh-dark-mode-only" width="55%">
</p>

[English](../../README.md) · **简体中文**

AC-Prof 用于分析容器化 AI 模型在不同 CPU、内存、GPU 配置和输入规模下的推理表现。它自动准备 Docker 运行环境、只读挂载共享模型权重，并生成可复现的测量和分析产物。兼容的模型无需修改模型代码即可采集。

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

实验采集需要满足要求的 Linux 主机、本机 Docker Engine 和 cgroup v2。Native Linux x86_64 支持符合前置条件的 FULL 测量；WSL2 仅有受限的 basic/PARTIAL 支持。详见[主机要求](../getting_started/installation.md#1-检查主机环境)。

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

当前源码默认复用通过校验的模型镜像，缺失时自动构建。需要单次重建时，使用 `--rebuild-image` 或 TUI 的“镜像高级操作”；仍可使用 Docker 缓存。详见[镜像准备](../models/images.md#构建复用和验证)。

TUI 通过结构化事件显示采集与探测进度。采集进程退出后，会核对已保存的实验身份并审计结果，再判断是否成功；仅有正常退出码不足以证明采集成功。

监控页支持日志级别配色、普通文字搜索、详细视图，以及复制或导出保留原文。模型下载使用独立的字节进度区域，区分缓存数据与 SDK 网络 payload。保留上限和测量期间的行为见[日志与下载进度](../usage/tui-experiments.md#日志查看与复制)。

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
