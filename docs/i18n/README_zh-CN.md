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

支持文本、视觉、音频、时间序列、Diffusion、多模态与结构化数据任务，但模型仍需满足[任务接口要求](../models/runtime.md#任务支持范围)，并非所有同类 checkpoint 都保证可运行。

## 快速开始

使用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 安装最新发布版本，启动交互式终端界面：

```bash
uv tool install acprof
acprof tui
```

实验采集需要满足要求的 Linux 主机、本机 Docker Engine 和 cgroup v2。Native Linux x86_64 支持符合前置条件的 FULL 测量；WSL2 仅有受限的 basic/PARTIAL 支持。详见[主机要求](../getting_started/installation.md#1-检查主机环境)。

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

## 许可与来源

项目代码采用 [Apache-2.0](../../LICENSE)。内置 LibriSpeech 音频保留 [CC-BY-4.0](../../licenses/CC-BY-4.0.txt)，模型代码和权重遵循各自许可。AC-Prof 属于 JNU DISTINT 的 DOR 项目；贡献记录及来源见 [NOTICE](../../NOTICE) 与[项目来源](../development/history.md)。
