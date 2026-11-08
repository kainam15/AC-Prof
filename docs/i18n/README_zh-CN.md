<p align="center">
  <img alt="AC-Prof" src="../assets/logos/acprof-logo-light.svg#gh-light-mode-only" width="55%">
  <img alt="AC-Prof" src="../assets/logos/acprof-logo-dark.svg#gh-dark-mode-only" width="55%">
</p>

[English](../../README.md) · **简体中文**

AC-Prof 用于分析 AI 模型在不同 CPU、内存、GPU 配置和输入规模下的推理表现。
它自动准备 Docker 或隔离的 Process 运行环境，复用固定版本的模型权重，
为兼容模型生成可复现的测量结果，无需修改模型代码。

## 功能概览

| 领域 | 可获得的结果 |
| --- | --- |
| 性能与资源 | 应用层延迟、吞吐量、CPU/内存占用及可用的 GPU 指标 |
| 完整画像 | 能耗、抓包计时、CPU 硬件计数器和按需启用的独立 Profiler |
| 可复现分析 | 输入与环境元数据、分层 CSV、图表和离线 HTML 报告 |

支持文本、视觉、音频、时间序列、Diffusion、多模态与结构化数据任务。
实际模型必须满足[任务接口要求](../models/supported-tasks.md#任务支持范围)，
并非所有同类 checkpoint 都保证可运行。

## 快速开始

使用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 安装 [PyPI 正式版](https://pypi.org/project/acprof/)：

```bash
uv tool install acprof
acprof --version
acprof tui
```

升级已安装的版本可运行 `uv tool upgrade acprof`。
如要主动体验 **预览版**（Alpha/Beta/RC），执行
`uv tool install acprof --prerelease allow`（已有安装时加 `--force`）。
后续直接使用 `uv tool upgrade acprof`，uv 会保留预发布选项。
要恢复仅选择正式版，执行 `uv tool install --force acprof --prerelease disallow`。
预览版只运行轻量发行验证，不能替代正式硬件测量验收；
详见[预览版安装说明](../getting_started/installation.md)。

主机准备、GitHub Release 文件及第一个实验的操作详见
[安装指南](../getting_started/installation.md)与[实验指南](../usage/experiments.md)。

## 采集与分析

- **实验采集：**通过 [TUI](../usage/tui.md) 或 [CLI](../usage/cli.md) 运行资源与输入矩阵；
  采集后按[指标和结果目录](../results/metrics.md)查看产物。
- **离线报告：**在 [HTML 报告](../results/reports.md#交互式配置比较报告)中查看配置比较、窗口统计、图表与导出；
  独立运行比较和置信区间另有[有效性条件](../results/comparisons.md#跨独立实验比较)。
- **流式服务：**使用 `acprof load` 测量已经运行的 OpenAI-compatible 服务；
  并发/到达率扫描、按顺序运行 JSONL 请求集及可选 vLLM/SGLang 服务观测详见
  [流式测量指南](../profiling/streaming.md)。

## 运行环境

原生 Linux x86_64 配合本机 Docker 可使用 `basic`，硬件与权限满足要求时可使用 `full`。
Linux 虚拟机/云环境和无 Docker 的 Process 环境支持受限的 `basic` 采集；
WSL2 也仅提供有限的 `basic` 支持，Windows/macOS 原生环境暂不支持采集。
实际可用指标取决于运行环境与权限，不能只依据模式名称判断。
详见 [环境支持矩阵](../platforms/linux-runtimes.md#支持环境与采集模式)、[WSL2](../platforms/wsl2.md)及
[主机要求](../getting_started/installation.md#1-检查主机环境)。

## 文档与开发

完整的安装、使用、模型、采集协议、结果分析、平台限制和开发文档，
统一从[文档索引](../README.md)进入。

从源码参与开发：

```bash
git clone https://github.com/kainam15/AC-Prof.git
cd AC-Prof
uv sync
uv run acprof tui
```

开发环境与验证方法详见[开发环境](../development/distribution.md#本地开发环境)和
[测试指南](../development/testing.md)。

## 贡献与安全

参与开发参阅 [CONTRIBUTING.md](../../CONTRIBUTING.md)；
漏洞报告与第三方模型代码信任边界参阅 [SECURITY.md](../../SECURITY.md)。

## 许可与来源

项目代码采用 [Apache-2.0](../../LICENSE)，内置 LibriSpeech 音频保留
[CC-BY-4.0](../../licenses/CC-BY-4.0.txt)。第三方模型遵守各自许可。
AC-Prof 属于 JNU DISTINT 的 DOR 项目；详见 [NOTICE](../../NOTICE) 与
[项目来源](../development/history.md)。
