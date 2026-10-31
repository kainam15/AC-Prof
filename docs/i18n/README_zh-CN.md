<p align="center">
  <img alt="AC-Prof" src="../assets/logos/acprof-logo-light.svg#gh-light-mode-only" width="55%">
  <img alt="AC-Prof" src="../assets/logos/acprof-logo-dark.svg#gh-dark-mode-only" width="55%">
</p>

[English](../../README.md) · **简体中文**

AC-Prof 用于分析 AI 模型在不同 CPU、内存、GPU 配置和输入规模下的推理表现。它自动准备 Docker 或隔离的 Process 运行环境、复用固定版本的模型权重，并生成可复现的测量和分析产物。兼容的模型无需修改模型代码即可采集。

## AC-Prof 能做什么

| 能力       | 说明                                                        |
| ---------- | ----------------------------------------------------------- |
| 性能与资源 | 应用层延迟、吞吐量、CPU 和内存占用，以及可用时的 GPU 指标   |
| 完整画像   | 能耗、抓包计时、CPU 硬件计数器，以及按需启用的独立 Profiler |
| 可复现实验 | 输入与环境元数据、分层 CSV 结果、图表和报告                 |

支持文本、视觉、音频、时间序列、Diffusion、多模态与结构化数据任务，但模型仍需满足[任务接口要求](../models/supported-tasks.md#任务支持范围)，并非所有同类 checkpoint 都保证可运行。

## 采集环境与指标

| 运行环境                                       |      支持的采集模式      | 可采集项目                                                                                                       | 主要限制                                                                                   |
| ---------------------------------------------- | :-----------------------: | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 原生 Linux x86_64 + 本机 Docker                | `basic`、<br />`full` | `basic`：延迟、吞吐、容器 CPU/内存及可用 GPU 指标；<br />`full` ：额外支持能耗、PMU、抓包计时及可选 Profiler | `full` 需要 cgroup v2、RAPL、perf 和抓包权限                                             |
| Linux 虚拟机/云主机 + 本机 Docker              |         `basic`         | 延迟、吞吐、容器 CPU/内存及可用 GPU 指标                                                                         | 环境能力等级仍为 `PARTIAL`；不采集能耗、PMU 或抓包                                       |
| Linux x86_64 + Process（含无 Docker 的云容器） |         `basic`         | 延迟、吞吐、进程树 CPU/RSS 及可用的整卡 GPU 利用率/显存                                                          | 不采集能耗、PMU、抓包或独立 Profiler；数值资源配额需要 cgroup v2 委托                      |
| WSL2 + Docker/Process                          |         `basic`         | 延迟、吞吐、Linux 客体 CPU/内存和部分 GPU 指标                                                                   | 属于 `PARTIAL`；无原生硬件能耗/PMU；Docker 采集需 WSL 内的 Engine，不支持 Docker Desktop |
| Windows/macOS 原生                             |            —            | —                                                                                                               | —                                                                                         |

`basic`/`full` 是采集模式，不等同于环境能力等级 `FULL`/`PARTIAL`。实际指标取决于硬件与权限，独立 Profiler 需显式启用；云 GPU 和 WSL2 仍需对应实例验证。详见 [Linux Runtime](../platforms/linux-runtimes.md)、[WSL2](../platforms/wsl2.md)、[指标说明](../results/metrics.md)与[主机要求](../getting_started/installation.md#1-检查主机环境)。

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

### 开发者快速开始

参与开发或使用最新源码时，克隆仓库，由 uv 准备 editable 项目和开发工具：

```bash
git clone https://github.com/kainam15/AC-Prof.git
cd AC-Prof
uv sync
uv run acprof tui
```

开发环境的配置和测试方法详见[开发环境](../development/distribution.md#本地开发环境)与[测试指南](../development/testing.md)。

## 文档

使用指南、技术参考和开发文档，请参阅[文档索引](../README.md)。

## 贡献与安全

参与开发请参阅 [CONTRIBUTING.md](../../CONTRIBUTING.md)；漏洞报告与第三方模型代码的信任边界请参阅 [SECURITY.md](../../SECURITY.md)。

## 许可与来源

项目代码采用 [Apache-2.0](../../LICENSE)。内置 LibriSpeech 音频保留 [CC-BY-4.0](../../licenses/CC-BY-4.0.txt)，模型代码和权重遵循各自许可。AC-Prof 属于 JNU DISTINT 的 DOR 项目；贡献记录及来源见 [NOTICE](../../NOTICE) 与[项目来源](../development/history.md)。
