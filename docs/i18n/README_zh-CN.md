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

[离线 HTML 报告](../results/reports.md#交互式配置比较报告)是统一分析入口。一份 `result_all.csv` 即可比较 CPU、内存和 GPU 配置并查看实验内窗口统计；显式独立运行分组复用原有身份与条件检查，每侧至少三次有效独立运行才估计 Bootstrap 置信区间。多个模型可以并列展示，并提示任务与输入的可比性限制。Python 定义统计口径、核验有效窗口并生成初始结果或复用匹配的统计 JSON；离线浏览器可按该口径重新计算，记录参数与来源。历史 CSV 无需补造运行身份或重新采集。实验完成后（含 `--resume`）也会在 `plots/analysis/` 生成新报告。

打开“窗口统计 → 统计参数”，可多选指标、置信水平（90%、95%、99%）和 Bootstrap 次数（1000、5000、10000）；Seed 与窗口块大小放在高级设置中。重新计算、取消、恢复默认参数或恢复初始结果均无需返回 TUI。“添加实验”支持本地 CSV、AC-Prof HTML 或分析 JSON。图表与 CSV / JSON 导出使用当前结果和筛选，另存 HTML 保留当前会话。只有 CSV 的导入保持可比性未知，不生成置信区间。超出 8 MiB 内嵌窗口数据预算的报告提供带哈希校验的 JSON 文件供离线导入，详见[统计参数与离线重算](../results/reports.md#html-统计参数与离线重算)。

```bash
acprof report result_all.csv --output resources.html
acprof report model-a.csv model-b.csv --output models.html
acprof compare --left run-a/ --right run-b/ --html-output comparison.html \
  --output comparison.json --csv-output comparison.csv
```

重复 `--left` 和 `--right` 可添加独立重复实验。每侧只有一次运行时展示描述性差值与比值，不生成运行级区间。TUI 选择实验目录后自动加载元数据概览和 Profiling 状态；“实验结果 → 分析报告”生成并打开 HTML，同时核对报告与当前数据的来源。补采仍需主动选择并确认；统计参数、图表、比较与导出在 HTML 中完成。CLI 保留高级统计参数；旧 TUI 统计、比较、导入导出和 PNG 控件保持隐藏，后端能力保留。

`acprof load --base-url` 可测量已启动的 OpenAI-compatible 流式服务。显式并发数、到达率列表支持 `--dry-run` 计划预览，每个负载点分别保留报告；详见[多负载点编排](../profiling/streaming.md#多负载点编排)。`acprof report <sweep.json>` 可生成[离线吞吐与时延曲线](../results/reports.md#多负载点流式容量报告)，保留条件分组、失败点和缺失证据。可选的 [vLLM／SGLang 身份探测](../profiling/streaming.md#服务身份)分别保留服务观测与用户声明。显式选择 `vllm-config` 可从已开放的开发端点读取部分引擎配置，SGLang 也记录部分启动配置。

`acprof load --request-jsonl` 支持按[固定顺序运行本地请求集](../profiling/streaming.md#本地请求集)，使用不同提示词和生成参数。请求集报告保留输入顺序与逐请求映射，供离线分析使用。

支持文本、视觉、音频、时间序列、Diffusion、多模态与结构化数据任务，但模型仍需满足[任务接口要求](../models/supported-tasks.md#任务支持范围)，并非所有同类 checkpoint 都保证可运行。

可选 `--server-metrics vllm` 在负载前后读取[服务端阶段直方图](../profiling/streaming.md#服务端阶段统计)。报告展示服务级 Prefill／Decode 统计与缺失证据，不能当作本次客户端请求或 GPU 的精确计时。

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

在**运行监控 → 工具 → 导出**中，日志默认按当前级别与关键词筛选导出。保存窗口也提供**导出全部保留日志**，并预览各范围的原始行数。导出内容以打开窗口时为准，保留原始行与顺序，不加入显示换行。详见[日志查看与导出](../usage/tui-experiments.md#日志查看与复制)。

通过 PyPI 安装的用户可使用以下命令升级：

```bash
uv tool upgrade acprof
```

TUI 启动时检查一次 PyPI，发现新版后显示当前版本、最新版本及升级命令。**应用设置 → 版本更新**保留检查结果，包括网络失败原因；详见[启动行为](../usage/tui-pages.md#启动和页面)。

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
