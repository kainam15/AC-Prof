<h1 align="center">AC-Prof</h1>

[English](../../README.md) · **简体中文**

AC-Prof 用来比较 Hugging Face 模型在不同 CPU、内存、GPU 配置和输入规模下的推理表现。
你提供一个模型 ID，它会准备 Docker runtime，通过只读挂载共享主机 Model Store 中的权重，运行实验，并保存延迟、能耗、资源占用等数据。
支持的模型无需修改代码；结果包含 CSV、复现所需的元数据和可生成的图表。

[快速开始](#快速开始) · [终端界面](#交互式终端界面) · [查看结果](#查看结果) · [完整文档](../README.md)

## AC-Prof 会采集什么

| 使用方式 | 可以看到什么 |
| --- | --- |
| `basic`，下面的入门示例 | 应用层延迟、吞吐量、容器 CPU 和内存占用 |
| `full`，命令行默认模式 | 在基础指标上增加能耗、抓包延迟和 CPU 硬件计数器等指标 |
| 显式启用或事后补采 profiler | Torch / NCU FLOP、Massif 内存峰值、Nsight Systems 执行时间线 |

支持范围包括文本、视觉、音频、时间序列、Diffusion、多模态与结构化数据任务。
具体模型需满足对应的[任务接口与运行环境](../Runtime_Compatibility.md#任务支持范围)，任务标签本身不保证任意 checkpoint 都能运行。
符合受限接口的自定义多模态 pipeline 可[自动生成模型契约](../Runtime_Compatibility.md#自动生成模型契约m1m6)；其余接口或未解决依赖可使用[本地模型声明](../Runtime_Compatibility.md#本地模型声明与自定义-pipeline)，通过独立推理验证后再采集。
[指标说明](../Metrics.md#采集能力概览)解释各项数据的含义和测量范围。

## 快速开始

### 1. 准备主机

FULL 采集需要 Native Linux x86_64、本机 Docker Engine 和统一 cgroup v2，推荐 Ubuntu 24.04。
下面的安装脚本会自动准备 uv 和 Python；也可使用无需源码的 [standalone](../Distribution.md#linux-standalone)。
当前用户应能直接运行 `docker info`，并能访问 Hugging Face 及依赖下载源。
WSL2 支持开发及 `--profiling-mode basic` 的 PARTIAL 采集，详见 [WSL2 支持范围](../platforms/wsl2.md)。
Docker Desktop、远程 Docker daemon、Windows 和 macOS 不支持实验采集。

| 平台 | 开发 | 采集 | Native baseline |
| --- | --- | --- | --- |
| Native Linux | 支持 | FULL（仍需满足各 collector 前置条件） | 验证后可纳入 |
| WSL2 | 支持 | PARTIAL / basic，需发行版内本机 Docker 和 cgroup v2 | 不可纳入 |
| Windows / macOS | 离线开发与分析 | 不支持 | 不可纳入 |

每个 dataset 保存环境身份；历史缺字段为 `unknown`，Native 与 WSL 结果不能合并进入同一 baseline。
下面的 CPU 示例不需要 GPU；GPU 实验另需 NVIDIA driver 和 NVIDIA Container Toolkit。

不确定环境是否满足要求时，先看[主机检查与配置](../Getting_Started.md#1-检查主机环境)。
`full` 模式还需要可读的 RAPL、可用的 `perf instructions`、`tcpdump`、`tshark` 和 Docker bridge。

### 2. 安装 AC-Prof 并检查环境

准备好 Git 和 Docker 后执行：

```bash
git clone https://github.com/kainam15/AC-Prof.git
cd AC-Prof
./setup.sh
```

脚本安装当前源码版本，运行 `doctor`，通过后在交互终端打开 TUI。
已预填 BERT、basic CPU、单次请求和新的结果目录；点击“开始采集”并核对确认页即可体验。
Docker 或基础采集条件缺失时会给出处理建议，修复后可重新执行。

只安装和检查、不自动打开界面：

```bash
./setup.sh --no-tui
```

安装完成后，打开新终端，使用统一命令启动 TUI：

```bash
acprof tui
```

源码开发与安装环境的公开入口均为 `acprof <command>`。
使用 `acprof --help` 或 `acprof <command> --help` 查看命令与参数。

需要入门预设时运行 `acprof tui --preset smoke`。当前终端可使用脚本输出的完整路径命令，
详见[安装与命令路径说明](../Distribution.md#clone-后初始化)。
后续可从任意工作目录启动，结果写入该目录；`setup.sh` 启动的工作目录为源码根目录。
模型推理依赖优先复用经过核验的 GHCR 预构建镜像；回退本机构建会明确显示，并须符合来源策略。预算模式下 pull 失败会停止，需重新预检；模型权重按需下载到共享 Model Store。
私有或 gated 模型可在 TUI 按 `F2` → **连接与权限**填写 `HF_TOKEN`；同处可配置代理、通知和采集权限。
连接配置保存到当前工作目录的 `.env.local`（仅当前用户可读写），该文件及其备份应加入 Git 忽略。
默认以 `mirror-only` 连接 `https://hf-mirror.com`，禁用 Xet，镜像失败或未允许的重定向直接停止。使用 `--max-download 5GB` 在大下载前检查预算；大小未知时预算模式也会停止。详见[下载与 Model Store 策略](../Runtime_Compatibility.md#下载网络与-model-store)和[来源配置](../CLI_Reference.md#主机环境与-hugging-face-认证)。
详见[认证配置](../Getting_Started.md#hugging-face-认证)、[开发环境安装](../Getting_Started.md#2-安装-python-依赖)和[发行包说明](../Distribution.md)。

### 3. 跑通第一个 CPU 实验

也可以在命令行运行同样的入门实验：只使用 1 个 CPU、4 GB 容器内存和一个输入规模，主测量发送一次请求。
它用于检查流程能否跑通，单次测量不足以得出性能结论。

```bash
acprof run --model google-bert/bert-base-uncased \
  --profiling-mode basic \
  --cpus 1 --mems 4 --gpus off --input-scales 64 \
  --warmup 0 --repeat 1 --repeat-in-window 1 \
  --compute-profile-tool none --execution-profile-tool none \
  --notify none --output-dir results/first-run
```

首次运行会下载模型和依赖、构建镜像，准备阶段可能较久。程序会先检查环境，再开始下载和实验。
这个 `basic` 示例只采集基础指标；能耗、抓包和独立 profiler 的字段为 `nan` 属于预期结果。
完成后按下一节查看结果。重新做一个实验请换新的 `--output-dir`；中断后在同一条 `acprof run` 命令中保留原参数并添加 `--resume` [恢复实验](../Profiling_Protocol.md#结果完整性与断点续跑)。

也可使用 [`acprof auto MODEL`](../CLI_Reference.md#acprof-auto) 完成权限和主机预检后采集；
资源参数与 `run` 相同。只有显式指定 `--profiling-mode auto` 才按能力选择 full/basic，
语义冲突仍会停止并保存解释。模型覆盖率可通过 [`acprof coverage`](../CLI_Reference.md#acprof-coverage)
冻结样本后单独检查。

正式矩阵默认在独立 startup probe 后按 seed `0` 排序并冻结计划；用 `--matrix-seed` 改变顺序，
或用 `--matrix-order declared` 保持声明顺序。resume 复用冻结计划。full 默认尝试可选 DRAM，
缺失不导致失败；参数与能量单位见 [CLI](../CLI_Reference.md) 和[能耗说明](../Energy_Measurement.md#rapl-topology-与-dram)。

## 查看结果

上面命令行示例的主要文件位于下方目录。通过 `setup.sh` 启动时，输出目录为
`results/first-run-<时间>-<随机后缀>/`，以界面显示的路径替换以下命令中的 `results/first-run`。

```text
results/first-run/google-bert--bert-base-uncased/
├── result_all.csv           # 测量数据
├── static_meta.json         # 模型、镜像和运行环境
├── capability_report.json   # 本次采集的能力与完成证据
├── result_manifest.json     # 布局版本与产物路径索引
├── metadata/                # 输入计划、矩阵计划、解析及补采记录
├── raw/                     # 请求样本与 profiler 原始报告
├── plots/                   # 图表、拟合与窗口统计
├── logs/                    # 终端及独立验证日志
├── debug/                   # 可选 idle 诊断
└── .acprof/                 # 状态、锁、case 工作文件及恢复备份
```

先检查结果是否完整，再生成有适用数据的图表：

```bash
acprof audit results/first-run/google-bert--bert-base-uncased/ --require-complete --require-ok
acprof plot results/first-run/google-bert--bert-base-uncased/result_all.csv
acprof report results/first-run/google-bert--bert-base-uncased/
```

用浏览器打开生成的 `report.html`，通过共用筛选和可选 baseline 查看 Comparison Matrix、
Pareto 取舍及 Scaling 视图。该文件可离线打开，也支持 Windows 浏览器，原始 CSV 保持不变。
已有报告不会被覆盖；使用 `--output another-report.html` 保存新的快照。
聚合口径与历史数据边界见[交互式配置比较](../Metrics.md#交互式配置比较报告)。

新实验的图表写入 `plots/cpu/`、`plots/gpu/`、`plots/gpu+cpu/` 和 `plots/latency_model/`，没有适用数据的部分会跳过。
采集过程中先写 `.acprof/work/cases/<case-id>/result.csv`，矩阵结束后才合并出 `result_all.csv`。
没有清单的旧目录保持原路径，可继续读取、绘图及补采；不自动搬迁数据。
目录协议与恢复边界见[Artifact Layout v2](../Profiling_Protocol.md#artifact-layout-v2)。
正式分析筛选 `status=ok` 且 `warmup=0`；字段、统计与缺失值说明见[结果阅读指南](../Metrics.md#从结果目录开始)。

## 交互式终端界面

正常使用直接点击“开始采集”：系统自动解析和验证模型，只在存在真正的未决选择时询问，确认后继续
同一次采集。运行验证失败可只重试失败阶段；“模型检查”保留为高级诊断工具。详见 [采集准备与模型检查](../TUI.md#模型契约解析与验证)。

也可以通过全屏 TUI 填写参数、查看日志、绘图和管理镜像。完成安装后运行：

```bash
acprof tui --model google-bert/bert-base-uncased --preset smoke
```

没有保存配置的首次启动使用 `smoke`：`basic`、CPU、单次请求和任务对应的最小输入尺度，关闭独立 profiler，初始通知关闭。
切换预设保留下载预算、下载源、缓存与输出路径以及通知选择。开始按钮附近显示实验规模和注明假设的时间估计；已保存配置保持不变。
需要完整指标时，在“高级参数”中改为 `full`，并完成相应的主机检查；开始前可在命令预览中核对参数。
唯一的自定义多模态 pipeline 会尝试自动解析；配套 `AutoModel` 的 `feature-extraction` 提示不会覆盖已声明的任务。固定配置可判定的依赖分支、本地参数转发和主模型自引用会自动处理，动态条件仍保留 review。Ultravox 的固定快照已覆盖 Llama weights 与 Whisper processor 的静态规划，详见[自动模型契约](../Runtime_Compatibility.md#自动生成模型契约m1m6)。输入或依赖仍有缺口时，可在“识别覆盖 → 模型接口声明”填写 JSON；格式与示例见[模型接口声明](../Runtime_Compatibility.md#本地模型声明与自定义-pipeline)。
页面、快捷键、日志复制、设置与 VS Code 按键问题见 [TUI 用户指南](../TUI.md)。

## 运行正式实验

先用最小实验确认环境，再逐步增加输入规模、CPU、内存或 GPU 配置；高开销 profiler 可以在主实验后补采。
使用 `full` 前完成[主机准备](../Getting_Started.md#1-检查主机环境)，并为新实验选择独立输出目录。

仅传 `--model` 会使用默认完整矩阵。自动规划出 6 档输入时，它计划生成 1,344 行（含 warmup），
仅主测量窗口就约 13 小时，下载、构建和 profiler 还需额外时间。详见[时间成本估算](../Profiling_Protocol.md#结果行数和时间成本估算)。

[CPU / GPU 矩阵示例](../Getting_Started.md#运行正式实验) · [先探测最大输入](../Getting_Started.md#先探测最大输入) · [选择与补采 profiler](../Profilers.md) · [企业微信通知](../CLI_Reference.md#企业微信通知)

## 文档导航

| 想继续做什么 | 阅读入口 |
| --- | --- |
| 配置环境，运行 Stable Diffusion、ONNX 或完整矩阵 | [安装与运行指南](../Getting_Started.md) |
| 查参数、输入规模或自定义 workload | [CLI 参考](../CLI_Reference.md) |
| 选择模型、backend 或了解镜像复用 | [运行兼容](../Runtime_Compatibility.md) |
| 理解指标、能耗、图表和统计 | [指标与结果](../Metrics.md)、[能耗测量](../Energy_Measurement.md) |
| 排查环境、OOM、超时或部分结果 | [运行排障](../Troubleshooting.md) |
| 查采集协议和其他专题 | [完整文档索引](../README.md) |

## 项目结构与开发

[`pyproject.toml`](../../pyproject.toml) 声明主机依赖；[`requirements/`](../../requirements/) 集中保存 `host`、`dev` 和 `tui-snapshot` 的输入与锁文件。
模块职责见[代码架构](../Architecture.md)，开发依赖、pre-commit 和测试入口见[测试指南](../Testing.md#开发质量检查)。
新增模型或 backend 参见[适配契约](../Runtime_Compatibility.md#新增一个模型适配)；Agent 协作规则见 [AGENTS.md](../../AGENTS.md)。

## 许可与来源

项目代码采用 [Apache-2.0](../../LICENSE)，延续原始项目定义的许可声明。
AC-Prof 属于 JNU DISTINT 的 DOR 项目，原始贡献者及后续维护来源见 [NOTICE](../../NOTICE)。
原始仓库、当前扩展范围与历史参考边界见[项目来源与演进](../Project_Origin.md)。
内置 LibriSpeech 音频保留 [CC-BY-4.0](../../licenses/CC-BY-4.0.txt)；模型代码和权重遵循各自仓库的许可。

使用 `acprof compare` 比较跨独立实验差异，使用 `acprof load` 执行独立非流式负载。详见[统计口径](../Metrics.md#跨独立实验比较)和[负载协议](../Profiling_Protocol.md#独立非流式负载)。
