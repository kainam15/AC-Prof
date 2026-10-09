<h1 align="center">AC-Prof</h1>

**English** · [简体中文](docs/i18n/README_zh-CN.md)

AC-Prof compares Hugging Face model inference across CPU, memory, and GPU configurations and input sizes.
Given a model ID, it prepares a Docker runtime with weights mounted read-only from a shared host Model Store, runs experiments, and records latency, energy use, and resource utilization.
Supported models require no code changes. Results include CSV measurements and metadata for reproduction, with commands to generate plots.

[Quick start](#quick-start) · [Terminal interface](#interactive-terminal-interface) · [View results](#view-results) · [Documentation](docs/README.md)

The detailed guides linked below are currently in Simplified Chinese.

<a id="ac-prof-会采集什么"></a>

## What AC-Prof measures

| Mode or tool | Measurements |
| --- | --- |
| `basic`, used in the introductory example below | Application latency, throughput, and container CPU and memory usage |
| `full`, the CLI default | Basic metrics plus energy use, packet capture latency, CPU hardware counters, and more |
| Explicitly enabled profilers or post-hoc profiling | Torch / NCU FLOP counts, Massif memory peaks, and Nsight Systems execution timelines |

Supported task families include text, vision, audio, time series, diffusion, multimodal, and structured data.
Each model must meet the relevant [task interface and runtime requirements](docs/Runtime_Compatibility.md#任务支持范围); a task label alone does not guarantee that every checkpoint will run.
Custom multimodal pipelines that fit the supported interfaces can use [automatic model contract generation](docs/Runtime_Compatibility.md#自动生成模型契约m1m6).
Other interfaces or unresolved dependencies can use a [local model specification](docs/Runtime_Compatibility.md#本地模型声明与自定义-pipeline), followed by independent inference validation before measurement.
The [metrics guide](docs/Metrics.md#采集能力概览) explains what each metric means and what it covers.

<a id="快速开始"></a>

## Quick start

Install the latest published release with [uv](https://docs.astral.sh/uv/getting-started/installation/), then launch AC-Prof:

```bash
uv tool install acprof
acprof
```

<a id="1-准备主机"></a>

### 1. Prepare the host

FULL collection requires Native Linux x86_64, a local Docker Engine, and unified cgroup v2. Ubuntu 24.04 is recommended.
AC-Prof requires Python 3.10+; uv can provision Python when needed. A [standalone distribution](docs/Distribution.md#linux-standalone) also includes the interpreter.
Your user account must be able to run `docker info` directly and access Hugging Face and dependency download sources.
WSL2 supports development and PARTIAL collection using `--profiling-mode basic`. See [WSL2 support](docs/platforms/wsl2.md).
Docker Desktop, remote Docker daemons, Windows, and macOS are not supported collection hosts.

| Platform | Development | Collection | Native baseline |
| --- | --- | --- | --- |
| Native Linux | Supported | FULL (subject to collector prerequisites) | Eligible after validation |
| WSL2 | Supported | PARTIAL / basic, local in-distro Docker and cgroup v2 | No |
| Windows / macOS | Offline development and analysis | Unsupported | No |

Environment identity is recorded with every dataset. Missing historical identity is `unknown`; Native and WSL results cannot be merged into one baseline.
The CPU example below does not need a GPU. GPU experiments additionally require an NVIDIA driver and NVIDIA Container Toolkit.

See [host checks and configuration](docs/Getting_Started.md#1-检查主机环境) if you are unsure whether your machine meets the requirements.
The `full` mode also requires readable RAPL counters, working `perf instructions`, `tcpdump`, `tshark`, and a Docker bridge.

<a id="2-安装-ac-prof-并检查环境"></a>

### 2. Install AC-Prof and check the environment

Check the installed version and the host prerequisites:

```bash
acprof --version
acprof --help
acprof doctor --profiling-mode basic --gpus off
```

Run `acprof` or `acprof tui` to open the TUI.
The interface defaults to Simplified Chinese. Press `F2` and select `English` under **界面语言 / Language** to switch languages.
Choose a model and profiling mode, then click **Start run** and review the confirmation screen.
If Docker or another prerequisite is missing, `doctor` reports the missing requirement and a suggested fix.

The public interface is `acprof <command>` in both source and installed environments.
Use `acprof --help` or `acprof <command> --help` to discover commands and options.

Use `acprof tui --preset smoke` to load the small preset; select `basic` for the CPU example below.
You can launch AC-Prof from any working directory; output paths are relative to that directory.
Model runtime dependencies reuse verified GHCR images when available; any local-build fallback is visible and must satisfy the source policy. Budgeted runs stop on pull failure and require a new preflight. Model weights are downloaded as needed into the shared Model Store.
For private or gated models, press `F2` in the TUI and enter `HF_TOKEN` under **Connections and permissions**. The same section configures notifications and profiling permissions.
Connection settings are stored in `.env.local` in the working directory, readable and writable only by the current user. Exclude this file and its backups from Git.
Downloads use `auto`: reuse the Model Store offline, then try the domestic Hub entry and official Hugging Face, following trusted CDN/Xet bridge redirects. A domestic entry does not guarantee domestic storage traffic. AC-Prof uses the system network environment and does not configure VPNs or proxies. If HF remains unavailable, retry after fixing system networking or explicitly choose a ModelScope model; the two sources retain separate artifact identities. Use `--max-download 5GB` to check the budget before bulk downloads; unknown sizes stop budgeted runs. See [download and Model Store policy](docs/Runtime_Compatibility.md#下载网络与-model-store) and [advanced network configuration](docs/CLI_Reference.md#主机环境与-hugging-face-认证).
See [authentication](docs/Getting_Started.md#hugging-face-认证), [installation options](docs/Getting_Started.md#安装), and [distribution details](docs/Distribution.md).

<a id="3-跑通第一个-cpu-实验"></a>

### 3. Run your first CPU experiment

You can run the same introductory experiment from the command line: one CPU, 4 GB of container memory, one input scale, and one request in the main measurement window.
This checks that the workflow runs successfully. A single measurement is not enough to draw performance conclusions.

```bash
acprof run --model google-bert/bert-base-uncased \
  --profiling-mode basic \
  --cpus 1 --mems 4 --gpus off --input-scales 64 \
  --warmup 0 --repeat 1 --repeat-in-window 1 \
  --compute-profile-tool none --execution-profile-tool none \
  --notify none --output-dir results/first-run
```

The first run downloads the model and any required dependencies and builds images, so preparation may take some time.
AC-Prof checks the environment before starting downloads and experiments.
This `basic` example collects only basic metrics; `nan` values in energy, packet capture, and independent profiler fields are expected.
After it finishes, follow the next section to inspect the results. Use a new `--output-dir` for a new experiment.
To [resume an interrupted experiment](docs/Profiling_Protocol.md#结果完整性与断点续跑), add `--resume` to the same `acprof run` command, preserving its parameters.

In the TUI, **Recovery / retry** checks the saved experiment before offering preparation retry or resume. If its sources or environment changed, **New experiment** preserves the original directory and selects a new output path automatically.

You can also use [`acprof auto MODEL`](docs/CLI_Reference.md#acprof-auto) to check permissions and host requirements before measurement;
it accepts the same resource options as `run`. It selects between `full` and `basic` based on host capabilities only when you explicitly pass `--profiling-mode auto`.
Semantic conflicts still stop the run and produce an explanation. Use [`acprof coverage`](docs/CLI_Reference.md#acprof-coverage)
to freeze a sample set and assess model coverage separately.

By default, the experiment matrix runs an independent startup probe, orders cases using seed `0`, and freezes the plan.
Use `--matrix-seed` to change the order or `--matrix-order declared` to retain the declared order. Resumed runs reuse the frozen plan.
The `full` mode attempts optional DRAM energy measurement by default; an unavailable DRAM domain does not fail the run.
See the [CLI reference](docs/CLI_Reference.md) and [energy guide](docs/Energy_Measurement.md#rapl-topology-与-dram) for options and energy units.

<a id="查看结果"></a>

## View results

The command-line example above writes its main artifacts to the directory below.
For a TUI run, replace `results/first-run` in the following commands with the output path shown in the interface.

```text
results/first-run/google-bert--bert-base-uncased/
├── result_all.csv           # Measurements
├── static_meta.json         # Model, image, and runtime environment
├── capability_report.json   # Run capabilities and completion evidence
├── result_manifest.json     # Layout version and artifact path index
├── metadata/                # Input and matrix plans, resolution and post-hoc records
├── raw/                     # Request samples and raw profiler reports
├── plots/                   # Plots, fitted models, and window statistics
├── logs/                    # Terminal and independent validation logs
├── debug/                   # Optional idle diagnostics
└── .acprof/                 # State, locks, case working files, and recovery backups
```

Check result completeness, then generate plots for the available data:

```bash
acprof audit results/first-run/google-bert--bert-base-uncased/ --require-complete --require-ok
acprof plot results/first-run/google-bert--bert-base-uncased/result_all.csv
acprof report results/first-run/google-bert--bert-base-uncased/
```

Open the generated `report.html` in a browser to explore the Comparison Matrix, Pareto trade-offs,
and Scaling views with shared filters and a selectable baseline. The file works offline, including
in a Windows browser, and preserves the original CSV. Existing reports are not overwritten;
use `--output another-report.html` for a new snapshot. See [interactive comparison](docs/Metrics.md#交互式配置比较报告)
for aggregation rules and historical-data limits.

New experiments write plots to `plots/cpu/`, `plots/gpu/`, `plots/gpu+cpu/`, and `plots/latency_model/`, skipping plots without applicable data.
During collection, results are first written to `.acprof/work/cases/<case-id>/result.csv`; `result_all.csv` is merged after the matrix finishes.
Older directories without a manifest retain their existing paths and remain usable for reading, plotting, and post-hoc profiling. Data is not moved automatically.
See [Artifact Layout v2](docs/Profiling_Protocol.md#artifact-layout-v2) for the directory contract and recovery boundaries.
For performance analysis, select rows with `status=ok` and `warmup=0`.
The [results guide](docs/Metrics.md#从结果目录开始) explains fields, statistics, and missing values.

<a id="交互式终端界面"></a>

## Interactive terminal interface

Fill in the experiment settings and select **Start run**. AC-Prof automatically resolves the model, checks
its interface, and validates a minimal request on each selected device. Only unresolved choices require input.
The preparation dialog shows progress or retryable errors; successful validation starts collection automatically.
See [model confirmation and preparation](docs/TUI.md#模型契约解析与验证).

The full-screen TUI lets you configure experiments, view logs, generate plots, and manage images. After installation, run:

```bash
acprof tui --model google-bert/bert-base-uncased --preset smoke
```

The first launch without saved settings uses `smoke`: `basic` mode, CPU execution, one request, and the smallest task-specific input scale, with independent profilers and initial notifications disabled.
Switching presets preserves the model source (Hugging Face or ModelScope), download budgets, cache and output paths, and notification choices. The TUI always uses the automatic download policy, including when loading a legacy policy. The start controls show experiment size and an estimated time with explicit assumptions; switching presets does not rewrite saved configurations.
Download confirmation offers **Confirm download** and **Cancel**. Download, image, and storage sizes automatically use decimal B/KB/MB/GB units; stored data and calculations retain the original bytes. See [download review](docs/TUI.md#下载与磁盘预检).
For the full set of metrics, select `full` under **Advanced** and complete the corresponding host checks.
Review the command preview before starting.

In image management, checking a parent selects its entire descendant branch, including hidden images.
`◩` keeps the parent while descendants are selected. After confirmation, deletion proceeds from descendants to ancestors;
container references or failed deletions keep the affected ancestors. See [image management](docs/TUI.md#镜像管理).

When a model declares a single custom multimodal pipeline, AC-Prof attempts automatic resolution.
A `feature-extraction` hint from the associated `AutoModel` does not override the declared task.
Dependency branches determined by fixed configuration, local parameter forwarding, and references to the main model itself are handled automatically; dynamic conditions still require review.
Static planning for a pinned Ultravox snapshot covers Llama weights and the Whisper processor; see [automatic model contracts](docs/Runtime_Compatibility.md#自动生成模型契约m1m6).
If inputs or dependencies remain unresolved, provide a local JSON specification under **Detection overrides (usually blank) → Model interface spec**.
See [model specifications](docs/Runtime_Compatibility.md#本地模型声明与自定义-pipeline) for the format and examples.
The [TUI user guide](docs/TUI.md) covers pages, shortcuts, log copying, settings, and VS Code key handling.

<a id="运行正式实验"></a>

## Run benchmark experiments

Start with a minimal experiment to check the environment, then increase input sizes, CPU and memory allocations, or GPU configurations.
You can run expensive profilers after the main experiment.
Before using `full`, complete [host preparation](docs/Getting_Started.md#1-检查主机环境) and choose a separate output directory for each new experiment.

Passing only `--model` uses the complete default matrix. When automatic input planning produces six scales, it plans 1,344 rows, including warmup,
with roughly 13 hours in the main measurement windows alone. Downloads, builds, and profilers take additional time.
See [estimating run time](docs/Profiling_Protocol.md#结果行数和时间成本估算).

[CPU / GPU matrix examples](docs/Getting_Started.md#运行正式实验) · [Probe the largest input first](docs/Getting_Started.md#先探测最大输入) · [Choose and run profilers](docs/Profilers.md) · [WeCom notifications](docs/CLI_Reference.md#企业微信通知)

<a id="文档导航"></a>

## Documentation

| Goal | Guide |
| --- | --- |
| Set up the host or run Stable Diffusion, ONNX, or a complete matrix | [Installation and experiments](docs/Getting_Started.md) |
| Look up options, input scales, or custom workloads | [CLI reference](docs/CLI_Reference.md) |
| Choose models and backends or understand image reuse | [Runtime compatibility](docs/Runtime_Compatibility.md) |
| Understand metrics, energy use, plots, and statistics | [Metrics and results](docs/Metrics.md), [Energy measurement](docs/Energy_Measurement.md) |
| Diagnose environment issues, OOM, timeouts, or partial results | [Troubleshooting](docs/Troubleshooting.md) |
| Read the measurement protocol and other topics | [Documentation index](docs/README.md) |

<a id="项目结构与开发"></a>

## Project structure and development

Clone the repository only for development or building an unpublished version:

```bash
git clone https://github.com/kainam15/AC-Prof.git
cd AC-Prof
uv venv --python 3.10
uv pip install --require-hashes -r requirements/host.lock -r requirements/dev.lock
uv pip install --no-deps -e .
source .venv/bin/activate
acprof --version
```

The existing hashed requirements remain the source of truth for development environments.
To install a checkout as an isolated tool with host checks, use the optional [`setup.sh` helper](docs/Distribution.md#clone-后初始化).

[`pyproject.toml`](pyproject.toml) declares host dependencies; [`requirements/`](requirements/) contains the `host`, `dev`, `test`, and `runtime-test` inputs and locks.

Run development tests with `python -m pytest` after installing `requirements/dev.lock`; test tools are separate from runtime dependencies.
See [architecture](docs/Architecture.md) for module responsibilities and the [testing guide](docs/Testing.md#开发质量检查) for development dependencies, pre-commit, and test commands.
To add a model or backend, follow the [adaptation contract](docs/Runtime_Compatibility.md#新增一个模型适配).
Agent collaboration rules are in [AGENTS.md](AGENTS.md).

<a id="许可与来源"></a>

## License and origins

The project code is licensed under [Apache-2.0](LICENSE), continuing the license declared in the original project definition.
AC-Prof is part of the DOR project at JNU DISTINT. See [NOTICE](NOTICE) for original contributors and subsequent maintenance credits.
See [project origins and evolution](docs/Project_Origin.md) for the original repository, the scope of current extensions, and how to use historical references.
Bundled LibriSpeech audio retains its [CC-BY-4.0](licenses/CC-BY-4.0.txt) license. Model code and weights remain subject to their respective repository licenses.

Independent repetitions can be compared with `acprof compare`; `acprof load` runs a separate non-streaming HTTP load protocol. See [comparison semantics](docs/Metrics.md#跨独立实验比较) and [load protocol](docs/Profiling_Protocol.md#独立非流式负载).
