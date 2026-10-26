<p align="center">
  <img alt="AC-Prof" src="docs/assets/logos/acprof-logo-light.svg#gh-light-mode-only" width="55%">
  <img alt="AC-Prof" src="docs/assets/logos/acprof-logo-dark.svg#gh-dark-mode-only" width="55%">
</p>

**English** · [简体中文](docs/i18n/README_zh-CN.md)

AC-Prof profiles AI model inference across CPU, memory, GPU, and input-size configurations. It prepares Docker or isolated Process runtimes, reuses pinned model weights, and produces reproducible measurements and analysis artifacts. Compatible models can be profiled without changing their code.

## What AC-Prof does

| Capability | Description |
| --- | --- |
| Performance and resources | Application latency, throughput, CPU and memory usage; GPU metrics when available |
| Full profiling | Energy, packet-level timing, CPU hardware counters, and optional independent profilers |
| Reproducible experiments | Input and environment metadata, layered CSV results, plots, and reports |

AC-Prof supports text, vision, audio, time series, diffusion, multimodal, and structured-data workloads when the model meets the [supported task interfaces](docs/models/supported-tasks.md#任务支持范围). A task label alone does not guarantee that every checkpoint will run.

## Quick start

Install the latest release from [PyPI](https://pypi.org/project/acprof/) with [uv](https://docs.astral.sh/uv/getting-started/installation/) and launch the interactive terminal interface:

```bash
uv tool install acprof
acprof --version
acprof tui
```

Upgrade an existing PyPI installation with:

```bash
uv tool upgrade acprof
```

For a GitHub Release wheel or Linux standalone binary, see the [latest release](https://github.com/kainam15/AC-Prof/releases/latest) and the [installation guide](docs/getting_started/installation.md#github-release-fallback) for checksum verification.

Native Linux x86_64 with local Docker and cgroup v2 remains the FULL measurement platform. Current source builds also provide basic collection through Process Runtime on Linux without Docker or root. Process CPU/RSS, inherited limits, and shared-device GPU metrics retain separate evidence and comparison rules. Cloud GPU deployments still require platform-specific validation. See [Linux runtimes](docs/platforms/linux-runtimes.md) and [host requirements](docs/getting_started/installation.md#1-检查主机环境).

### Develop from source

To contribute or work on the latest source code, clone the repository and let uv prepare the editable project and development tools:

```bash
git clone https://github.com/kainam15/AC-Prof.git
cd AC-Prof
uv sync
uv run acprof tui
```

`uv sync` creates and updates this checkout's `.venv`; no manual activation is needed. Each Git worktree has its own environment and shares uv's download cache. Ordinary release users continue to use `uv tool install acprof` and `acprof tui`.

See [developer setup](docs/development/distribution.md#本地开发环境) and [testing](docs/development/testing.md) for the full workflow.

Docker Runtime reuses verified model images automatically and builds missing ones. Use `--rebuild-image` or the TUI's advanced image options for a one-time rebuild; Docker cache may still be used. For Process, choose basic mode and inherited CPU/memory limits in the TUI. See [image preparation](docs/models/images.md#构建复用和验证) and [runtime selection](docs/platforms/linux-runtimes.md#选择与准备).

The TUI tracks collection and probe progress through structured events. After a collection exits, it verifies the saved experiment identity and audits the result before reporting success; a successful process exit alone is insufficient.

Enter a model ID to discover its declared use cases from lightweight metadata. AC-Prof groups applications that share the same inference interface and compute-relevant decoding settings into one benchmark workload: the three translation pairs and summarization declared by `google-t5/t5-small` normally share one Seq2Seq profile instead of four redundant runs. Only multiple distinct inference profiles require selection before downloading weights. CLI users can select declared use cases or profiles with `--tasks`, or choose all with `--tasks all`; identical profiles are collected once. Each profile retains the original task evidence, deterministic exemplar input, independent Smoke validation, and reproducible results. See [inference grouping](docs/models/routing.md#任务发现与选择) and [profile results and resume](docs/profiling/artifacts.md#按任务保存与恢复).

The monitor shows download progress, runtime validation results, or case and resource configuration according to the current phase, hiding unrelated information. Error reasons and related logs stay visible above INFO output; logs still support colored levels, literal search, a detailed view, and copying or exporting retained original text. Download metrics distinguish cached data from SDK network payload. See [logs and download progress](docs/usage/tui-experiments.md#日志查看与复制) for retention limits and measurement behavior.

Once the validated result directory is initialized, each preparation attempt retains a redacted host Python log and a structured stage/failure summary. Resume preserves earlier attempts, and both UI log queues have explicit retention limits. See [preparation diagnostics](docs/profiling/artifacts.md#准备阶段诊断) for scope and failure handling.

Download confirmation groups the overview, Runtime and sources, storage, and network preflight into labeled sections with all information expanded. Before confirmation, a bounded model GET probe reads up to 256 KiB in total, within the download budget; zero budget skips GET. Full downloads begin only after confirmation. A passing probe does not guarantee the complete transfer. See [download preflight](docs/usage/tui-experiments.md#下载与磁盘预检).

CSV headers and result directories exist before inference starts. Each completed measurement window appends to the root module CSVs; completion freezes an immutable snapshot. Use AC-Prof readers for a consistent committed view, including during recovery and postprocessing. See [live results and recovery](docs/profiling/artifacts.md#实时提交与后处理).

### Generate an interactive report

Run `acprof tui` → **Reports** → **Select experiment** → **Generate HTML**. The page shows the full output path, window counts, and data quality limitations. Choose **Open HTML** on a local graphical desktop, or **Export / Remote access** → **Export HTML** to copy it to a new file. Each generation preserves older reports; collection completion never generates HTML automatically.

The self-contained HTML includes **Comparison Matrix**, **Pareto / Trade-off**, and **Scaling**, with no network or web service required. On an SSH server such as dor-server, download the displayed file using SFTP or your remote IDE and open it in your local browser. See the [TUI report workflow](docs/usage/tui-pages.md#交互式-html-报告) for multiple inputs and output locations.

The standalone CLI remains available:

```bash
acprof report results/run-a/ results/run-b/ --output comparison.html
```

## Documentation

| Goal | Guide |
| --- | --- |
| Prepare the host, run a first CPU experiment, or inspect results | [Installation and experiments](docs/getting_started/installation.md) |
| Understand the terminal interface | [TUI user guide](docs/usage/tui.md) |
| Use command-line options and tools | [CLI reference](docs/usage/cli.md) |
| Interpret metrics, CSV layers, and reports | [Metrics and results](docs/results/metrics.md) |
| Understand supported models and runtimes | [Runtime compatibility](docs/models/runtime.md) |
| Develop or contribute to AC-Prof | [Developer installation](docs/development/distribution.md#本地开发环境), [architecture](docs/development/architecture.md), [testing](docs/development/testing.md) |

[All documentation](docs/README.md) · [Troubleshooting](docs/usage/troubleshooting.md)

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and review practices, and [SECURITY.md](SECURITY.md) for confidential vulnerability reporting and the third-party model trust boundary.

## License and origins

Project code is licensed under [Apache-2.0](LICENSE). Bundled LibriSpeech audio retains [CC-BY-4.0](licenses/CC-BY-4.0.txt); model code and weights have their own licenses. AC-Prof is part of the DOR project at JNU DISTINT. See [NOTICE](NOTICE) and [project origins](docs/development/history.md).
