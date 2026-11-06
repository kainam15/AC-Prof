<p align="center">
  <img alt="AC-Prof" src="docs/assets/logos/acprof-logo-light.svg#gh-light-mode-only" width="55%">
  <img alt="AC-Prof" src="docs/assets/logos/acprof-logo-dark.svg#gh-dark-mode-only" width="55%">
</p>

**English** · [简体中文](docs/i18n/README_zh-CN.md)

AC-Prof profiles AI model inference across CPU, memory, GPU, and input-size configurations. It prepares Docker or isolated Process runtimes, reuses pinned model weights, and produces reproducible measurements and analysis artifacts. Compatible models can be profiled without changing their code.

## What AC-Prof does

| Capability                | Description                                                                            |
| ------------------------- | -------------------------------------------------------------------------------------- |
| Performance and resources | Application latency, throughput, CPU and memory usage; GPU metrics when available      |
| Full profiling            | Energy, packet-level timing, CPU hardware counters, and optional independent profilers |
| Reproducible experiments  | Input and environment metadata, layered CSV results, plots, and reports                |

The [offline HTML report](docs/results/reports.md#交互式配置比较报告) is the unified analysis entry point. One `result_all.csv` supports descriptive CPU, memory and GPU comparisons and within-run window statistics. Explicit groups of independent runs use the existing strict identity and condition checks; each side needs at least three valid runs for a Bootstrap confidence interval. Multiple models can be viewed side by side with task/input comparability limits. Python computes all estimates and reuses matching statistics JSON; the offline browser filters, plots and exports CSV / JSON. Historical CSVs remain usable without fabricated run identities or new collection. Completed experiments (including `--resume`) also generate a new report under `plots/analysis/`.

```bash
acprof report result_all.csv --output resources.html
acprof report model-a.csv model-b.csv --output models.html
acprof compare --left run-a/ --right run-b/ --html-output comparison.html \
  --output comparison.json --csv-output comparison.csv
```

Repeat `--left` and `--right` for independent replicates. A single run per side shows descriptive differences and ratios without a run-level interval. In the TUI, **Results → Analysis** provides inputs, analysis options, HTML actions and optional JSON / CSV exports. The **Compare** page is hidden; comparison backends, CLI commands and HTML reports remain available.

`acprof load --base-url` measures an existing OpenAI-compatible streaming service. Explicit concurrency and request-rate sweeps support `--dry-run` planning and separate reports for each point; see [streaming load sweeps](docs/profiling/streaming.md#多负载点编排). `acprof report <sweep.json>` generates [offline throughput and latency curves](docs/results/reports.md#多负载点流式容量报告), with condition groups, failed points and missing evidence preserved. Optional [vLLM/SGLang identity probes](docs/profiling/streaming.md#服务身份) preserve service observations separately from user declarations. Explicit `vllm-config` probes read selected engine settings from an already available development endpoint; SGLang also reports selected startup configuration.

`acprof load --request-jsonl` runs an [ordered local request dataset](docs/profiling/streaming.md#本地请求集) with distinct prompts and generation settings. Dataset reports preserve input order and per-request mappings for offline analysis.

AC-Prof supports text, vision, audio, time series, diffusion, multimodal, and structured-data workloads when the model meets the [supported task interfaces](docs/models/supported-tasks.md#任务支持范围). A task label alone does not guarantee that every checkpoint will run.

## Collection environments and metrics

| Environment                                                     | Supported collection modes | Collectable metrics                                                                                                                                                     | Main limitations                                                                                                      |
| --------------------------------------------------------------- | :------------------------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Native Linux x86_64 + local Docker                              |  `basic`,<br />`full`  | `basic`: latency, throughput, container CPU/memory and available GPU metrics;<br />`full`: additionally supports energy, PMU, packet timing, and optional profilers | `full` requires cgroup v2, RAPL, perf, and packet-capture permissions                                               |
| Linux VM/cloud host + local Docker                              |         `basic`         | Latency, throughput, container CPU/memory and available GPU metrics                                                                                                     | Environment capability tier remains `PARTIAL`; no energy, PMU, or packet collection                                 |
| Linux x86_64 + Process (including Docker-free cloud containers) |         `basic`         | Latency, throughput, process-tree CPU/RSS, and available device-wide GPU utilization/VRAM                                                                               | No energy, PMU, packet capture, or independent profilers; numeric resource quotas require delegated cgroup v2         |
| WSL2 + Docker/Process                                           |         `basic`         | Latency, throughput, Linux guest CPU/memory, and limited GPU metrics                                                                                                    | `PARTIAL` tier; no native hardware energy/PMU; Docker collection requires an engine inside WSL2, not Docker Desktop |
| Native Windows/macOS                                            |             —             | —                                                                                                                                                                      | —                                                                                                                    |

`basic`/`full` are collection modes, not the environment capability tiers `FULL`/`PARTIAL`. Available metrics depend on hardware and permissions; independent profilers must be explicitly enabled, and GPU collection on cloud hosts or WSL2 requires instance-specific validation. See [Linux runtimes](docs/platforms/linux-runtimes.md), [WSL2](docs/platforms/wsl2.md), [metric coverage](docs/results/metrics.md), and [host requirements](docs/getting_started/installation.md#1-检查主机环境).

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

### Develop from source

To contribute or work on the latest source code, clone the repository and let uv prepare the editable project and development tools:

```bash
git clone https://github.com/kainam15/AC-Prof.git
cd AC-Prof
uv sync
uv run acprof tui
```

For developer environment setup and testing, see [developer setup](docs/development/distribution.md#本地开发环境) and [testing](docs/development/testing.md).

## Documentation

For user guides, technical references, and development documentation, see the [documentation index](docs/README.md).

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and review practices, and [SECURITY.md](SECURITY.md) for confidential vulnerability reporting and the third-party model trust boundary.

## License and origins

Project code is licensed under [Apache-2.0](LICENSE). Bundled LibriSpeech audio retains [CC-BY-4.0](licenses/CC-BY-4.0.txt); model code and weights have their own licenses. AC-Prof is part of the DOR project at JNU DISTINT. See [NOTICE](NOTICE) and [project origins](docs/development/history.md).
