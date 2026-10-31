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
