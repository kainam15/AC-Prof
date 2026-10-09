<p align="center">
  <img alt="AC-Prof" src="docs/assets/logos/acprof-logo-light.svg#gh-light-mode-only" width="55%">
  <img alt="AC-Prof" src="docs/assets/logos/acprof-logo-dark.svg#gh-dark-mode-only" width="55%">
</p>

**English** · [简体中文](docs/i18n/README_zh-CN.md)

AC-Prof profiles containerized AI model inference across CPU, memory, GPU, and input-size configurations. It prepares Docker runtimes, mounts shared model weights read-only, and produces reproducible measurements and analysis artifacts. Compatible models can be profiled without changing their code.

## What AC-Prof does

| Capability | Description |
| --- | --- |
| Performance and resources | Application latency, throughput, CPU and memory usage; GPU metrics when available |
| Full profiling | Energy, packet-level timing, CPU hardware counters, and optional independent profilers |
| Reproducible experiments | Input and environment metadata, layered CSV results, plots, and reports |

AC-Prof supports text, vision, audio, time series, diffusion, multimodal, and structured-data workloads when the model meets the [supported task interfaces](docs/Runtime_Compatibility.md#任务支持范围). A task label alone does not guarantee that every checkpoint will run.

## Quick start

Install the latest published release with [uv](https://docs.astral.sh/uv/getting-started/installation/) and launch the interactive terminal interface:

```bash
uv tool install acprof
acprof tui
```

Collection requires a suitable Linux host with a local Docker Engine and cgroup v2. Native Linux x86_64 is the FULL measurement platform; WSL2 has limited basic/PARTIAL support. See [host requirements](docs/Getting_Started.md#1-检查主机环境).

## Documentation

| Goal | Guide |
| --- | --- |
| Prepare the host, run a first CPU experiment, or inspect results | [Installation and experiments](docs/Getting_Started.md) |
| Understand the terminal interface | [TUI user guide](docs/TUI.md) |
| Use command-line options and tools | [CLI reference](docs/CLI_Reference.md) |
| Interpret metrics, CSV layers, and reports | [Metrics and results](docs/Metrics.md) |
| Understand supported models and runtimes | [Runtime compatibility](docs/Runtime_Compatibility.md) |
| Develop or contribute to AC-Prof | [Developer installation](docs/Getting_Started.md#开发安装), [architecture](docs/Architecture.md), [testing](docs/Testing.md) |

[All documentation](docs/README.md) · [Troubleshooting](docs/Troubleshooting.md)

## License and origins

Project code is licensed under [Apache-2.0](LICENSE). Bundled LibriSpeech audio retains [CC-BY-4.0](licenses/CC-BY-4.0.txt); model code and weights have their own licenses. AC-Prof is part of the DOR project at JNU DISTINT. See [NOTICE](NOTICE) and [project origins](docs/Project_Origin.md).
