<p align="center">
  <img alt="AC-Prof" src="docs/assets/logos/acprof-logo-light.svg#gh-light-mode-only" width="55%">
  <img alt="AC-Prof" src="docs/assets/logos/acprof-logo-dark.svg#gh-dark-mode-only" width="55%">
</p>

**English** · [简体中文](docs/i18n/README_zh-CN.md)

AC-Prof profiles AI model inference across CPU, memory, GPU, and input-size configurations.
It prepares Docker or isolated Process runtimes, reuses pinned model weights, and produces
reproducible measurements without requiring changes to compatible model code.

## Capabilities

| Area | What you get |
| --- | --- |
| Performance and resources | Application latency, throughput, CPU/memory usage, and available GPU metrics |
| Full profiling | Energy, packet-level timing, CPU hardware counters, and optional independent profilers |
| Reproducible analysis | Input/environment metadata, layered CSV results, plots, and offline HTML reports |

Supported workloads include text, vision, audio, time series, diffusion, multimodal, and structured data.
Model support depends on the [task interface requirements](docs/models/supported-tasks.md#任务支持范围);
a task label alone does not guarantee that every checkpoint will run.

## Quick start

Install the [PyPI release](https://pypi.org/project/acprof/) with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv tool install acprof
acprof --version
acprof tui
```

To upgrade an existing installation, run `uv tool upgrade acprof`.
For host preparation, verified release assets, and the first experiment, see the
[installation guide](docs/getting_started/installation.md) and [experiment guide](docs/usage/experiments.md).

## Measurement and analysis

- **Experiments:** Run resource/input matrices from the [TUI](docs/usage/tui.md) or [CLI](docs/usage/cli.md).
  Inspect the [metrics and result layout](docs/results/metrics.md) after collection.
- **Offline reports:** Explore resource comparisons, window statistics, charts, and exports in
  [HTML reports](docs/results/reports.md#交互式配置比较报告). Independent run comparisons and confidence
  intervals have separate [eligibility rules](docs/results/comparisons.md#跨独立实验比较).
- **Streaming services:** Measure an already-running OpenAI-compatible service with `acprof load`.
  Concurrency/rate sweeps, ordered JSONL requests, and optional vLLM/SGLang service observations
  are described in the [streaming guide](docs/profiling/streaming.md).

## Platform support

Native Linux x86_64 with local Docker supports `basic` and, when hardware/permissions allow, `full`.
Linux VM/cloud and Docker-free Process setups support a more limited `basic` collection path.
WSL2 also has restricted `basic` support; native Windows/macOS collection is not supported.
Actual metrics depend on environment capabilities and permissions—not on the mode name alone.
See [platform support matrix](docs/platforms/linux-runtimes.md#支持环境与采集模式), [WSL2](docs/platforms/wsl2.md), and
[host requirements](docs/getting_started/installation.md#1-检查主机环境).

## Documentation and development

Start at the [documentation index](docs/README.md) for installation, usage, models, measurement
protocols, result analysis, platform limitations, and development references.

To work from source:

```bash
git clone https://github.com/kainam15/AC-Prof.git
cd AC-Prof
uv sync
uv run acprof tui
```

See [developer setup](docs/development/distribution.md#本地开发环境) and
[testing](docs/development/testing.md).

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution practices and
[SECURITY.md](SECURITY.md) for vulnerability reporting and model-code trust boundaries.

## License and origins

Project code is [Apache-2.0](LICENSE); bundled LibriSpeech audio is
[CC-BY-4.0](licenses/CC-BY-4.0.txt). Third-party models retain their own licenses.
AC-Prof is part of the DOR project at JNU DISTINT; see [NOTICE](NOTICE) and
[project origins](docs/development/history.md).
