# Contributing to AC-Prof

AC-Prof is a research-oriented measurement tool. Changes must preserve experiment provenance, measurement-window boundaries, and the explicit representation of unavailable metrics.

## Development setup

Follow the [source installation instructions](README.md#develop-from-source) and the [testing guide](docs/development/testing.md). Python 3.10+ is supported. Use pinned, hash-locked development dependencies and a separate virtual environment for each worktree.

## Before submitting a change

1. Open a narrowly scoped branch and describe the behavior being changed, including measurement or schema implications.
2. Add focused pytest regression tests. Avoid using real network access, model downloads, or Docker/GPU for ordinary unit tests.
3. Run `python -m pytest` for affected tests, `python -m mypy` in the locked development environment, `python scripts/check_private_api.py`, and the documented pre-commit checks.
4. Update relevant Chinese/English user-facing documentation together, and explicitly identify any Native Linux hardware validation still required.
5. Use the repository's [commit message conventions](.github/commit-message.instructions.md) and open a pull request with verification evidence. Never include passwords, access tokens, private model weights, or experimental data that should remain private.

See [architecture](docs/development/architecture.md), [CI and validation](docs/development/ci.md), and [security reporting](SECURITY.md) for more information. A successful offline test does not prove GPU energy, PMU, NVML, or real-model runtime correctness.
