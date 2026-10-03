"""Explicit host overrides that select a locked inference environment."""
from __future__ import annotations

import os


def runtime_build_overrides() -> dict[str, str]:
    """Keep preparation, fingerprints and coverage recovery on the same inputs."""
    return {name: os.environ.get(name, "").strip() for name in (
        "ACPROF_NLP_TORCH_INDEX_URL", "ACPROF_NLP_TORCH_SPEC", "ACPROF_HOST_CUDA_VERSION")}
