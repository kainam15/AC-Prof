"""Read-only schema and artifact routing for recorded hardware conditions.

Host observation lives in ``acprof.host.hardware_conditions``; offline analysis
must not import the collectors just to locate historic evidence.
"""
from __future__ import annotations

from pathlib import Path

from acprof.artifact_layout import ArtifactLayout

HARDWARE_FIELDS = ("host_id", "cpu_model", "cpu_affinity", "cpu_policy", "gpu", "runtime_threads")


def conditions_path(layout: ArtifactLayout) -> Path:
    """Locate optional case conditions without changing legacy layout semantics."""
    name = "metadata/hardware_conditions.json" if layout.layout_version == 2 else "hardware_conditions.json"
    return layout.contained(name)
