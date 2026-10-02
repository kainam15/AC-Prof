"""Result quality evidence, independent of execution/measurement capabilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QualityCheck:
    code: str
    severity: str
    observed: Any
    threshold: Any
    detail: str
    evidence: dict

    def to_dict(self):
        return asdict(self)


def loading_quality(info: dict, *, source: str) -> list[dict]:
    checks = []
    for key, code in (("missing_keys", "weights_reinitialized"), ("mismatched_keys", "weights_reinitialized"),
                      ("unexpected_keys", "unused_checkpoint_weights")):
        values = info.get(key) or []
        if values:
            checks.append(QualityCheck(code, "warning", sorted(str(item) for item in values), 0,
                f"Checkpoint loading reported {key}", {"source": source, "loading_info_key": key}).to_dict())
    return checks


def cli_exit_quality(exit_code, *, source: str) -> list[dict]:
    if exit_code is not None:
        return []
    return [QualityCheck("missing_cli_exit_code", "warning", None, "recorded process exit code",
                         "Process termination evidence is missing", {"source": source}).to_dict()]


def write_quality(path: Path, checks: list[dict], *, append=False):
    from acprof.artifacts import atomic_write_json
    previous = json.loads(path.read_text()).get("checks", []) if append and path.exists() else []
    unique = {json.dumps(value, sort_keys=True): value for value in [*previous, *checks]}
    atomic_write_json(path, {"schema_version": 1, "checks": list(unique.values())})


def collect_quality(root: Path, case_csvs=()) -> list[dict]:
    from acprof.artifact_layout import ArtifactLayout, case_sidecar
    layout = ArtifactLayout.discover(root)
    checks = []
    validation_path = layout.path("runtime_validation.json")
    if validation_path.exists():
        for device, result in json.loads(validation_path.read_text()).get("devices", {}).items():
            for check in result.get("quality_checks", []):
                checks.append({**check, "evidence": {**check["evidence"], "device": device}})
    for csv in case_csvs:
        path = case_sidecar(csv, "quality_checks")
        if path.exists():
            checks.extend(json.loads(path.read_text())["checks"])
    write_quality(layout.path("quality_checks.json"), checks)
    return checks
