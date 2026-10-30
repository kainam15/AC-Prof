"""Recover metadata accompanying a posthoc commit after process death.

This coordinates recovery, not whole-directory reader transactions.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from acprof.artifacts import loads_finite_json, read_json_object, sync_directory
from acprof.result_store.protocol import read_reference, write_bytes
from acprof.results.csv import ResultValidationError
from acprof.results.layout import ArtifactLayout


@dataclass(frozen=True)
class MetadataRecoveryPlan:
    before: dict[Path, bytes | None]
    after: dict[Path, bytes | None]


def prepare_metadata_recovery(layout: ArtifactLayout, intent: dict, *, run_id: str) -> MetadataRecoveryPlan:
    """Validate both versions and encode every document before any file changes."""
    if set(intent) != {"run_id", "commit", "before", "after"} or intent["run_id"] != run_id:
        raise ResultValidationError("invalid posthoc metadata recovery identity or fields")
    baseline = intent["commit"]
    if (not isinstance(baseline, dict) or set(baseline) != {"path", "sha256"}
            or not isinstance(baseline["path"], str) or not isinstance(baseline["sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", baseline["sha256"]) is None):
        raise ResultValidationError("invalid posthoc metadata recovery commit")
    if not layout.contained(baseline["path"]).is_relative_to(layout.result_store_path("commits")):
        raise ResultValidationError("posthoc metadata recovery commit is outside its journal")
    allowed = {layout.path(name).relative_to(layout.root).as_posix()
               for name in ("static_meta.json", "collection_history.json")}
    versions = {}
    for phase in ("before", "after"):
        documents = intent[phase]
        if not isinstance(documents, dict) or set(documents) != allowed:
            raise ResultValidationError(f"invalid posthoc metadata recovery paths in {phase}")
        prepared: dict[Path, bytes | None] = {}
        for name, content in documents.items():
            path = layout.contained(name)
            if content is None and phase == "before":
                prepared[path] = None
                continue
            if not isinstance(content, str):
                raise ResultValidationError(f"invalid posthoc metadata recovery content in {phase}: {name}")
            try:
                raw = content.encode("utf-8")
                payload = loads_finite_json(content.removeprefix("\ufeff"))
                if not isinstance(payload, dict):
                    raise ValueError("top-level JSON value must be an object")
            except (UnicodeError, ValueError, RecursionError) as error:
                raise ResultValidationError(f"invalid posthoc metadata recovery JSON in {phase}: {name}: {error}") from error
            prepared[path] = raw
        versions[phase] = prepared
    return MetadataRecoveryPlan(versions["before"], versions["after"])


def write_metadata(documents: Mapping[Path, bytes | None]) -> None:
    """Apply a validated plan; the caller retains its journal until all syncs succeed."""
    for path, raw in documents.items():
        if raw is None:
            path.unlink(missing_ok=True)
            sync_directory(path.parent)
        else:
            write_bytes(path, raw)


def _validate_baseline(layout: ArtifactLayout, manifest: dict, baseline: dict) -> None:
    # The writer has already replayed this chain under its publication lock.
    # Usually only the posthoc commit and its optional freeze follow baseline.
    current = manifest["commit"]
    for _ in range(manifest["sequence"] + 1):
        if current == baseline:
            return
        if current is None:
            break
        current = read_reference(layout, current).get("previous")
    raise ResultValidationError("posthoc metadata recovery commit is not on the current result chain")


def recover_metadata(layout, manifest):
    pending = layout.result_store_path("metadata_pending.json")
    if not pending.exists():
        return
    intent = read_json_object(pending, bounded=False)
    plan = prepare_metadata_recovery(layout, intent, run_id=manifest["run_id"])
    _validate_baseline(layout, manifest, intent["commit"])
    write_metadata(plan.before if manifest["commit"] == intent["commit"] else plan.after)
    pending.unlink()
    sync_directory(pending.parent)
