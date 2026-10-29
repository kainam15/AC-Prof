"""Pure selection data derived from already normalized inference profiles.

No grouping, default inference, translation, I/O or workload execution belongs
here. The execution signature is the sole source of computational differences.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from acprof.models.evidence import content_digest

type ConfigPath = tuple[str | int, ...]
_MISSING = object()


@dataclass(frozen=True)
class ConfigField:
    path: ConfigPath
    value: Any
    present: bool = True


@dataclass(frozen=True)
class ConfigurationComparison:
    common: tuple[ConfigField, ...]
    differences: tuple[tuple[ConfigField, ...], ...]


def compare_configurations(configurations: list[dict]) -> ConfigurationComparison:
    """Compare every JSON field, preserving types, missing values and list order.

    Mapping keys sort lexically and list indices numerically. Incompatible
    container types (including absent containers) remain complete leaf values.
    Equality uses the signature's canonical JSON identity: False != 0 != 0.0.
    Defaults and overrides must already be resolved by inference_groups.
    """
    common: list[ConfigField] = []
    differences: list[list[ConfigField]] = [[] for _ in configurations]

    def visit(path: ConfigPath, values: list[Any]) -> None:
        if all(isinstance(value, dict) for value in values):
            keys = sorted({key for value in values for key in value})
            if keys:
                for key in keys:
                    visit((*path, key), [value.get(key, _MISSING) for value in values])
                return
        if all(isinstance(value, list) for value in values):
            length = max(map(len, values))
            if length:
                for index in range(length):
                    visit((*path, index), [value[index] if index < len(value) else _MISSING
                                          for value in values])
                return
        fields = [ConfigField(path, copy.deepcopy(value) if value is not _MISSING else None,
                              value is not _MISSING) for value in values]
        if all(field.present for field in fields) and len({content_digest(field.value) for field in fields}) == 1:
            common.append(fields[0])
        else:
            for target, field in zip(differences, fields, strict=True):
                target.append(field)

    if configurations:
        visit((), configurations)
    return ConfigurationComparison(tuple(common), tuple(tuple(fields) for fields in differences))


@dataclass(frozen=True)
class TaskPurpose:
    kind: str
    task_ids: tuple[str, ...]
    directions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SelectionGroup:
    task_id: str
    purposes: tuple[TaskPurpose, ...]
    representative: dict
    differences: tuple[ConfigField, ...]
    profile: dict


@dataclass(frozen=True)
class TaskSelection:
    common: tuple[ConfigField, ...]
    groups: tuple[SelectionGroup, ...]
    semantic_count: int


def build_task_selection(profiles: list[dict]) -> TaskSelection:
    """Build stable display data without changing IDs, provenance or grouping."""
    ordered = sorted(copy.deepcopy(profiles), key=lambda item: item["task_id"])
    configurations = []
    for item in ordered:
        signature = copy.deepcopy(item["execution_profile"]["signature"])
        # Full protocols, component hashes and evidence belong in artifacts.
        # The interface and effective generation parameters explain the choice.
        execution = signature.pop("execution", {})
        if execution:
            signature["operation"] = execution["operation"]
        configurations.append(signature)
    comparison = compare_configurations(configurations)
    groups = []
    semantic_ids: set[str] = set()
    for profile, differences in zip(ordered, comparison.differences, strict=True):
        members = sorted(profile["semantic_tasks"], key=lambda item: item["task_id"])
        semantic_ids.update(member["task_id"] for member in members)
        buckets: dict[str, list[dict]] = {}
        for member in members:
            buckets.setdefault(member["kind"], []).append(member)
        purposes = tuple(TaskPurpose(kind, tuple(item["task_id"] for item in tasks), tuple(sorted({
            (item["config"]["source_language"], item["config"]["target_language"])
            for item in tasks if "source_language" in item["config"] and "target_language" in item["config"]
        }))) for kind, tasks in sorted(buckets.items()))
        representative_id = profile["execution_profile"]["representative_task_id"]
        representative = next(item for item in members if item["task_id"] == representative_id)
        groups.append(SelectionGroup(profile["task_id"], purposes, representative, differences, profile))
    return TaskSelection(comparison.common, tuple(groups), len(semantic_ids))
