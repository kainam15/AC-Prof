"""Versioned project baselines and evidence-backed, published task presets."""
from __future__ import annotations

import copy
import json
from typing import Any

from acprof.extensions.parameters import validate_parameter_value

BENCHMARK_VERSION = "acprof-benchmark-v1"
_BASELINE = {
    "max_new_tokens": 64, "max_length": 64, "do_sample": False, "num_beams": 1,
    "temperature": 1.0, "top_p": 1.0, "prediction_length": 64,
    "language": "en", "asr_task": "transcribe", "num_inference_steps": 25, "guidance_scale": 7.5,
}


def _benchmark_values(fields: list[dict]) -> tuple[dict[str, Any], str | None]:
    """Validate project baselines against each inference parameter's declared spec."""
    inference_fields = [field for field in fields if field["spec"].scope == "inference"]
    if not inference_fields:
        return {}, None
    baselines: dict[str, Any] = {}
    for field in inference_fields:
        spec = field["spec"]
        name = spec.name
        if name not in _BASELINE:
            return {}, name
        candidate = _BASELINE[name]
        try:
            validate_parameter_value(spec, candidate, path=field["path"])
        except ValueError:
            return {}, name
        baselines[field["path"]] = copy.deepcopy(candidate)
    return baselines, None


def author_defaults(fields: list[dict]) -> dict:
    """Only raw, revision-bound task_specific_params count as an author preset.

    Global generation defaults and AC-Prof defaults are not recommendations.
    Conflicting semantic tasks cannot be compressed to one recommendation.
    """
    selected = {}
    for field in fields:
        candidates = [
            candidate for candidate in field.get("candidates", [])
            if candidate["validation"] == "valid"
            and candidate["source"].get("raw_sha256")
            and candidate["source"].get("revision")
            and candidate["source"]["json_pointer"].startswith("/task_specific_params/")
        ]
        if not candidates:
            continue
        if len({json.dumps(item["value"], sort_keys=True) for item in candidates}) != 1:
            raise ValueError("author task presets conflict; choose explicit parameters")
        selected[field["path"]] = candidates[0]
    return selected


def apply_preset(fields: list[dict], preset: str) -> None:
    if preset == "author":
        selected = author_defaults(fields)
        if not selected:
            raise ValueError("author preset unavailable: no pinned published task parameters")
        for field in fields:
            if field["path"] in selected:
                candidate = selected[field["path"]]
                field["default"] = copy.deepcopy(candidate["value"])
                field["source"] = {"kind": "author_task_preset", "preset_version": "author-task-v1",
                                   "evidence": copy.deepcopy(candidate["source"])}
    elif preset == "benchmark":
        inference_fields = [field for field in fields if field["spec"].scope == "inference"]
        if not inference_fields:
            raise ValueError("benchmark preset unavailable: no registered task parameters")
        baselines, missing = _benchmark_values(fields)
        if missing is not None:
            raise ValueError(f"benchmark preset has no registered baseline for {missing}")
        for field in inference_fields:
            field["default"] = copy.deepcopy(baselines[field["path"]])
            field["source"] = {"kind": "benchmark_preset", "preset_version": BENCHMARK_VERSION,
                               "scope": "AC-Prof task parameters; not an external benchmark certification"}


def additional_presets(fields: list[dict]) -> list[dict]:
    if not any(field["spec"].scope == "inference" for field in fields):
        return []
    presets: list[dict] = []
    _, missing_baseline = _benchmark_values(fields)
    if missing_baseline is None:
        presets.append({
            "id": "benchmark", "label": "AC-Prof Benchmark v1",
            "description": "Fixed AC-Prof task parameters; retain the reviewed workload and loading policy.",
            "requests": {"schema_version": 1, "preset": "benchmark"},
        })
    try:
        supported = author_defaults(fields)
    except ValueError:
        supported = {}
    if supported:
        presets.append({
            "id": "author", "label": "Author task preset",
            "description": "Published task_specific_params at the pinned revision; other fields keep their displayed defaults.",
            "requests": {"schema_version": 1, "preset": "author"},
        })
    for preset in presets:
        request = preset["requests"]
        for field in fields:
            if field["spec"].scope == "inference" and (
                preset["id"] == "benchmark" or field["path"] in supported
            ):
                continue
            value = copy.deepcopy(field["request"])
            if "device" in field:
                request.setdefault("loading", {}).setdefault(field["device"], {})[field["spec"].name] = value
            elif "task_id" in field:
                request.setdefault("tasks", {}).setdefault(field["task_id"], {})[field["spec"].name] = value
            else:
                request[field["path"]] = value
    return presets
