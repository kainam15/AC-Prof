"""Static, task-scoped generation controls for the interactive preparation phase.

Metadata is evidence for defaults, not proof that a particular checkpoint accepts
every generation setting. The runtime's separate smoke validation remains authoritative.
Nothing in this module loads weights, probes hardware or runs during measurement.
"""
from __future__ import annotations

import copy
import math
from typing import Any

from acprof.task_workloads import (
    GENERATIVE_INTERFACES,
    merge_generation_parameters,
    task_fingerprint,
    task_label,
    validate_generation,
)

# The native Transformers generation adapters support this *small* editable set.
# Backend implementation is still validated with an actual request before collection.
_EDITABLE = (
    ("do_sample", "boolean", False, "随机采样"),
    ("temperature", "float", 1.0, "Temperature"),
    ("top_p", "float", 1.0, "Top-p"),
    ("num_beams", "integer", 1, "Beam 数量"),
)
_SUPPORTED_BACKENDS = frozenset({"transformers_model", "transformers_pipeline"})


def describe_task_parameters(model: Any, selected: dict) -> list[dict]:
    """Build a display/validation description from the selected task's pinned metadata.

    'interface_candidate' intentionally does not mean checkpoint-verified.
    Input scales and batch size already belong to the independent experiment plan.
    """
    if selected["interface"] not in GENERATIVE_INTERFACES or model.runtime_backend not in _SUPPORTED_BACKENDS:
        return []
    generation = selected["config"].get("generation")
    if not isinstance(generation, dict):
        return []
    source_defaults = model.repository_metadata.get("generation_config.json") or {}
    declarations = model.model_config.get("task_specific_params") or {}
    declared = declarations.get(selected["task_id"], {}) if isinstance(declarations, dict) else {}
    if not isinstance(declared, dict):
        declared = {}
    manual = "workload_spec.task_declaration" in selected.get("evidence", [])

    def source(key: str) -> str:
        if manual or key in declared:
            return "任务声明"
        if isinstance(source_defaults, dict) and key in source_defaults:
            return "模型生成配置"
        return "AC-Prof 默认值"

    label = task_label(selected)
    length_key = "max_new_tokens" if "max_new_tokens" in generation else "max_length"
    length_name = "输出 Token 上限" if length_key == "max_new_tokens" else "最大序列 Token 数"
    fields = [{
        "name": length_key, "task_name": label, "caption": length_name,
        "kind": "integer", "value": generation[length_key], "source": source(length_key),
        "support": "interface_candidate", "advanced": False,
    }]
    for name, kind, default, caption in _EDITABLE:
        fields.append({
            "name": name, "task_name": label, "caption": caption, "kind": kind,
            "value": generation.get(name, default), "source": source(name) if name in generation else "接口默认值",
            "support": "interface_candidate", "advanced": name not in {"do_sample"},
        })
    return fields


def parameter_review(model: Any, selected_tasks: list[dict], *,
                     batch_size: int = 1, input_scales: str = "") -> dict:
    """A stable list of typed questions, shared by the TUI and reply validator.

    Input size and batch size are workload controls for every task family.
    Generation controls are offered only for native compatible interfaces.
    """
    questions: list[dict] = []
    advanced: list[dict] = []
    for index, selected in enumerate(selected_tasks):
        for field in describe_task_parameters(model, selected):
            question = {
                "path": f"task_{index}.{field['name']}",
                "label": f"{field['task_name']} · {field['caption']}",
                "task_name": field["task_name"], "caption": field["caption"],
                "kind": field["kind"],
                "value": field["value"],
                "reason": field["source"],
            }
            if field["kind"] == "boolean":
                question["options"] = [False, True]
                question["labels"] = {"false": "关闭", "true": "开启"}
            (advanced if field["advanced"] else questions).append(question)
        from acprof.extensions import CATALOG
        task_name = task_label(selected)
        family = CATALOG.task_families.get(selected["interface"], model.task_family)
        for name, caption, kind, value, reason in (
            ("input_scales", "输入规模", "scales", input_scales, "留空自动规划"),
            ("batch_size", "Batch Size", "integer", batch_size, "实验配置"),
        ):
            questions.append({
                "path": f"task_{index}.{name}",
                "label": f"{task_name} · {caption}", "task_name": task_name,
                "caption": caption, "kind": kind, "value": value, "reason": reason,
                "task_family": family,
            })
    return {"questions": questions, "parameter_advanced": advanced}


def _validated_value(field: dict, value: Any) -> Any:
    name = field["path"].split(".", 1)[1]
    kind = field["kind"]
    if kind == "boolean":
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
    elif kind == "integer":
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    elif kind == "float":
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if name == "temperature" and value <= 0:
            raise ValueError("temperature must be positive")
        if name == "top_p" and not (0 < value <= 1):
            raise ValueError("top_p must be in (0, 1]")
    elif kind == "scales":
        if not isinstance(value, str):
            raise ValueError("input_scales must be text")
        if not value.strip():
            return ""
        parts = value.split(",")
        if len(parts) > 128 or any(not part.strip() for part in parts):
            raise ValueError("input_scales must be at most 128 comma-separated positive values")
        try:
            numbers = [float(part.strip()) for part in parts]
        except ValueError as exc:
            raise ValueError("input_scales must contain numbers") from exc
        if any(not math.isfinite(number) or number <= 0 for number in numbers):
            raise ValueError("input_scales must contain finite positive values")
        if field.get("task_family") in {"nlp", "timeseries"} and any(
            not number.is_integer() for number in numbers
        ):
            raise ValueError("this task requires integer input scales")
        return ",".join(str(int(item)) if item.is_integer() else str(item)
                        for item in sorted(set(numbers)))
    else:
        raise ValueError(f"unsupported parameter input kind: {kind}")
    return value


def apply_parameter_review(model: Any, selected_tasks: list[dict], answers: dict, *,
                           batch_size: int = 1, input_scales: str = "") -> list[dict]:
    """Validate all answers before changing any task or its frozen fingerprint.

    The resource knobs and generation kwargs have separate consumers; keeping
    them separate avoids accidentally passing batch_size to model.generate().
    """
    review = parameter_review(model, selected_tasks, batch_size=batch_size, input_scales=input_scales)
    fields = review["questions"] + review["parameter_advanced"]
    if not isinstance(answers, dict) or set(answers) != {field["path"] for field in fields}:
        raise ValueError("parameter review answers must cover exactly the displayed fields")
    generation_updates: list[dict] = [{} for _ in selected_tasks]
    experiment_updates: list[dict] = [{} for _ in selected_tasks]
    for field in fields:
        value = _validated_value(field, answers[field["path"]])
        if value != _validated_value(field, field["value"]):
            task_index = int(field["path"].split(".", 1)[0].removeprefix("task_"))
            name = field["path"].split(".", 1)[1]
            target = experiment_updates if name in {"batch_size", "input_scales"} else generation_updates
            target[task_index][name] = value
    changed = copy.deepcopy(selected_tasks)
    for task, generation_choices, experiment_choices in zip(
        changed, generation_updates, experiment_updates, strict=True,
    ):
        if not generation_choices and not experiment_choices:
            continue
        if generation_choices:
            generation = task["config"]["generation"]
            merged = merge_generation_parameters(generation, generation_choices)
            if not merged.get("do_sample", False) and any(
                name in generation_choices for name in ("temperature", "top_p")
            ):
                raise ValueError("开启随机采样后才能调整 Temperature 或 Top-p")
            beams = merged.get("num_beams", 1)
            sequences = merged.get("num_return_sequences", 1)
            if type(beams) is int and type(sequences) is int and (
                beams > 1 and sequences > beams or
                beams == 1 and sequences > 1 and merged.get("do_sample", False) is False
            ):
                raise ValueError("生成序列数与 Beam 数量、随机采样设置不兼容")
            validate_generation(merged)
            task["config"]["generation"] = merged
            task["config"]["user_generation_overrides"] = dict(generation_choices)
        if experiment_choices:
            task["config"]["user_experiment_overrides"] = dict(experiment_choices)
        task["fingerprint"] = task_fingerprint(task)
    return changed
