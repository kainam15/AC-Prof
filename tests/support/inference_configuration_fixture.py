"""Offline frozen configuration evidence shared by host boundary tests."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

from acprof.host.detect import TaskInfo
from acprof.models.inference_configuration import (
    configuration_identity,
    validate_frozen_configuration,
)
from acprof.tasks.workloads import parse_task

if TYPE_CHECKING:
    from acprof.host.run_state import RunState


def frozen_task(*, tokens: int = 7, generation: dict | None = None) -> TaskInfo:
    selected = parse_task("text-generation", {"max_new_tokens": tokens} if generation is None else generation,
                          encoder_decoder=False, source="fixture")
    task = TaskInfo(
        "fixture/frozen", "text-generation", "nlp", "transformers_model", "transformers",
        "a" * 40, "fixture", runtime_profile_id="nlp-cpu", model_adapter="family-default",
        model_config={"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]},
        selected_tasks=[selected],
    )
    configuration = {
        "schema_version": 1, "state": "frozen", "rule_version": "acprof-inference-v1",
        "model": {"source": "huggingface", "model_id": task.model_id, "revision": task.model_revision},
        "backend": {"id": task.runtime_backend, "adapter": task.model_adapter,
                    "extension_id": "torch-nlp", "runtime_profile": task.runtime_profile_id},
        "loading": {"off": {"dtype": "FP32", "attention_implementation": "sdpa"}},
        "tasks": {selected["task_id"]: {
            "interface": selected["interface"], "fingerprint": selected["fingerprint"],
            "generation": copy.deepcopy(selected["config"]["generation"]),
        }},
        "requests": {"loading.dtype": {"mode": "explicit", "value": "FP32"}},
        "decisions": [], "workload": {"batch_size": 1},
        "validation": {"static": "validated", "runtime": "not_run"},
    }
    configuration["identity_sha256"] = configuration_identity(configuration)
    validate_frozen_configuration(configuration)
    task.inference_configuration = configuration
    return task


def verified_runtime_report(task: TaskInfo) -> dict:
    digest = task.inference_configuration["identity_sha256"]
    generation = task.selected_tasks[0]["config"].get("generation", {}).copy()
    for total, new in (("max_length", "max_new_tokens"), ("min_length", "min_new_tokens")):
        if total in generation:
            generation.setdefault(new, None)
    return {
        "status": "ok", "device": "cpu", "inference_configuration_sha256": digest,
        "configuration_validation": {
            "status": "verified", "inference_configuration_sha256": digest,
            "task_id": task.selected_tasks[0]["task_id"], "device": "cpu",
            "loading": {
                "dtype": {"status": "verified", "expected": "FP32", "observed": ["FP32"],
                          "method": "floating_parameter_dtypes", "scope": "model_parameters"},
                "attention_implementation": {"status": "verified", "expected": "sdpa", "observed": ["sdpa"],
                                             "method": "model_config", "scope": "configured_attention"},
            },
        },
        "consumer_receipt": {
            "status": "verified", "inference_configuration_sha256": digest,
            "task_id": task.selected_tasks[0]["task_id"],
            "calls": [{"parameters": {
                name: {"status": "verified", "expected": copy.deepcopy(value), "observed": copy.deepcopy(value),
                       "source": "generate.kwargs"}
                for name, value in generation.items()
            }}],
        },
    }


def write_frozen_run(directory: Path, options: dict, task: TaskInfo) -> RunState:
    """Persist an interrupted run with its bound configuration and runtime evidence."""
    from acprof.artifacts import atomic_write_json
    from acprof.host.images.runtime import ImageInfo
    from acprof.host.input_plan import PlannedInputScales
    from acprof.host.input_plan.materialize import write_scale_plan_file
    from acprof.host.run_state import RunState

    state = RunState(directory, options, resume=False, project_dir=".")
    try:
        plan_path = state.layout.path("input_scale_plan.json")
        payload = {"text": "fixture", "params": task.selected_tasks[0]["config"].get("generation", {})}
        plan_sha = write_scale_plan_file(str(plan_path), task, [{"input_scale": 1, "payload": payload}])
        validation = {
            "schema_version": 1, "status": "ok",
            "inference_configuration_sha256": task.inference_configuration["identity_sha256"],
            "devices": {"off": verified_runtime_report(task)},
        }
        atomic_write_json(state.layout.path("runtime_validation.json"), validation)
        atomic_write_json(state.layout.path("static_meta.json"), {
            "schema_version": 8, "inference_configuration": task.inference_configuration,
            "runtime_validation": validation,
        })
        planned = PlannedInputScales([1.0], "fixture", plan_file=str(plan_path), plan_sha256=plan_sha)
        state.bind_runtime(task, ImageInfo("sha256:" + "b" * 64), planned, "", "")
    finally:
        state.close("interrupted")
    return state
