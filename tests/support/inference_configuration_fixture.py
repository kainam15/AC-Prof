"""Offline frozen configuration evidence shared by host boundary tests."""
from __future__ import annotations

import copy

from acprof.host.detect import TaskInfo
from acprof.models.inference_configuration import (
    configuration_identity,
    validate_frozen_configuration,
)
from acprof.tasks.workloads import parse_task


def frozen_task(*, tokens: int = 7) -> TaskInfo:
    selected = parse_task("text-generation", {"max_new_tokens": tokens}, encoder_decoder=False, source="fixture")
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
                for name, value in task.selected_tasks[0]["config"].get("generation", {}).items()
            }}],
        },
    }
