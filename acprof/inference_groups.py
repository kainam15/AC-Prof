"""Group authored semantic tasks by the computation they execute.

A task declaration describes a use case and its prompt. It does not authorize a
full resource matrix per prompt. The grouping signature contains only inference
operations that change the compute path; request lengths are workload axes.
"""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping

from acprof.model_evidence import content_digest
from acprof.task_workloads import (
    GENERATION_KEYS,
    GENERATIVE_INTERFACES,
    merge_generation_parameters,
    task_fingerprint,
    validate_generation,
)

# These options affect the execution algorithm, unlike prompt text, target
# language, task-specific token limits or beam scoring heuristics.
_SAMPLING_OPTIONS = ("temperature", "top_k", "top_p")
_TOKEN_OPTIONS = ("decoder_start_token_id", "bos_token_id", "eos_token_id",
                  "pad_token_id", "forced_bos_token_id", "forced_eos_token_id")
_DEFAULTS = {"num_beams": 1, "num_return_sequences": 1, "do_sample": False,
             "use_cache": True, "no_repeat_ngram_size": 0, "encoder_no_repeat_ngram_size": 0,
             "early_stopping": False}
# Standard output budget is a bound, not an assertion about actual output tokens.
_STANDARD_OUTPUT_TOKENS = 64


def inference_signature(task: dict, *, backend: str = "") -> dict:
    """Exclude task meaning and input/output lengths from execution identity."""
    interface = task["interface"]
    signature = {"interface": interface, "backend": backend}
    if interface in GENERATIVE_INTERFACES:
        original = task["config"].get("generation", {})
        compute = {key: original.get(key, value) for key, value in _DEFAULTS.items()}
        if compute["do_sample"]:
            compute.update({key: original[key] for key in _SAMPLING_OPTIONS if key in original})
        signature["generation"] = compute
    return signature


def _profile_generation(representative: dict, signature: dict) -> dict:
    """Use one explicit, length-bounded generation plan for this compute path."""
    observed = representative["config"].get("generation", {})
    generation = {"max_new_tokens": _STANDARD_OUTPUT_TOKENS, **signature["generation"]}
    # Special token IDs carry language/decoder semantics, not another compute
    # profile. Retain those of the chosen representative input only.
    generation.update({key: observed[key] for key in _TOKEN_OPTIONS if key in observed})
    return generation


def group_inference_tasks(tasks: list[dict], *, backend: str | Mapping[str, str] = "",
                          generation_override: dict | None = None) -> list[dict]:
    """Return one materializable workload per distinct inference signature.

    Keep all authored tasks as provenance, while choosing a deterministic prompt
    exemplar for each group. Selecting fewer semantic tasks changes the exemplar
    and provenance, not the number of profiles if their compute path is the same.
    """
    overrides = generation_override or {}
    if not isinstance(overrides, dict) or set(overrides) - GENERATION_KEYS:
        raise ValueError("inference generation override contains unsupported parameters")
    validate_generation(overrides)
    buckets: dict[str, list[dict]] = {}
    signatures: dict[str, dict] = {}
    for task in sorted(tasks, key=lambda value: value["task_id"]):
        if "execution_profile" in task:
            raise ValueError("cannot regroup an already materialized inference profile")
        effective = task
        if overrides and task["interface"] in GENERATIVE_INTERFACES:
            effective = copy.deepcopy(task)
            effective["config"]["generation"] = merge_generation_parameters(
                task["config"].get("generation", {}), overrides)
            validate_generation(effective["config"]["generation"])
        resolved_backend = backend.get(task["interface"], "") if isinstance(backend, Mapping) else backend
        signature = inference_signature(effective, backend=resolved_backend)
        digest = content_digest(signature)
        buckets.setdefault(digest, []).append(task)
        signatures[digest] = signature

    result = []
    for digest, members in sorted(buckets.items()):
        signature = signatures[digest]
        representative = copy.deepcopy(members[0])
        identifier = re.sub(r"[^a-z0-9]+", "-", signature["interface"].lower()).strip("-")
        representative["task_id"] = f"{identifier}-{digest[:12]}"
        if signature["interface"] in GENERATIVE_INTERFACES:
            representative["config"]["generation"] = _profile_generation(representative, signature)
        representative["semantic_tasks"] = copy.deepcopy(members)
        representative["execution_profile"] = {
            "signature": signature,
            "representative_task_id": members[0]["task_id"],
            "semantic_task_ids": [item["task_id"] for item in members],
            "output_length_policy": "standard_max_new_tokens_64" if
                                    signature["interface"] in GENERATIVE_INTERFACES else "not_applicable",
        }
        representative["fingerprint"] = task_fingerprint(representative)
        result.append(representative)
    return sorted(result, key=lambda value: value["task_id"])
