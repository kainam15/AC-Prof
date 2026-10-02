"""One declarative dtype policy for preflight and model loading (no Torch import)."""

from __future__ import annotations

from acprof.extensions.schema import merge_template
from acprof.failures import Failure, RuntimeFailure

DTYPES = {"FP32": "float32", "FP16": "float16", "BF16": "bfloat16", "FP64": "float64"}


def resolve_precision(profile, extension, *, device, task, model_type="", version="", requested=None):
    device = "cpu" if device in {"cpu", "off"} else "gpu"
    policy = merge_template(profile.precision_policy, extension.precision_policy)
    overrides = (policy.get("task_overrides", {}).get(task, {}),
                 policy.get("model_type_overrides", {}).get(model_type, {}))
    policy = merge_template(policy, policy.get("device_overrides", {}).get(device, {}))
    for override in overrides:
        policy = merge_template(policy, override)
        policy = merge_template(policy, override.get("device_overrides", {}).get(device, {}))
    supported = [dtype for dtype in policy.get("supported_dtypes", extension.dtypes) if dtype in extension.dtypes]
    preferred = requested or policy.get("preferred_dtype")
    evidence = []
    blocked = set()
    for rule in policy.get("rules", []):
        if (rule.get("task", task) == task and rule.get("device", device) == device
                and model_type in rule.get("model_types", [model_type])
                and version in rule.get("transformers_versions", [version])):
            blocked.update(rule.get("blocked_dtypes", []))
            evidence.append(rule.get("evidence", {}))
    if preferred not in supported or preferred not in DTYPES or preferred in blocked:
        raise RuntimeFailure(Failure("precision_preflight", "precision_mismatch",
            f"dtype {preferred!r} is not supported for {task}/{model_type} on {device}", device,
            profile.profile_id, "after_configuration", {"requested_dtype": preferred, "supported_dtypes": supported,
            "blocked_dtypes": sorted(blocked), "transformers_version": version, "policy_evidence": evidence}))
    return {"dtype": preferred, "torch_dtype": DTYPES[preferred], "supported_dtypes": supported,
            "device": device, "runtime_profile": profile.profile_id, "evidence": evidence}


def remote_code_allowed(profile, extension) -> bool:
    return profile.trust_remote_code and extension.trust_remote_code is not False
