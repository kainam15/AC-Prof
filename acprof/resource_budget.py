"""Conservative planning evidence; a budget exclusion is never measured OOM."""

from acprof.failures import Failure


def assess_resource_budget(*, parameter_count=None, max_parameters=None, plan=None, max_download_bytes=None):
    selected = None
    if plan is not None:
        selected = plan.get("total_selected_bytes", plan.get("selected_bytes"))
    evidence = {"parameter_count": parameter_count, "max_parameters": max_parameters,
                "selected_artifact_bytes": selected, "max_download_bytes": max_download_bytes,
                "size_source": "selected_artifact_plan" if selected is not None else "unknown",
                "plan_sha256": (plan or {}).get("plan_sha256"), "measured_oom": False,
                "scope": "conservative_preflight; no_runtime_memory_claim"}
    reasons, unknown = [], []
    for observed, limit, reason in ((parameter_count, max_parameters, "parameter budget"),
                                   (selected, max_download_bytes, "selected artifact download budget")):
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("resource budgets must be positive integer limits")
        if type(observed) is int and limit is not None and observed > limit:
            reasons.append(reason)
        elif limit is not None and type(observed) is not int:
            unknown.append(reason)
    evidence["unknown_budget_inputs"] = unknown
    if reasons or unknown:
        return {"status": "unverified", "evidence": evidence, "failure": Failure(
            "resource_preflight", "resource_limit", "; ".join(reasons + [f"unknown {name}" for name in unknown]),
            retryability="after_configuration" if unknown else "higher_budget", evidence=evidence).to_dict()}
    return {"status": "within_budget", "evidence": evidence}
