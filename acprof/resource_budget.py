"""Conservative planning evidence; a budget exclusion is never measured OOM."""

from collections.abc import Mapping

from acprof.failures import Failure


def _parameter_budget_count(count, dtype_counts, unnamed_elements, limit, evidence):
    if type(unnamed_elements) is not int or unnamed_elements < 0:
        raise ValueError("unnamed dtype element count must be a nonnegative integer")
    if unnamed_elements:
        evidence["unnamed_dtype_elements"] = unnamed_elements
    known_elements = 0
    if isinstance(dtype_counts, Mapping) and (dtype_counts or unnamed_elements):
        known_elements = sum(n for n in dtype_counts.values() if type(n) is int and n >= 0)
        evidence["known_dtype_elements"] = known_elements
    if type(count) is int and count <= limit and known_elements + unnamed_elements > count:
        # Conflicting evidence cannot establish that the model fits.
        return None
    return count


def assess_resource_budget(*, parameter_count=None, parameter_dtype_counts=None, unnamed_dtype_elements=0, max_parameters=None,
                           plan=None, max_download_bytes=None):
    for limit in (max_parameters, max_download_bytes):
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("resource budgets must be positive integer limits")
    selected = None
    if plan is not None:
        selected = plan.get("total_selected_bytes", plan.get("selected_bytes"))
    evidence = {"parameter_count": parameter_count, "max_parameters": max_parameters,
                "selected_artifact_bytes": selected, "max_download_bytes": max_download_bytes,
                "size_source": "selected_artifact_plan" if selected is not None else "unknown",
                "plan_sha256": (plan or {}).get("plan_sha256"), "measured_oom": False,
                "scope": "conservative_preflight; no_runtime_memory_claim"}
    if max_parameters is not None:
        parameter_count = _parameter_budget_count(parameter_count, parameter_dtype_counts, unnamed_dtype_elements,
                                                  max_parameters, evidence)
    reasons, unknown = [], []
    for observed, limit, reason in ((parameter_count, max_parameters, "parameter budget"),
                                   (selected, max_download_bytes, "selected artifact download budget")):
        if limit is None:
            continue
        if type(observed) is not int or observed < 0:
            unknown.append(reason)
        elif observed > limit:
            reasons.append(reason)
    evidence["unknown_budget_inputs"] = unknown
    if reasons or unknown:
        return {"status": "unverified", "evidence": evidence, "failure": Failure(
            "resource_preflight", "resource_limit", "; ".join(reasons + [f"unknown {name}" for name in unknown]),
            retryability="after_configuration" if unknown else "higher_budget", evidence=evidence).to_dict()}
    return {"status": "within_budget", "evidence": evidence}
