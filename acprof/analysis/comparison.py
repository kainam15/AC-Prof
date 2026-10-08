"""只读比较已有实验条件；不替代严格的续跑身份或模型质量验收。"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

from acprof.analysis.audit import audit_result, number
from acprof.analysis.conditions import (
    ALLOWED_RESOURCE_DIMENSIONS,
    PURPOSES,
    compare_profiles,
    compare_workload_profiles,
    comparison_profile,
    workload_case_profile,
)
from acprof.artifacts import read_json_object
from acprof.hardware_conditions import HARDWARE_FIELDS, conditions_path
from acprof.metric_registry import METRICS
from acprof.platform import recorded_identity
from acprof.quality import QUALITY_FIELDS
from acprof.result_csv import measurement_key, read_result_csv
from acprof.runtime_settings import RUNTIME_ENV_NAMES

MEASUREMENT_OPTIONS = (
    "profiling_mode", "warmup", "repeat", "repeat_in_window", "repeat_window_seconds",
    "request_timeout_seconds", "sample_hz", "idle_seconds", "idle_cooldown_seconds",
    "compute_profile_tool", "execution_profile_tool", "prune_startup_oom",
    "matrix_order", "matrix_seed", "dram_energy",
)
RUNTIME_ENVIRONMENT = (set(RUNTIME_ENV_NAMES) - {"ACPROF_REQUEST_TIMEOUT_S"}) | {"OMP_NUM_THREADS", "MKL_NUM_THREADS"}


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _unknown(value) -> bool:
    if value is None or value == "unknown":
        return True
    if isinstance(value, dict):
        return not value or any(_unknown(item) for item in value.values())
    if isinstance(value, list):
        return not value or any(_unknown(item) for item in value)
    return False


def _condition(left, right) -> dict:
    if isinstance(left, dict) and isinstance(right, dict) and left and right:
        statuses = {_condition(left.get(key), right.get(key))["status"] for key in left.keys() | right.keys()}
        status = ("incompatible" if "incompatible" in statuses
                  else "unknown" if "unknown" in statuses else "compatible")
    elif _unknown(left) or _unknown(right):
        status = "unknown"
    else:
        status = "compatible" if left == right else "incompatible"
    return {"status": status, "left": left, "right": right}


def _actual_workload(rows: list[dict]) -> dict | None:
    """Pool observed request counts per case without losing the mixture weights."""
    facts: dict[str, Counter] = {}
    for row in rows:
        row_key = measurement_key(row)
        if row.get("status") not in {"ok", "warn"} or row_key[4] != "0":
            continue
        raw = row.get("workload_contract")
        if not raw or raw.lower() in {"nan", "null", "none"}:
            return None
        summary = json.loads(raw)
        if not isinstance(summary, dict) or summary.get("schema_version") != 1:
            return None
        variants, total = summary.get("variants"), summary.get("request_count")
        if not isinstance(variants, list) or not variants or type(total) is not int or total <= 0:
            return None
        counts: list[int] = []
        for variant in variants:
            count = variant.get("count") if isinstance(variant, dict) else None
            if type(count) is not int or count <= 0:
                return None
            counts.append(count)
        if sum(counts) != total:
            return None
        key = _canonical(row_key[:4])
        for variant in variants:
            contract = variant.get("contract")
            if not isinstance(contract, dict) or contract.get("schema_version") != 1:
                return None
            # Backend parameters are identity evidence, not workload semantics.
            contract = {name: item for name, item in contract.items() if name != "runtime"}
            facts.setdefault(key, Counter())[_canonical(contract)] += variant["count"]
    return {key: {"request_count": sum(counts.values()), "variants": [
        {"contract": json.loads(contract), "count": count,
         "proportion": str(Fraction(count, sum(counts.values())))}
        for contract, count in sorted(counts.items())]}
        for key, counts in sorted(facts.items())} or None


def _distribution(case, projection=lambda contract: contract):
    counts = Counter()
    for variant in case["variants"]:
        counts[_canonical(projection(variant["contract"]))] += variant["count"]
    total = case["request_count"]
    return {value: str(Fraction(count, total)) for value, count in sorted(counts.items())}


def _dimensions(contract, prefix=""):
    result = {}
    for key, value in contract.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_dimensions(value, name))
        else:
            result[name] = value
    return result


def _workload_condition(left, right):
    result = {"status": "unknown", "left": left, "right": right,
              "request_counts_changed": None, "changed_dimensions": {},
              "reason": "actual_workload_evidence_missing"}
    if left is None or right is None:
        return result
    if left.keys() != right.keys():
        return {**result, "status": "incompatible", "reason": "workload_cases_changed"}
    result["request_counts_changed"] = any(left[key]["request_count"] != right[key]["request_count"] for key in left)
    incomplete = any(_unknown(v["contract"]) for cases in (left, right) for case in cases.values() for v in case["variants"])
    changed = {}
    differing_cases = [key for key in left if _distribution(left[key]) != _distribution(right[key])]
    if not differing_cases:
        return {**result, "status": "unknown" if incomplete else "compatible",
                "reason": "actual_workload_evidence_incomplete" if incomplete else "same_per_request_distribution"}
    profile_status = compare_workload_profiles(*({key: workload_case_profile(case) for key, case in cases.items()}
                                                for cases in (left, right)))
    for key in differing_cases:
        a, b = left[key], right[key]
        names = set().union(*(_dimensions(v["contract"]).keys() for c in (a, b) for v in c["variants"]))
        for name in sorted(names):
            lhs = _distribution(a, lambda c, name=name: _dimensions(c).get(name))
            rhs = _distribution(b, lambda c, name=name: _dimensions(c).get(name))
            if lhs != rhs:
                changed.setdefault(name, {})[key] = {"left": lhs, "right": rhs}
    # Equal marginal distributions may still hide different joint mixtures.
    if not changed:
        changed["joint_distribution"] = {key: {"left": _distribution(left[key]), "right": _distribution(right[key])}
                                         for key in differing_cases}
    return {**result, "status": profile_status, "changed_dimensions": changed,
            "reason": ("per_request_distribution_changed" if profile_status == "incompatible" else
                       "actual_workload_evidence_incomplete" if incomplete else "output_distribution_equivalence_unverified")}


def load_comparison_snapshot(source: str | Path) -> dict:
    source = Path(source)
    directory = source if source.is_dir() else source.parent
    audit = audit_result(source)
    issues = []

    def read_json(name):
        try:
            from acprof.artifact_layout import ArtifactLayout
            layout = ArtifactLayout.discover(source) if source.is_dir() else ArtifactLayout.from_csv(source)
            path = layout.path(name)
            if not path.exists():
                return {}
            return read_json_object(path, label=name)
        except (OSError, ValueError) as error:
            issues.append(f"{name}: {error}")
            return {}

    state = read_json("run_state.json")
    metadata = read_json("static_meta.json")
    plan = read_json("input_scale_plan.json")
    from acprof.artifact_layout import ArtifactLayout
    layout = ArtifactLayout.discover(source) if source.is_dir() else ArtifactLayout.from_csv(source)
    hardware_payload = read_json(str(conditions_path(layout).relative_to(layout.root)))
    hardware_cases = hardware_payload.get("cases", {})
    if hardware_payload and (hardware_payload.get("schema_version") != 1 or not isinstance(hardware_cases, dict)):
        issues.append("hardware_conditions.json: unsupported schema")
        hardware_cases = {}
    for case_id, record in list(hardware_cases.items()):
        if not isinstance(record, dict):
            issues.append(f"hardware_conditions.json: invalid case {case_id}")
            hardware_cases[case_id] = {}

    def object_field(parent, name):
        value = parent.get(name, {})
        if not isinstance(value, dict):
            issues.append(f"{name}: 应为 JSON 对象")
            return {}
        return value

    options = object_field(state, "options")
    environment = object_field(options, "measurement_environment") if "measurement_environment" in options else None
    runtime_environment = {key: value for key, value in (environment or {}).items() if key in RUNTIME_ENVIRONMENT}
    measurement_environment = (_digest({key: value for key, value in environment.items()
                                       if key not in RUNTIME_ENVIRONMENT})
                               if isinstance(environment, dict) else None)
    entries = plan.get("entries")
    inputs = None
    if plan.get("schema_version") == 2 and isinstance(entries, list) and entries:
        if all(isinstance(entry, dict) and isinstance(entry.get("payload"), dict)
               and entry.get("input_scale") is not None for entry in entries):
            try:
                inputs = _digest([{key: entry[key] for key in ("input_scale", "payload")} for entry in entries])
            except (ValueError, TypeError) as error:
                issues.append(f"input_scale_plan.json: {error}")
    threads = {}
    validation = object_field(metadata, "runtime_validation")
    devices = object_field(validation, "devices")
    # The independent probe injects quota-derived TORCH_NUM_THREADS, while the
    # ordinary server preserves runtime defaults. Its effective count only
    # establishes a shared setting when this run explicitly requested threads.
    thread_names: tuple[str, ...] = (("ACPROF_ONNX_INTRA_OP_THREADS",) if metadata.get("runtime_backend") == "onnxruntime" else ())
    thread_names += ("ACPROF_RUNTIME_THREADS", "TORCH_NUM_THREADS")
    thread_request = next(((environment or {})[name] for name in thread_names if name in (environment or {})), None)
    try:
        explicit_threads = thread_request is not None and not isinstance(thread_request, bool) and int(thread_request) > 0
    except (TypeError, ValueError):
        explicit_threads = False
    for device in devices:
        result = object_field(devices, device)
        parameters = object_field(result, "runtime_parameters")
        effective = object_field(parameters, "effective")
        observed = effective.get("threads")
        threads[device] = (observed if explicit_threads and result.get("status") == "ok"
                           and type(observed) is int and observed > 0 else None)
    case_ids = set()
    rows = []
    try:
        _, rows = read_result_csv(directory / "result_all.csv" if source.is_dir() else source)
        actual = _actual_workload(rows)
        case_ids = {f"{int(float(row['cpu_cores']))}c_{int(float(row['mem_cap_gb']))}g_{row['gpu_mode']}"
                    for row in rows if row.get("status") in {"ok", "warn"}}

    except (OSError, ValueError, TypeError, AttributeError) as error:
        actual = None
        issues.append(f"actual workload: {error}")
    # Only the input payload and semantic conditions are compared. Model/export,
    # packages and backend hashes continue to belong to the existing run identity.
    host = object_field(state, "host")
    measured_rows = [row for row in rows if row.get("status") == "ok" and measurement_key(row)[4] == "0"]
    return {
        **recorded_identity(metadata),
        "metric_availability": {name: any(number(row.get(name)) is not None
                                           for row in measured_rows)
                                for name, metric in METRICS.items() if metric.kind == "number"},
        "run_id": state.get("run_id"), "result_csv": audit["result_csv"],
        "run_status": audit["run_status"], "measurement_status": audit["measurement_status"],
        "missing_metrics": audit["missing_metrics"],
        **{key: audit[key] for key in QUALITY_FIELDS},
        "valid": audit["valid"] and not issues, "issues": issues,
        "planned_input_entries": ([{key: entry[key] for key in ("input_scale", "payload")} for entry in entries]
                                  if inputs is not None else None),
        "conditions": {
            "comparability_class": recorded_identity(metadata)["comparability_class"],
            "task_semantics": {key: plan.get(key) for key in ("task_family", "pipeline_tag", "scenario")},
            "planned_inputs": inputs,
            "resources": {**{key: options.get(key) for key in ("cpus", "mems", "gpus", "batch_size")},
                          **{key: metadata.get(key) for key in ("cgroup_version", "cgroup_collection_mode")}},
            "runtime_threads": threads or None,
            "measurement_protocol": {**{key: options.get(key) for key in MEASUREMENT_OPTIONS},
                                     "measurement_environment": measurement_environment,
                                     "static_schema_version": metadata.get("schema_version")},
            "quality_constraints": plan.get("quality_constraints"),
            "actual_workload": actual,
        },
        "hardware": {
            name: {case_id: hardware_cases.get(case_id, {}).get(name) for case_id in sorted(case_ids)} or None
            for name in HARDWARE_FIELDS
        },
        "identity": {**{key: metadata.get(key) for key in (
            "model_name", "model_revision", "runtime_backend", "image_id", "runtime_environment")},
            "model_source": metadata.get("model_source", "huggingface"),
            "runtime_requested_environment": runtime_environment,
            **{key: host.get(key) for key in ("source_sha256", "packages_sha256")}},
    }


def compare_results(left: str | Path, right: str | Path, *, purpose: str = "same-hardware") -> dict:
    """Return compatible/incompatible/unknown for recorded comparison conditions.

    Quality constraints describe a shared target, not proof that either model
    meets it. Missing legacy evidence is unknown and never reconstructed.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"comparison purpose must be one of {PURPOSES}")
    snapshots = {"left": load_comparison_snapshot(left), "right": load_comparison_snapshot(right)}
    lhs, rhs = snapshots.values()
    conditions = {name: _condition(value, rhs["conditions"][name])
                  for name, value in lhs["conditions"].items()}
    conditions["actual_workload"] = _workload_condition(lhs["conditions"]["actual_workload"],
                                                        rhs["conditions"]["actual_workload"])
    if purpose == "resource-scaling":
        def resource_workload(cases):
            if cases is None:
                return None
            normalized = {}
            for key, value in cases.items():
                projected = _canonical(json.loads(key)[2:])
                if projected in normalized and workload_case_profile(normalized[projected]) != workload_case_profile(value):
                    return None
                normalized[projected] = value
            return normalized
        conditions["actual_workload"] = _workload_condition(resource_workload(lhs["conditions"]["actual_workload"]),
                                                            resource_workload(rhs["conditions"]["actual_workload"]))
    for name in HARDWARE_FIELDS:
        left_value, right_value = lhs["hardware"].get(name), rhs["hardware"].get(name)
        item = _condition(left_value, right_value)
        if purpose == "cross-hardware" and name != "runtime_threads":
            # A known model difference cannot turn an unknown power policy into
            # verified evidence. Complete but different CPU-ID maps are expected.
            item["status"] = ("unknown" if _unknown(left_value) or _unknown(right_value)
                              else "compatible" if left_value == right_value else "expected_difference")
        conditions[f"hardware_{name}"] = item
    # Observed formal-server threads supersede independent-probe approximations.
    if not _unknown(lhs["hardware"].get("runtime_threads")) and not _unknown(rhs["hardware"].get("runtime_threads")):
        conditions["runtime_threads"] = conditions["hardware_runtime_threads"]
    profiles = {side: comparison_profile(snapshot, purpose=purpose) for side, snapshot in snapshots.items()}
    qualification = compare_profiles(profiles["left"], profiles["right"])
    for name, item in conditions.items():
        statuses = {value for key, value in qualification["checks"].items() if key == name or key.startswith(name + ".")}
        item["status"] = ("incompatible" if "incompatible" in statuses else "unknown" if "unknown" in statuses else
                          "expected_difference" if "expected_difference" in statuses else "compatible")
    valid = lhs["valid"] and rhs["valid"]
    status = qualification["status"]
    differences = {name: {"left": value, "right": rhs["identity"][name]}
                   for name, value in lhs["identity"].items() if value != rhs["identity"][name]}
    environment_condition = conditions["comparability_class"]
    warnings = []
    if environment_condition["status"] != "compatible":
        warnings.append(f"Environment comparison blocked: {lhs['comparability_class']} vs "
                        f"{rhs['comparability_class']}. WSL measurements must not be treated as "
                        "native Linux measurements; unknown provenance cannot enter a Native baseline.")
    metric_comparability = {}
    for name in lhs["metric_availability"]:
        available = lhs["metric_availability"][name] and rhs["metric_availability"][name]
        metric_comparability[name] = {
            "status": "comparable" if status == "compatible" and available else "not comparable",
            "reason": ("missing_measurement" if not available else
                       "environment_" + environment_condition["status"]
                       if environment_condition["status"] != "compatible" else
                       "conditions_" + status if status != "compatible" else ""),
        }
    return {"schema_version": 2, "purpose": purpose, "status": status, "valid": valid,
            "allowed_resource_dimensions": ALLOWED_RESOURCE_DIMENSIONS if purpose == "resource-scaling" else [],
            "comparison_reasons": qualification["reasons"],
            "warnings": warnings, "metric_comparability": metric_comparability,
            "native_baseline_eligible": status == "compatible" and lhs["comparability_class"] == "native_linux",
            "scope": "recorded_comparison_conditions_not_model_quality_or_resume_identity",
            "limitations": ["hardware conditions are boundary observations, not proof of continuous isolation or identical thermal state",
                            "resource-scaling allows CPU and memory quotas; changing both cannot isolate either causal effect",
                            "missing hardware or effective thread evidence remains unknown; affinity does not prove exclusive CPU use",
                            "matching quality constraints do not prove either model meets them"],
            "conditions": conditions, "expected_differences": differences,
            "experiments": {side: {key: snapshot[key] for key in ("run_id", "result_csv", "valid", "issues", "environment_class",
                    "run_status", "measurement_status", "missing_metrics", *QUALITY_FIELDS)}
                            for side, snapshot in snapshots.items()}}
