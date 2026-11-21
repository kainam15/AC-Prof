"""只读比较已有实验条件；不替代严格的续跑身份或模型质量验收。"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from acprof.analysis.audit import AuditedRows, audit_result_snapshot, audit_with_metadata, number
from acprof.analysis.conditions import (
    ALLOWED_RESOURCE_DIMENSIONS,
    PURPOSES,
    PreparedComparison,
    compare_profiles,
    compare_workload_profiles,
    workload_case_profile,
)
from acprof.hardware_conditions import HARDWARE_FIELDS, conditions_path
from acprof.platform import recorded_identity
from acprof.quality import QUALITY_FIELDS
from acprof.results import is_inferred_result
from acprof.results.csv import (
    MISSING_CSV_TEXT,
    ResultCsvSnapshot,
    measurement_key,
    result_row_status,
    verify_result_source,
)
from acprof.results.layout import resolve_single_result_source
from acprof.results.metadata import ResultMetadataSnapshot
from acprof.results.metrics import METRICS
from acprof.results.profile import USABLE_QUALITIES, metric_observation, observation_signature
from acprof.runtime.settings import RUNTIME_ENV_NAMES

MEASUREMENT_OPTIONS = (
    "profiling_mode", "warmup", "repeat", "repeat_in_window", "repeat_window_seconds",
    "request_timeout_seconds", "sample_hz", "sampling_policy", "raw_sample_limit",
    "idle_seconds", "idle_cooldown_seconds",
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
        if result_row_status(row) not in {"ok", "warn"} or row_key[4] != "0":
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
    counts: Counter[str] = Counter()
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
    changed: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
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


def _metric_observations(rows: list[dict], name: str) -> Iterator[dict]:
    for row in rows:
        value = row.get(name)
        if value is None or (type(value) is str and value in MISSING_CSV_TEXT):
            continue
        if number(value) is not None:
            yield metric_observation(row, name)


def _metric_availability(rows: list[dict[str, str]]) -> dict[str, bool]:
    return {name: any(item['quality'] in USABLE_QUALITIES for item in _metric_observations(rows, name))
            for name, metric in METRICS.items() if metric.kind == "number"}


def _metric_contracts(rows: list[dict]) -> dict:
    contracts = {}
    for name, metric in METRICS.items():
        if metric.kind != 'number':
            continue
        observed = list(_metric_observations(rows, name))
        signatures = {observation_signature(item) for item in observed}
        contracts[name] = {'quality': 'incomplete' if any(item['quality'] not in USABLE_QUALITIES for item in observed)
                          else 'complete' if observed and all(item['quality'] == 'complete' for item in observed) else 'unknown',
                           'all_windows': bool(rows) and len(observed) == len(rows),
                           'signature': next(iter(signatures)) if len(signatures) == 1 else None}
    return contracts


def _read_snapshot_json(metadata_snapshot: ResultMetadataSnapshot, name: str, issues: list[str]) -> dict:
    try:
        if not metadata_snapshot.exists(name):
            return {}
        return metadata_snapshot.read(name).payload
    except (OSError, ValueError) as error:
        issues.append(f"{name}: {error}")
        return {}


def _object_field(parent: dict, name: str, issues: list[str]) -> dict:
    value = parent.get(name, {})
    if not isinstance(value, dict):
        issues.append(f"{name}: 应为 JSON 对象")
        return {}
    return value


def _collector_protocol(metadata: dict, issues: list[str]) -> dict:
    """Keep the measurement-affecting operations, without runtime-specific paths or PIDs."""
    plan = _object_field(metadata, "collection_plan", issues)
    probes = _object_field(plan, "probes", issues)
    protocol = {}
    for name in ("cpu_energy", "cpu_instructions", "packet_latency", "gpu_power"):
        if name not in probes:
            continue
        item = _object_field(probes, name, issues)
        if not isinstance(item.get("status"), str):
            continue
        selection: dict = {"enabled": item["status"] == "available"}
        if selection["enabled"] and name in {"cpu_instructions", "gpu_power"}:
            evidence = _object_field(item, "evidence", issues)
            if name == "cpu_instructions":
                events = evidence.get("events")
                if not isinstance(events, list) or not events or not all(isinstance(event, str) for event in events):
                    continue
                selection["events"] = sorted(events)
            else:
                counter = _object_field(evidence, "counter", issues)
                power = _object_field(evidence, "power", issues)
                if not counter.get("status") or not power.get("status"):
                    continue
                selection["methods"] = [key for key, probe in (("counter", counter), ("power", power))
                                        if probe["status"] == "available"]
        protocol[name] = selection
    return protocol


def _normalize_hardware_cases(hardware_payload: dict, issues: list[str]) -> dict[str, dict]:
    hardware_cases = hardware_payload.get("cases", {})
    if hardware_payload and (hardware_payload.get("schema_version") != 1 or not isinstance(hardware_cases, dict)):
        issues.append("hardware_conditions.json: unsupported schema")
        hardware_cases = {}
    hardware_cases = dict(hardware_cases)
    for case_id, record in list(hardware_cases.items()):
        if not isinstance(record, dict):
            issues.append(f"hardware_conditions.json: invalid case {case_id}")
            hardware_cases[case_id] = {}
    return hardware_cases


def _planned_inputs_from_plan(plan: dict, issues: list[str]) -> tuple[str | None, list[dict] | None]:
    entries = plan.get("entries")
    if not (plan.get("schema_version") == 2 and isinstance(entries, list) and entries):
        return None, None
    if not all(isinstance(entry, dict) and isinstance(entry.get("payload"), dict)
               and entry.get("input_scale") is not None for entry in entries):
        return None, None
    projected = [{key: entry[key] for key in ("input_scale", "payload")} for entry in entries]
    try:
        return _digest(projected), projected
    except (ValueError, TypeError) as error:
        issues.append(f"input_scale_plan.json: {error}")
        return None, None


def _observed_runtime_threads(
    metadata: dict,
    environment: dict | None,
    issues: list[str],
) -> dict[str, int | None]:
    threads: dict[str, int | None] = {}
    validation = _object_field(metadata, "runtime_validation", issues)
    devices = _object_field(validation, "devices", issues)
    # The independent probe injects quota-derived TORCH_NUM_THREADS, while the
    # ordinary server preserves runtime defaults. Its effective count only
    # establishes a shared setting when this run explicitly requested threads.
    thread_names: tuple[str, ...] = (
        ("ACPROF_ONNX_INTRA_OP_THREADS",) if metadata.get("runtime_backend") == "onnxruntime" else ()
    )
    thread_names += ("ACPROF_RUNTIME_THREADS", "TORCH_NUM_THREADS")
    env_map = environment or {}
    thread_request = next((env_map[name] for name in thread_names if name in env_map), None)
    try:
        explicit_threads = (
            thread_request is not None and not isinstance(thread_request, bool) and int(thread_request) > 0
        )
    except (TypeError, ValueError):
        explicit_threads = False
    for device in devices:
        result = _object_field(devices, device, issues)
        parameters = _object_field(result, "runtime_parameters", issues)
        effective = _object_field(parameters, "effective", issues)
        observed = effective.get("threads")
        threads[device] = (
            observed if explicit_threads and result.get("status") == "ok"
            and type(observed) is int and observed > 0 else None
        )
    return threads


def _resolve_actual_workload_and_cases(
    result: AuditedRows,
    issues: list[str],
) -> tuple[list[dict[str, str]], dict | None, set[str]]:
    try:
        if not isinstance(result, ResultCsvSnapshot):
            raise result
        rows = [row for row in result.rows if not is_inferred_result(row)]
        actual = _actual_workload(rows)
        from acprof.results.layout import resource_case_id
        case_ids = {
            resource_case_id(row["cpu_cores"], row["mem_cap_gb"], row["gpu_mode"])
            for row in rows if result_row_status(row) in {"ok", "warn"}
        }
        return rows, actual, case_ids
    except (OSError, ValueError, TypeError, AttributeError) as error:
        issues.append(f"actual workload: {error}")
        return [], None, set()


def load_comparison_snapshot(source: str | Path, *, result_snapshot: ResultCsvSnapshot | None = None,
                             metadata_snapshot: ResultMetadataSnapshot | None = None,
                             verify_result_snapshot: bool = True) -> dict:
    """Share parsed input while its outer owner retains the final content check."""
    source = resolve_single_result_source(source)
    metadata_snapshot = metadata_snapshot or ResultMetadataSnapshot(source)
    metadata_snapshot.validate_source(source)
    audit, result = audit_result_snapshot(source, result_snapshot=result_snapshot, metadata_snapshot=metadata_snapshot,
                                          verify_result_snapshot=verify_result_snapshot)
    return _comparison_evidence(metadata_snapshot, audit, result)


def _comparison_evidence(metadata_snapshot: ResultMetadataSnapshot | None, audit: dict, result: AuditedRows) -> dict:
    issues: list[str] = []
    if metadata_snapshot is None:
        state, metadata, plan, hardware_payload = {}, {}, {}, {}
        issues.extend(issue["message"] for issue in audit["issues"])
    else:
        layout = metadata_snapshot.layout
        state = _read_snapshot_json(metadata_snapshot, "run_state.json", issues)
        metadata = _read_snapshot_json(metadata_snapshot, "static_meta.json", issues)
        plan = _read_snapshot_json(metadata_snapshot, "input_scale_plan.json", issues)
        hardware_payload = _read_snapshot_json(
            metadata_snapshot, str(conditions_path(layout).relative_to(layout.root)), issues,
        )
    configuration = _object_field(metadata, "inference_configuration", issues)
    if configuration:
        from acprof.models.inference_configuration import validate_configuration_binding

        execution_environment = _object_field(metadata, "runtime_environment", issues)
        selected_task = plan.get("task_workload")
        recorded_task = SimpleNamespace(
            model_source=metadata.get("model_source", "huggingface"), model_id=metadata.get("model_name"),
            model_revision=metadata.get("model_revision"), runtime_backend=metadata.get("runtime_backend"),
            model_adapter=execution_environment.get("adapter"),
            runtime_profile_id=execution_environment.get("profile_id"),
            selected_tasks=[selected_task] if isinstance(selected_task, dict) else [],
        )
        try:
            validate_configuration_binding(configuration, recorded_task, allow_task_subset=True)
        except (ValueError, TypeError, KeyError) as exc:
            issues.append(f"inference_configuration: {exc}")
            configuration = {}
        else:
            if plan.get("inference_configuration_sha256") != configuration["identity_sha256"]:
                issues.append("input_scale_plan.json: inference configuration identity differs from static metadata")
    hardware_cases = _normalize_hardware_cases(hardware_payload, issues)

    options = _object_field(state, "options", issues)
    environment = (
        _object_field(options, "measurement_environment", issues)
        if "measurement_environment" in options else None
    )
    runtime_environment = {key: value for key, value in (environment or {}).items() if key in RUNTIME_ENVIRONMENT}
    measurement_environment = (_digest({key: value for key, value in environment.items()
                                       if key not in RUNTIME_ENVIRONMENT})
                               if isinstance(environment, dict) else None)
    inputs, planned_entries = _planned_inputs_from_plan(plan, issues)
    threads = _observed_runtime_threads(metadata, environment, issues)
    rows, actual, case_ids = _resolve_actual_workload_and_cases(result, issues) if metadata_snapshot else ([], None, set())
    # Only the input payload and semantic conditions are compared. Model/export,
    # packages and backend hashes continue to belong to the existing run identity.
    host = _object_field(state, "host", issues)
    collector_protocol = _collector_protocol(metadata, issues)
    measured_rows = [row for row in rows if result_row_status(row) == "ok" and measurement_key(row)[4] == "0"]
    formal_count = audit["counts"]["rows"] - audit["counts"]["warmup"]
    return {
        **recorded_identity(metadata),
        "inference_configuration_validation": _object_field(metadata, "runtime_validation", issues)
            if configuration else None,
        "metric_availability": _metric_availability(measured_rows),
        "metric_contracts": _metric_contracts(measured_rows),
        "run_id": state.get("run_id"), "result_csv": audit["result_csv"],
        "run_status": audit["run_status"], "measurement_status": audit["measurement_status"],
        "missing_metrics": audit["missing_metrics"],
        "audit": {key: audit[key] for key in ("valid", "issues", "counts", "coverage", "completion")},
        **{key: audit[key] for key in QUALITY_FIELDS},
        "valid": audit["valid"] and not issues, "issues": issues,
        "planned_input_entries": planned_entries,
        "all_formal_measured": bool(measured_rows) and len(measured_rows) == formal_count,
        "verified_limits": {key: record.get("resource_plan", {}) for key, record in hardware_cases.items()},
        "collector_protocol": collector_protocol,
        "model_profile": (_digest({**{key: configuration[key] for key in ("model", "backend", "loading", "tasks")},
                                   "packages": metadata["runtime_environment"]["packages"]})
                          if configuration and isinstance(metadata.get("runtime_environment"), dict)
                          and metadata["runtime_environment"].get("packages") else None),
        "conditions": {
            "execution_runtime": {"kind": metadata.get("runtime_kind", "docker"),
                                  "resource_scope": "observed_process_tree" if metadata.get("runtime_kind") == "process" else "container_cgroup",
                                  **({"allocation": options.get("process_allocation"),
                                      "gpu_visibility": options.get("process_gpu")}
                                     if metadata.get("runtime_kind") == "process" else {})} if metadata_snapshot else {},
            "comparability_class": recorded_identity(metadata)["comparability_class"],
            "task_semantics": {**{key: plan.get(key) for key in ("task_family", "pipeline_tag", "scenario")},
                               **({"task_workload": {key: plan["task_workload"].get(key)
                                                     for key in ("task_id", "interface", "fingerprint")}}
                                  if plan.get("task_workload") else {})},
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
            "model_name", "model_revision", "runtime_backend", "image_id", "runtime_environment", "runtime_kind", "runtime_id")},
            "model_source": metadata.get("model_source", "huggingface") if metadata_snapshot else None,
            "inference_configuration_sha256": configuration.get("identity_sha256"),
            "inference_loading": configuration.get("loading"),
            "inference_generation": {
                task_id: item["generation"] for task_id, item in configuration.get("tasks", {}).items()
            } if configuration else None,
            "inference_parameters": {
                task_id: item.get("inference", {}) for task_id, item in configuration.get("tasks", {}).items()
            } if configuration else None,
            "runtime_requested_environment": runtime_environment,
            **{key: host.get(key) for key in ("source_sha256", "packages_sha256")}},
    }


def _audited_input(source: str | Path) -> tuple[dict, dict, dict[Path, str | None]]:
    audit, read, metadata = audit_with_metadata(source, verify_result_snapshot=False)
    evidence = _comparison_evidence(metadata, audit, read)
    fingerprints = metadata.observed_fingerprints() if metadata is not None else {}
    if isinstance(read, ResultCsvSnapshot):
        fingerprints[read.path] = read.sha256
    return audit, evidence, fingerprints


def audit_and_compare_results(left: str | Path, right: str | Path, *, purpose: str = "same-hardware") -> dict:
    """Share the audit and release each row table before the final content checks."""
    if purpose not in PURPOSES:
        raise ValueError(f"comparison purpose must be one of {PURPOSES}")
    left_audit, lhs, left_files = _audited_input(left)
    right_audit, rhs, right_files = _audited_input(right)
    for audit, evidence, fingerprints in ((left_audit, lhs, left_files), (right_audit, rhs, right_files)):
        for path, digest in fingerprints.items():
            try:
                verify_result_source(path, digest)
            except (OSError, ValueError, csv.Error) as exc:
                message = f"{path}: {exc}"
                audit["valid"] = evidence["valid"] = evidence["audit"]["valid"] = False
                audit["issues"].append({"code": "changing_snapshot", "message": message, "severity": "error"})
                evidence["issues"].append(message)
    return {**left_audit, "comparison": compare_snapshots(lhs, rhs, purpose=purpose)}


def compare_results(left: str | Path, right: str | Path, *, purpose: str = "same-hardware") -> dict:
    """Return compatible/incompatible/unknown for recorded comparison conditions.

    Quality constraints describe a shared target, not proof that either model
    meets it. Missing legacy evidence is unknown and never reconstructed.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"comparison purpose must be one of {PURPOSES}")
    return compare_snapshots(load_comparison_snapshot(left), load_comparison_snapshot(right), purpose=purpose)


def _resource_scaling_workload(cases: dict | None) -> dict[str, dict] | None:
    if cases is None:
        return None
    normalized: dict[str, dict] = {}
    for key, value in cases.items():
        projected = _canonical(json.loads(key)[2:])
        if projected in normalized and workload_case_profile(normalized[projected]) != workload_case_profile(value):
            return None
        normalized[projected] = value
    return normalized


def _metric_comparability_map(
    lhs: dict,
    rhs: dict,
    status: str,
    environment_status: str,
    *, cross_runtime: bool = False,
) -> dict[str, dict[str, str]]:
    metric_comparability = {}
    for name in lhs["metric_availability"]:
        available = lhs["metric_availability"][name] and rhs["metric_availability"][name]
        contracts = [snapshot.get('metric_contracts', {}).get(name) for snapshot in (lhs, rhs)]
        contract_reason = ''
        if all(contracts):
            if any(item['quality'] == 'incomplete' for item in contracts):
                contract_reason = 'incomplete_metric_quality'
            elif any(item['signature'] is None for item in contracts):
                contract_reason = 'mixed_or_missing_metric_contract'
            elif contracts[0]['signature'] != contracts[1]['signature']:
                contract_reason = 'different_source_scope_window_or_sampling'
        if cross_runtime and not contract_reason and available:
            from acprof.results.profile import common_runtime_metric
            if common_runtime_metric(name) and all(item and item["quality"] == "complete" and item.get("all_windows") for item in contracts):
                metric_comparability[name] = {"status": "descriptive", "reason": "matched_cross_runtime_metric"}
                continue
        metric_comparability[name] = {
            "status": "comparable" if status == "compatible" and available and not contract_reason else "not comparable",
            "reason": (contract_reason or "missing_measurement" if not available or contract_reason else
                       "environment_" + environment_status
                       if environment_status != "compatible" else
                       "conditions_" + status if status != "compatible" else ""),
        }
    return metric_comparability


def compare_snapshots(left: dict, right: dict, *, purpose: str = "same-hardware") -> dict:
    """Compare loaded evidence without reading files or mutating the snapshots."""
    if purpose not in PURPOSES:
        raise ValueError(f"comparison purpose must be one of {PURPOSES}")
    snapshots = {"left": left, "right": right}
    lhs, rhs = snapshots.values()
    conditions = {name: _condition(value, rhs["conditions"][name])
                  for name, value in lhs["conditions"].items()}
    conditions["actual_workload"] = _workload_condition(lhs["conditions"]["actual_workload"],
                                                        rhs["conditions"]["actual_workload"])
    if purpose == "resource-scaling":
        conditions["actual_workload"] = _workload_condition(
            _resource_scaling_workload(lhs["conditions"]["actual_workload"]),
            _resource_scaling_workload(rhs["conditions"]["actual_workload"]),
        )
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
    profiles = {side: PreparedComparison(snapshot).profile(purpose=purpose) for side, snapshot in snapshots.items()}
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
    metric_comparability = _metric_comparability_map(
        lhs, rhs, status, environment_condition["status"], cross_runtime=qualification.get("matched_cross_runtime", False),
    )
    return {"schema_version": 2, "purpose": purpose, "status": status, "valid": valid,
            "allowed_resource_dimensions": ALLOWED_RESOURCE_DIMENSIONS if purpose == "resource-scaling" else [],
            "comparison_reasons": qualification["reasons"],
            "warnings": warnings, "metric_comparability": metric_comparability,
            "native_baseline_eligible": (status == "compatible" and lhs["comparability_class"] == "native_linux"
                                         and lhs["conditions"]["execution_runtime"]["kind"] == "docker"),
            "scope": "recorded_comparison_conditions_not_model_quality_or_resume_identity",
            "limitations": ["hardware conditions are boundary observations, not proof of continuous isolation or identical thermal state",
                            "resource-scaling allows CPU and memory quotas; changing both cannot isolate either causal effect",
                            "missing hardware or effective thread evidence remains unknown; affinity does not prove exclusive CPU use",
                            "matching quality constraints do not prove either model meets them"],
            "conditions": conditions, "expected_differences": differences,
            "experiments": {side: {key: snapshot[key] for key in ("run_id", "result_csv", "valid", "issues", "environment_class",
                    "run_status", "measurement_status", "missing_metrics", *QUALITY_FIELDS)}
                            for side, snapshot in snapshots.items()}}
