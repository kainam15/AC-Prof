"""只读比较已有实验条件；不替代严格的续跑身份或模型质量验收。"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
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
    dimensions,
    hardware_case_binding,
    unknown,
    workload_case_profile,
    workload_contract_complete,
)
from acprof.analysis.workload import actual_workload, merge_workload_cases
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
from acprof.results.metrics import NUMERIC_FIELDS
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


def _condition(left, right) -> dict:
    if isinstance(left, dict) and isinstance(right, dict) and left and right:
        statuses = {_condition(left.get(key), right.get(key))["status"] for key in left.keys() | right.keys()}
        status = ("incompatible" if "incompatible" in statuses
                  else "unknown" if "unknown" in statuses else "compatible")
    elif unknown(left) or unknown(right):
        status = "unknown"
    else:
        status = "compatible" if left == right else "incompatible"
    return {"status": status, "left": left, "right": right}


def _contract_counts(case) -> Counter[str]:
    counts: Counter[str] = Counter()
    for variant in case["variants"]:
        counts[_canonical(variant["contract"])] += variant["count"]
    return counts


def _proportions(counts: Counter[str], total: int) -> dict[str, str]:
    return {value: str(Fraction(count, total)) for value, count in sorted(counts.items())}


def _same_proportions(left: Counter[str], right: Counter[str], left_total: int, right_total: int) -> bool:
    if left_total == right_total:
        return left == right
    return left.keys() == right.keys() and all(
        count * right_total == right[value] * left_total for value, count in left.items()
    )


def _marginal_counts(case) -> dict[str, Counter[str]]:
    histograms: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for variant in case["variants"]:
        for name, value in dimensions(variant["contract"]).items():
            histograms[name][_canonical(value)] += variant["count"]
    # Absent leaves and explicit JSON null share a bucket in the detailed report.
    for counts in histograms.values():
        absent = case["request_count"] - counts.total()
        if absent:
            counts["null"] += absent
    return histograms


def _changed_marginals(a, b):
    left, right = _marginal_counts(a), _marginal_counts(b)
    lhs_total, rhs_total = a["request_count"], b["request_count"]
    lhs_missing, rhs_missing = Counter({"null": lhs_total}), Counter({"null": rhs_total})
    changed = {}
    for name in left.keys() | right.keys():
        lhs, rhs = left.get(name, lhs_missing), right.get(name, rhs_missing)
        if not _same_proportions(lhs, rhs, lhs_total, rhs_total):
            changed[name] = {"left": _proportions(lhs, lhs_total), "right": _proportions(rhs, rhs_total)}
    return changed


def _request_counts_changed(left, right) -> bool | None:
    uncertain = left.keys() != right.keys()
    for key in left.keys() & right.keys():
        a, b = left[key], right[key]
        lhs, rhs = a["request_count"], b["request_count"]
        # An observed subtotal is a lower bound when any window lacks a count.
        if ((lhs is not None and b.get("observed_request_count", rhs) > lhs)
                or (rhs is not None and a.get("observed_request_count", lhs) > rhs)):
            return True
        if lhs is None or rhs is None:
            uncertain = True
    return None if uncertain else False


def _workload_differences(left, right):
    changed: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    joint_only = {}
    incomplete = False
    for key, a in left.items():
        b = right[key]
        if any(case.get("ambiguous") or not case["request_count"] or any(v["contract"] is None for v in case["variants"])
               for case in (a, b)):
            incomplete = True
            continue
        if any(not workload_contract_complete(v["contract"]) for case in (a, b) for v in case["variants"]):
            incomplete = True
        lhs, rhs = _contract_counts(a), _contract_counts(b)
        lhs_total, rhs_total = a["request_count"], b["request_count"]
        if _same_proportions(lhs, rhs, lhs_total, rhs_total):
            continue
        marginals = _changed_marginals(a, b)
        for name in sorted(marginals):
            changed.setdefault(name, {})[key] = marginals[name]
        if not marginals:
            joint_only[key] = {"left": _proportions(lhs, lhs_total), "right": _proportions(rhs, rhs_total)}
    if joint_only:
        changed.setdefault("joint_distribution", {}).update(joint_only)
    return incomplete, changed


def _workload_condition(left, right):
    result = {"status": "unknown", "left": left, "right": right,
              "request_counts_changed": None, "changed_dimensions": {},
              "reason": "actual_workload_evidence_missing"}
    if left is None or right is None:
        return result
    result["request_counts_changed"] = _request_counts_changed(left, right)
    if left.keys() != right.keys():
        return {**result, "status": "incompatible", "reason": "workload_cases_changed"}
    incomplete, changed = _workload_differences(left, right)
    if not changed:
        return {**result, "status": "unknown" if incomplete else "compatible",
                "reason": "actual_workload_evidence_incomplete" if incomplete else "same_per_request_distribution"}
    profile_status = compare_workload_profiles(*({key: workload_case_profile(case) for key, case in cases.items()}
                                                for cases in (left, right)))
    return {**result, "status": profile_status, "changed_dimensions": changed,
            "reason": ("per_request_distribution_changed" if profile_status == "incompatible" else
                       "actual_workload_evidence_incomplete" if incomplete else "output_distribution_equivalence_unverified")}


def _has_metric_value(value: object) -> bool:
    if value is None or (type(value) is str and value in MISSING_CSV_TEXT):
        return False
    return number(value) is not None


def _metric_availability(rows: list[dict[str, str]]) -> dict[str, bool]:
    return {name: any(_has_metric_value(row.get(name)) and metric_observation(row, name)["quality"] in USABLE_QUALITIES
                      for row in rows) for name in NUMERIC_FIELDS}


@dataclass(slots=True)
class _MetricEvidence:
    count: int = 0
    available: bool = False
    quality: str = "complete"
    signature: str | None = None

    def observe(self, item: dict) -> None:
        quality = item["quality"]
        if quality in USABLE_QUALITIES:
            self.available = True
            if quality != "complete" and self.quality == "complete":
                self.quality = "unknown"
        else:
            self.quality = "incomplete"
        if not self.count:
            self.signature = observation_signature(item)
        elif self.signature is not None and self.signature != observation_signature(item):
            self.signature = None
        self.count += 1


def _metric_evidence(rows: list[dict]) -> tuple[dict[str, bool], dict[str, dict]]:
    """Reduce finite observations once, retaining one contract summary per numeric field."""
    states = {name: _MetricEvidence() for name in NUMERIC_FIELDS}
    for row in rows:
        for name, value in row.items():
            state = states.get(name)
            if state is not None and _has_metric_value(value):
                state.observe(metric_observation(row, name))
    windows = len(rows)
    availability = {name: state.available for name, state in states.items()}
    contracts = {name: {"quality": state.quality if state.count else "unknown",
                        "all_windows": windows > 0 and state.count == windows,
                        "signature": state.signature} for name, state in states.items()}
    return availability, contracts


def _metric_contracts(rows: list[dict]) -> dict:
    return _metric_evidence(rows)[1]


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
) -> tuple[list[dict[str, str]], dict | None, dict[str, str | None]]:
    try:
        if not isinstance(result, ResultCsvSnapshot):
            raise result
        rows = [row for row in result.rows if not is_inferred_result(row)]
        resource_rows = {(row["cpu_cores"], row["mem_cap_gb"], row["gpu_mode"]): row
                         for row in rows if result_row_status(row) in {"ok", "warn"}}
        case_ids = dict(hardware_case_binding(*measurement_key(row)[:3]) for row in resource_rows.values())
    except (OSError, ValueError, TypeError, AttributeError, OverflowError) as error:
        issues.append(f"actual workload: {error}")
        return [], None, {}
    try:
        actual = actual_workload(rows)
    except ValueError as error:
        issues.append(f"actual workload: {error}")
        actual = None
    return rows, actual, case_ids


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
    rows, actual, case_ids = _resolve_actual_workload_and_cases(result, issues) if metadata_snapshot else ([], None, {})
    observed_hardware = {label: hardware_cases.get(case_id, {}) if case_id is not None else {}
                         for label, case_id in sorted(case_ids.items())}
    # Only the input payload and semantic conditions are compared. Model/export,
    # packages and backend hashes continue to belong to the existing run identity.
    host = _object_field(state, "host", issues)
    collector_protocol = _collector_protocol(metadata, issues)
    measured_rows = [row for row in rows if result_row_status(row) == "ok" and measurement_key(row)[4] == "0"]
    availability, contracts = _metric_evidence(measured_rows)
    formal_count = audit["counts"]["rows"] - audit["counts"]["warmup"]
    return {
        **recorded_identity(metadata),
        "inference_configuration_validation": _object_field(metadata, "runtime_validation", issues)
            if configuration else None,
        "metric_availability": availability,
        "metric_contracts": contracts,
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
            name: {label: record.get(name) for label, record in observed_hardware.items()} or None
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


@dataclass
class _AuditedInput:
    source: str | Path
    audit: dict
    evidence: dict
    metadata: ResultMetadataSnapshot | None
    result_file: tuple[Path, str] | None

    def verification_errors(self) -> Iterator[OSError | ValueError]:
        if self.metadata is not None:
            yield from self.metadata.verification_errors(self.source)
        if self.result_file is not None:
            path, digest = self.result_file
            try:
                verify_result_source(path, digest)
            except (OSError, ValueError, csv.Error) as exc:
                yield ValueError(f"{path}: {exc}")


def _audited_input(source: str | Path, *, raise_layout_errors: bool) -> _AuditedInput:
    audit, read, metadata = audit_with_metadata(source, verify_result_snapshot=False)
    if raise_layout_errors and metadata is None and isinstance(read, (ValueError, OSError)):
        raise read
    evidence = _comparison_evidence(metadata, audit, read)
    result_file = (read.path, read.sha256) if isinstance(read, ResultCsvSnapshot) else None
    return _AuditedInput(source, audit, evidence, metadata, result_file)


def _verified_comparison(
    left: str | Path, right: str | Path, purpose: str, *, raise_layout_errors: bool,
) -> tuple[dict, dict]:
    """Share the audit and release each row table before the final content checks."""
    if purpose not in PURPOSES:
        raise ValueError(f"comparison purpose must be one of {PURPOSES}")
    lhs = _audited_input(left, raise_layout_errors=raise_layout_errors)
    rhs = _audited_input(right, raise_layout_errors=raise_layout_errors)
    verdict = compare_snapshots(lhs.evidence, rhs.evidence, purpose=purpose)
    changed = False
    for side in (lhs, rhs):
        for error in side.verification_errors():
            changed = True
            message = str(error)
            side.audit["valid"] = side.evidence["valid"] = side.evidence["audit"]["valid"] = False
            side.audit["issues"].append({"code": "changing_snapshot", "message": message, "severity": "error"})
            side.evidence["issues"].append(message)
    if changed:
        verdict = compare_snapshots(lhs.evidence, rhs.evidence, purpose=purpose)
    return lhs.audit, verdict


def audit_and_compare_results(left: str | Path, right: str | Path, *, purpose: str = "same-hardware") -> dict:
    """Include unreadable layouts in the audit so the other side can still be inspected."""
    audit, verdict = _verified_comparison(left, right, purpose, raise_layout_errors=False)
    return {**audit, "comparison": verdict}


def compare_results(left: str | Path, right: str | Path, *, purpose: str = "same-hardware") -> dict:
    """Return compatible/incompatible/unknown for recorded comparison conditions.

    Quality constraints describe a shared target, not proof that either model
    meets it. Missing legacy evidence is unknown and never reconstructed.
    """
    _, verdict = _verified_comparison(left, right, purpose, raise_layout_errors=True)
    return verdict


def _resource_scaling_workload(cases: dict | None) -> dict[str, dict] | None:
    if cases is None:
        return None
    groups: dict[str, dict[str, dict]] = {}
    for key, case in cases.items():
        groups.setdefault(_canonical(json.loads(key)[2:]), {})[key] = case
    normalized = {}
    for projected, group in groups.items():
        cases_in_group = iter(group.values())
        first = next(cases_in_group)
        if len(group) == 1:
            normalized[projected] = first
            continue
        merged = merge_workload_cases(group)
        profile = workload_case_profile(first)
        if any(workload_case_profile(case) != profile for case in cases_in_group):
            merged["ambiguous"] = True
        normalized[projected] = merged
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
                  for name, value in lhs["conditions"].items() if name != "actual_workload"}
    left_workload, right_workload = (side["conditions"]["actual_workload"] for side in (lhs, rhs))
    if purpose == "resource-scaling":
        left_workload = _resource_scaling_workload(left_workload)
        right_workload = _resource_scaling_workload(right_workload)
    conditions["actual_workload"] = _workload_condition(left_workload, right_workload)
    for name in HARDWARE_FIELDS:
        left_value, right_value = lhs["hardware"].get(name), rhs["hardware"].get(name)
        item = _condition(left_value, right_value)
        if purpose == "cross-hardware" and name != "runtime_threads":
            # A known model difference cannot turn an unknown power policy into
            # verified evidence. Complete but different CPU-ID maps are expected.
            item["status"] = ("unknown" if unknown(left_value) or unknown(right_value)
                              else "compatible" if left_value == right_value else "expected_difference")
        conditions[f"hardware_{name}"] = item
    # Observed formal-server threads supersede independent-probe approximations.
    if not unknown(lhs["hardware"].get("runtime_threads")) and not unknown(rhs["hardware"].get("runtime_threads")):
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
