"""Portable, purpose-specific condition evidence for Python and offline reports.

Task and resource policy lives here. Consumers only compare recorded tokens and
exact rational distributions; missing evidence never becomes an equality proof.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator
from fractions import Fraction

from acprof.results.csv import numeric_coordinate_key

PURPOSES = ("same-hardware", "cross-hardware", "resource-scaling")
REPORT_PURPOSES = ("within-experiment", *PURPOSES)
ALLOWED_RESOURCE_DIMENSIONS = ["cpu", "memory"]
_FIXED_OUTPUT_TASKS = {"tabular-regression", "tabular-classification", "time-series-forecasting"}
_WORKLOAD_SECTIONS = {
    "scenario": {"type"},
    "input": {"planned_scale", "actual_scale"},
    "output": {"type", "shape", "count"},
}
_DYNAMIC_GENERATION = {f"generation.{name}" for name in (
    "actual_output_tokens", "actual_output_tokens_per_sequence", "actual_output_tokens_status",
    "decoded_text_token_count", "stop_reason")}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def hardware_case_binding(cpu: str, memory: str, gpu: str) -> tuple[str, str | None]:
    """Return the measured coordinate label and its optional recorded hardware case ID."""
    from acprof.results.layout import resource_case_id

    try:
        case_id = resource_case_id(cpu, memory, gpu)
    except ValueError:
        # CSV coordinates can be fractional. Keep them visible without using
        # their display label to look up observations from a collection case.
        return f"{cpu}c_{memory}g_{gpu}", None
    return case_id, case_id


def _verified_limits(allocation, key, runtime):
    from acprof.results.layout import resource_case_id

    method = "docker_cgroup_v2" if runtime == "docker" else "cgroup_v2"
    if not isinstance(allocation, dict) or allocation.get("method") != method or allocation.get("verified") is not True:
        return False
    values = allocation.get("observed" if runtime == "docker" else "requested", {})
    try:
        return "inherited" not in key and resource_case_id(
            values["cpu_quota"], values["memory_gib"], key.rsplit("_", 1)[-1],
        ) == key
    except (KeyError, ValueError, TypeError, OverflowError):
        return False


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def unknown(value) -> bool:
    """Stop at the first unknown fact while keeping one iterator per nested container."""
    pending = [iter((value,))]
    while pending:
        for item in pending[-1]:
            if item is None or item == "unknown":
                return True
            if isinstance(item, (dict, list)):
                if not item:
                    return True
                pending.append(iter(item.values() if isinstance(item, dict) else item))
                break
        else:
            pending.pop()
    return False


def effective_thread_evidence(conditions, hardware):
    """Prefer complete formal service observations over independently probed counts."""
    formal = hardware.get("runtime_threads")
    return conditions.get("runtime_threads") if unknown(formal) else formal


def dimensions(contract, prefix=""):
    result = {}
    for key, value in contract.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(dimensions(value, name))
        else:
            result[name] = value
    return result


def _check_tokens(name: str, value) -> Iterator[tuple[str, str | None]]:
    if isinstance(value, dict) and value:
        for key, item in value.items():
            yield from _check_tokens(f"{name}.{key}", item)
    else:
        yield name, None if unknown(value) else digest(value)


def workload_contract_complete(contract: dict | None) -> bool:
    """A v1 contract needs all base observations before equal distributions establish equivalence."""
    if contract is None:
        return False
    task = contract.get("task")
    if not isinstance(task, str) or not task.strip() or "batch_size" not in contract:
        return False
    for name, required in _WORKLOAD_SECTIONS.items():
        section = contract.get(name)
        if not isinstance(section, dict) or not required.issubset(section):
            return False
    return not unknown(contract)


def workload_case_profile(case):
    if case.get("ambiguous") or not case["request_count"]:
        return {"ambiguous": True}
    variants = []
    distribution: defaultdict[str, Fraction] = defaultdict(Fraction)
    complete = True
    for variant in case["variants"]:
        observation = variant["contract"]
        weight = Fraction(variant["count"], case["request_count"])
        distribution[canonical(observation)] += weight
        contract = observation if observation is not None else {}
        fields = {name: canonical(value) for name, value in dimensions(contract).items() if not unknown(value)}
        task = contract.get("task")
        fixed_output = isinstance(task, str) and task in _FIXED_OUTPUT_TASKS
        controlled = [name for name in fields if fixed_output or
                      ((not name.startswith("output.") or name == "output.type") and name not in _DYNAMIC_GENERATION)]
        variants.append({"proportion": str(weight), "fields": fields, "controlled": sorted(controlled)})
        complete = complete and workload_contract_complete(observation)
    return {"observed": digest({key: str(value) for key, value in sorted(distribution.items())}),
            "complete": complete, "variants": variants}


def compare_workload_profiles(left, right):
    if left is None or right is None:
        return "unknown"
    if left.keys() != right.keys():
        return "incompatible"
    status = "compatible"
    for key in left:
        a, b = left[key], right[key]
        if a.get("ambiguous") or b.get("ambiguous"):
            status = "unknown"
            continue
        variants = [*a["variants"], *b["variants"]]
        known = set.intersection(*(set(item["fields"]) for item in variants))

        def controlled_distribution(case):
            values: defaultdict[str, Fraction] = defaultdict(Fraction)
            for item in case["variants"]:
                fields = {name: item["fields"][name] for name in item["controlled"] if name in known}
                values[canonical(fields)] += Fraction(item["proportion"])
            return values

        if controlled_distribution(a) != controlled_distribution(b):
            return "incompatible"
        if not a["complete"] or not b["complete"] or a["observed"] != b["observed"]:
            status = "unknown"
    return status


def planned_input_scale_key(entry: dict) -> str:
    value = entry.get("exact_input_scale", entry["input_scale"])
    return numeric_coordinate_key("input_scale", str(value))


class PreparedComparison:
    """Reuse derived evidence during one read-only snapshot's lifetime.

    The caller owns the snapshot and must not mutate it while preparing profiles.
    Returned profiles own their mutable values; no cache escapes into report data.
    """

    def __init__(self, snapshot: dict):
        self._snapshot = snapshot
        self._input_groups: dict[str, list[dict]] | None = None
        self._input_digests: dict[str, str | None] = {}
        self._workload_profiles: dict[str, dict] = {}
        self._fixed_checks: dict[str, tuple[tuple[str, str | None], ...]] = {}

    def _planned_inputs(self, scale):
        entries = self._snapshot.get("planned_input_entries")
        if not entries:
            return None
        if self._input_groups is None:
            groups: dict[str, list[dict]] = defaultdict(list)
            for entry in entries:
                groups[planned_input_scale_key(entry)].append(entry)
            self._input_groups = groups
        key = numeric_coordinate_key("input_scale", str(scale))
        if key not in self._input_digests:
            selected = self._input_groups.get(key)
            self._input_digests[key] = digest(selected) if selected else None
        return self._input_digests[key]

    def _workload_profile(self, key, item):
        if key not in self._workload_profiles:
            self._workload_profiles[key] = workload_case_profile(item)
        profile = self._workload_profiles[key]
        # Evidence leaves are canonical strings or booleans. Copy their owning
        # containers so consumers cannot mutate cached or sibling profiles.
        if profile.get("ambiguous"):
            return dict(profile)
        return {**profile, "variants": [
            {**variant, "fields": dict(variant["fields"]), "controlled": list(variant["controlled"])}
            for variant in profile["variants"]]}

    def _project_case_evidence(self, conditions, hardware, workload, case):
        cpu, memory, gpu, scale = case
        label, case_id = hardware_case_binding(cpu, memory, gpu)
        conditions["resources"] = {
            **conditions["resources"], "cpus": str(cpu), "mems": str(memory), "gpus": gpu,
        }
        conditions["planned_inputs"] = self._planned_inputs(scale)
        selected_case = canonical(list(case))
        projected_workload = (
            {selected_case: workload[selected_case]}
            if workload and selected_case in workload else None
        )
        projected_hardware = {
            name: {label: values.get(case_id)} if case_id is not None and isinstance(values, dict) else None
            for name, values in hardware.items()
        }
        return projected_workload, projected_hardware

    def _build_workload_profiles(self, workload, purpose):
        if workload is None:
            return None
        workload_profiles: dict[str, dict[str, object]] = {}
        for key, item in workload.items():
            normalized_key = canonical(json.loads(key)[2:]) if purpose == "resource-scaling" else key
            profile = self._workload_profile(key, item)
            previous = workload_profiles.get(normalized_key)
            if previous and previous != profile:
                profile = {"ambiguous": True}
            workload_profiles[normalized_key] = profile
        return workload_profiles

    def _populate_condition_checks(self, checks, conditions, hardware, purpose):
        def add(name, value, difference="incompatible"):
            for key, token in _check_tokens(name, value):
                checks[key] = {"value": token, "difference": difference}

        for name, value in conditions.items():
            if name == "resources" and purpose == "resource-scaling":
                for key, item in value.items():
                    add(
                        f"resources.{key}",
                        item,
                        "expected_difference" if key in {"cpus", "mems"} else "incompatible",
                    )
            elif name in {"resources", "planned_inputs", "runtime_threads"}:
                add(name, value)
            else:
                # These source conditions are unchanged by case or purpose.
                # Retain immutable tokens and give every profile its own checks.
                if name not in self._fixed_checks:
                    self._fixed_checks[name] = tuple(_check_tokens(name, value))
                for key, token in self._fixed_checks[name]:
                    checks[key] = {"value": token, "difference": "incompatible"}
        for name, value in hardware.items():
            if purpose == "cross-hardware" and name != "runtime_threads":
                # Whole maps are allowed to differ, but every recorded leaf must exist.
                checks[f"hardware_{name}"] = {
                    "value": None if unknown(value) else digest(value),
                    "difference": "expected_difference",
                }
            else:
                add(f"hardware_{name}", value)

    def profile(self, *, purpose="same-hardware", case=None):
        """Project recorded evidence to a selected CSV case, without rereading files."""
        if purpose not in PURPOSES:
            raise ValueError(f"comparison purpose must be one of {PURPOSES}")
        snapshot = self._snapshot
        conditions = dict(snapshot["conditions"])
        hardware = dict(snapshot["hardware"])
        workload = conditions.pop("actual_workload")
        if case is not None:
            workload, hardware = self._project_case_evidence(conditions, hardware, workload, case)
        workload_profiles = self._build_workload_profiles(workload, purpose)
        if purpose == "resource-scaling":
            # Resource labels in case maps are coordinates, not hardware facts.
            # Different measured policies/threads across resource cases stay visible.
            hardware = {name: _remove_resource_keys(values) for name, values in hardware.items()}
        conditions["runtime_threads"] = effective_thread_evidence(conditions, hardware)
        checks: dict[str, dict[str, object]] = {}
        self._populate_condition_checks(checks, conditions, hardware, purpose)
        if case is None and snapshot.get("unmatched_input_scales"):
            # Retain the whole-plan digest so known plan disagreements remain visible.
            checks["planned_inputs.coverage"] = {"value": None, "difference": "incompatible"}
        profile = {"purpose": purpose, "valid": snapshot["valid"], "checks": checks, "workload": workload_profiles}
        if purpose == "same-hardware":
            from acprof.hub.repository import model_source
            from acprof.models.evidence import pinned_revision
            identity = snapshot.get("identity", {})
            runtime = conditions.get("execution_runtime", {}).get("kind")
            limits = snapshot.get("verified_limits", {})
            ids = hardware.get("host_id") or {}
            bound = bool(ids) and all(_verified_limits(limits.get(key, {}), key, runtime) for key in ids)
            protocol = snapshot.get("collector_protocol", {})
            if (runtime in {"docker", "process"} and bound
                    and all(identity.get(name) for name in ("model_name", "model_revision", "runtime_backend"))
                    and pinned_revision(identity["model_revision"]) and snapshot.get("model_profile")
                    and (snapshot.get("inference_configuration_validation") or {}).get("status") == "ok"
                    and snapshot.get("all_formal_measured")
                    and all(identity.get(key) for key in ("source_sha256", "packages_sha256"))
                    and set(protocol) == {"cpu_energy", "cpu_instructions", "packet_latency", "gpu_power"}):
                profile["runtime_metric_identity"] = digest([model_source(identity.get("model_source")), identity["model_name"],
                    identity["model_revision"], identity["runtime_backend"], snapshot["model_profile"], protocol,
                    identity["source_sha256"], identity["packages_sha256"], identity.get("runtime_requested_environment")])
                profile["runtime_kind"] = runtime
        verdict = compare_profiles(profile, profile)
        profile["status"] = verdict["status"]
        profile["reasons"] = verdict["reasons"]
        # Identical complete evidence forms an equivalence group. Expected differences
        # do not split it; incomplete evidence cannot create an automatic candidate.
        profile["cohort"] = digest({"checks": {key: item["value"] if item["difference"] != "expected_difference" else True
                                                for key, item in checks.items()}, "workload": workload_profiles})
        return profile


def _remove_resource_keys(values):
    if not isinstance(values, dict):
        return values
    by_device: dict[str, dict[str, object]] = {}
    for key, value in values.items():
        by_device.setdefault(key.rsplit("_", 1)[-1], {})[canonical(value)] = value
    return {device: [items[key] for key in sorted(items)] for device, items in by_device.items()}


def _has_formal_thread_evidence(checks):
    # Also works for saved profiles that only carry the original check tokens.
    values = [item["value"] for name, item in checks.items()
              if name == "hardware_runtime_threads" or name.startswith("hardware_runtime_threads.")]
    return bool(values) and all(value is not None for value in values)


def compare_profiles(left, right):
    if left["purpose"] != right["purpose"]:
        raise ValueError("comparison profile purposes differ")
    statuses = {}
    mixed_thread_sources = _has_formal_thread_evidence(left["checks"]) != _has_formal_thread_evidence(right["checks"])
    for key in left["checks"].keys() | right["checks"].keys():
        if mixed_thread_sources and (key == "runtime_threads" or key.startswith("runtime_threads.")):
            statuses[key] = "unknown"
            continue
        a, b = left["checks"].get(key), right["checks"].get(key)
        statuses[key] = ("unknown" if not a or not b or a["value"] is None or b["value"] is None else
                         "compatible" if a["value"] == b["value"] else a["difference"])
    statuses["actual_workload"] = compare_workload_profiles(left["workload"], right["workload"])
    if not left["valid"] or not right["valid"]:
        statuses["invalid_artifacts"] = "incompatible"
    status = ("incompatible" if "incompatible" in statuses.values() else
              "unknown" if "unknown" in statuses.values() else "compatible")
    runtime_match = (left["purpose"] == "same-hardware" and {left.get("runtime_kind"), right.get("runtime_kind")} == {"docker", "process"}
        and bool(left.get("runtime_metric_identity")) and left["runtime_metric_identity"] == right.get("runtime_metric_identity")
        and any(name.startswith("hardware_host_id") for name in statuses) and left["workload"] and right["workload"]
        and all(value == "compatible" for name, value in statuses.items() if not name.startswith("execution_runtime.")
                and name not in {"resources.cgroup_version", "resources.cgroup_collection_mode"}))
    return {**({"matched_cross_runtime": True} if runtime_match else {}), "status": status, "reasons": sorted({name.split(".")[0] for name, value in statuses.items()
                                                 if value in {"incompatible", "unknown"}}), "checks": statuses}
