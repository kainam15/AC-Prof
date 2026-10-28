"""Portable, purpose-specific condition evidence for Python and offline reports.

Task and resource policy lives here. Consumers only compare recorded tokens and
exact rational distributions; missing evidence never becomes an equality proof.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterator
from fractions import Fraction

PURPOSES = ("same-hardware", "cross-hardware", "resource-scaling")
ALLOWED_RESOURCE_DIMENSIONS = ["cpu", "memory"]
_FIXED_OUTPUT_TASKS = {"tabular-regression", "tabular-classification", "time-series-forecasting"}
_DYNAMIC_GENERATION = {f"generation.{name}" for name in (
    "actual_output_tokens", "actual_output_tokens_per_sequence", "actual_output_tokens_status",
    "decoded_text_token_count", "stop_reason")}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def unknown(value):
    if value is None or value == "unknown":
        return True
    if isinstance(value, dict):
        return not value or any(unknown(item) for item in value.values())
    if isinstance(value, list):
        return not value or any(unknown(item) for item in value)
    return False


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


def workload_case_profile(case):
    variants = []
    distribution: defaultdict[str, Fraction] = defaultdict(Fraction)
    complete = True
    for variant in case["variants"]:
        contract = variant["contract"]
        weight = Fraction(variant["count"], case["request_count"])
        distribution[canonical(contract)] += weight
        fields = {name: canonical(value) for name, value in dimensions(contract).items() if not unknown(value)}
        controlled = [name for name in fields if contract.get("task") in _FIXED_OUTPUT_TASKS or
                      ((not name.startswith("output.") or name == "output.type") and name not in _DYNAMIC_GENERATION)]
        variants.append({"proportion": str(weight), "fields": fields, "controlled": sorted(controlled)})
        complete = complete and not unknown(contract)
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


class PreparedComparison:
    """Reuse derived evidence during one read-only snapshot's lifetime.

    The caller owns the snapshot and must not mutate it while preparing profiles.
    Returned profiles own their mutable values; no cache escapes into report data.
    """

    def __init__(self, snapshot: dict):
        self._snapshot = snapshot
        self._input_groups: dict[float, list[dict]] | None = None
        self._input_digests: dict[float, str | None] = {}
        self._workload_profiles: dict[str, dict] = {}
        self._fixed_checks: dict[str, tuple[tuple[str, str | None], ...]] = {}

    def _planned_inputs(self, scale):
        entries = self._snapshot.get("planned_input_entries")
        if not entries:
            return None
        if self._input_groups is None:
            groups: dict[float, list[dict]] = defaultdict(list)
            for entry in entries:
                key = float(entry["input_scale"])
                # Float equality never matches NaN, even the same object.
                if not math.isnan(key):
                    groups[key].append(entry)
            self._input_groups = groups
        key = float(scale)
        if math.isnan(key):
            return None
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
        return {**profile, "variants": [
            {**variant, "fields": dict(variant["fields"]), "controlled": list(variant["controlled"])}
            for variant in profile["variants"]]}

    def profile(self, *, purpose="same-hardware", case=None):
        """Project recorded evidence to a selected CSV case, without rereading files."""
        if purpose not in PURPOSES:
            raise ValueError(f"comparison purpose must be one of {PURPOSES}")
        snapshot = self._snapshot
        conditions = dict(snapshot["conditions"])
        hardware = dict(snapshot["hardware"])
        workload = conditions.pop("actual_workload")
        if case is not None:
            cpu, memory, gpu, scale = case
            from acprof.artifact_layout import resource_case_id
            case_id = resource_case_id(cpu, memory, gpu)
            conditions["resources"] = {**conditions["resources"], "cpus": str(cpu), "mems": str(memory), "gpus": gpu}
            conditions["planned_inputs"] = self._planned_inputs(scale)
            selected_case = canonical(list(case))
            workload = ({selected_case: workload[selected_case]} if workload and selected_case in workload else None)
            hardware = {name: {case_id: values.get(case_id)} if isinstance(values, dict) else None
                        for name, values in hardware.items()}
        if not unknown(hardware.get("runtime_threads")):
            conditions["runtime_threads"] = hardware["runtime_threads"]
        workload_profiles: dict[str, dict[str, object]] | None = None
        if workload is not None:
            workload_profiles = {}
            for key, item in workload.items():
                normalized_key = canonical(json.loads(key)[2:]) if purpose == "resource-scaling" else key
                profile = self._workload_profile(key, item)
                previous = workload_profiles.get(normalized_key)
                if previous and previous != profile:
                    profile = {"ambiguous": True}
                workload_profiles[normalized_key] = profile
        if purpose == "resource-scaling":
            # Resource labels in case maps are coordinates, not hardware facts.
            # Different measured policies/threads across resource cases stay visible.
            def remove_resource_keys(values):
                if not isinstance(values, dict):
                    return values
                by_device: dict[str, dict[str, object]] = {}
                for key, value in values.items():
                    by_device.setdefault(key.rsplit("_", 1)[-1], {})[canonical(value)] = value
                return {device: [items[key] for key in sorted(items)] for device, items in by_device.items()}
            hardware = {name: remove_resource_keys(values) for name, values in hardware.items()}
            if not unknown(snapshot["hardware"].get("runtime_threads")):
                conditions["runtime_threads"] = hardware["runtime_threads"]
        checks = {}

        def add(name, value, difference="incompatible"):
            for key, token in _check_tokens(name, value):
                checks[key] = {"value": token, "difference": difference}

        for name, value in conditions.items():
            if name == "resources" and purpose == "resource-scaling":
                for key, item in value.items():
                    add(f"resources.{key}", item, "expected_difference" if key in {"cpus", "mems"} else "incompatible")
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
                checks[f"hardware_{name}"] = {"value": None if unknown(value) else digest(value),
                                             "difference": "expected_difference"}
            else:
                add(f"hardware_{name}", value)
        profile = {"purpose": purpose, "valid": snapshot["valid"], "checks": checks, "workload": workload_profiles}
        verdict = compare_profiles(profile, profile)
        profile["status"] = verdict["status"]
        profile["reasons"] = verdict["reasons"]
        # Identical complete evidence forms an equivalence group. Expected differences
        # do not split it; incomplete evidence cannot create an automatic candidate.
        profile["cohort"] = digest({"checks": {key: item["value"] if item["difference"] != "expected_difference" else True
                                                for key, item in checks.items()}, "workload": workload_profiles})
        return profile


def compare_profiles(left, right):
    if left["purpose"] != right["purpose"]:
        raise ValueError("comparison profile purposes differ")
    statuses = {}
    for key in left["checks"].keys() | right["checks"].keys():
        a, b = left["checks"].get(key), right["checks"].get(key)
        statuses[key] = ("unknown" if not a or not b or a["value"] is None or b["value"] is None else
                         "compatible" if a["value"] == b["value"] else a["difference"])
    statuses["actual_workload"] = compare_workload_profiles(left["workload"], right["workload"])
    if not left["valid"] or not right["valid"]:
        statuses["invalid_artifacts"] = "incompatible"
    status = ("incompatible" if "incompatible" in statuses.values() else
              "unknown" if "unknown" in statuses.values() else "compatible")
    return {"status": status, "reasons": sorted({name.split(".")[0] for name, value in statuses.items()
                                                 if value in {"incompatible", "unknown"}}), "checks": statuses}
