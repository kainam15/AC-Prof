"""Comparison preparation is bounded by source evidence, not output purposes."""
import copy
import csv
import hashlib
import json
from unittest.mock import Mock

import pytest

from acprof.analysis import conditions
from acprof.analysis.comparison import compare_results, compare_snapshots, load_comparison_snapshot
from acprof.analysis.model import load_analysis
from tests.support.comparison_fixtures import ComparisonFixture


@pytest.fixture
def multi_case_sources(tmp_path):
    def build(cpu_count, *, update_workload=None):
        fixture = ComparisonFixture()
        fixture.build(tmp_path)
        for directory in (fixture.left, fixture.right):
            with (directory / "result_all.csv").open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields, row = reader.fieldnames, next(reader)
            rows = []
            for cpu in range(1, cpu_count + 1):
                for scale in (2, 4):
                    contract = copy.deepcopy(fixture.contract)
                    contract["input"].update(planned_scale=scale, actual_scale=scale, rows=scale)
                    contract["output"].update(shape=[scale, 1], count=scale)
                    summary = {"schema_version": 1, "request_count": 3, "variants": [{"count": 3, "contract": contract}]}
                    if update_workload is not None:
                        summary = update_workload(directory, cpu, scale, summary)
                    rows.append({**row, "cpu_cores": cpu, "input_scale": scale,
                                 "workload_contract": json.dumps(summary) if summary is not None else "N/A"})
            with (directory / "result_all.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fields)
                writer.writeheader()
                writer.writerows(rows)
            fixture.change_json(directory, "run_state.json", lambda state: (
                state["options"].update(cpus=",".join(map(str, range(1, cpu_count + 1)))),
                state["runtime"]["planned"].update(scales=[2, 4])))
            fixture.change_json(directory, "input_scale_plan.json", lambda plan: plan.update(entries=[
                {"input_scale": scale, "payload": {"features": [[1, 2]] * scale, "batch_size": 1}}
                for scale in (2, 4)]))
            fixture.change_json(directory, "hardware_conditions.json", lambda data: data.update(cases={
                f"{cpu}c_4g_off": data["cases"]["1c_4g_off"] for cpu in range(1, cpu_count + 1)}))
            fixture.refresh(directory)
        return fixture

    return build


@pytest.mark.parametrize(("sibling_diff", "status"), [(False, "unknown"), (True, "incompatible")])
def test_resource_scaling_keeps_sibling_workload_evidence(multi_case_sources, sibling_diff, status):
    def different_work(directory, cpu, scale, summary):
        if directory.name == "onnx":
            if cpu == 2 and scale == 2:
                return None
            if sibling_diff and scale == 4:
                summary["variants"][0]["contract"]["input"]["feature_dim"] = 8
        return summary

    fixture = multi_case_sources(2, update_workload=different_work)
    purpose = "resource-scaling"
    report = compare_results(fixture.left, fixture.right, purpose=purpose)
    compact = conditions.compare_profiles(*(
        conditions.PreparedComparison(load_comparison_snapshot(path)).profile(purpose=purpose)
        for path in (fixture.left, fixture.right)))
    workload = report["conditions"]["actual_workload"]
    assert report["valid"]
    assert report["status"] == compact["status"] == status
    assert workload["status"] == compact["checks"]["actual_workload"] == status

    partial = workload["right"]['["off","2"]']
    assert partial["ambiguous"] is True
    assert partial["request_count"] is None
    assert partial["observed_request_count"] == 3
    assert partial["uncounted_windows"] == 1
    assert partial["variants"][0]["proportion"] is None
    assert {key: case["request_count"] for key, case in partial["resource_cases"].items()} == {
        '["1","4","off","2"]': 3, '["2","4","off","2"]': None}
    sibling = '["off","4"]'
    assert workload["left"][sibling]["request_count"] == workload["right"][sibling]["request_count"] == 6
    if sibling_diff:
        assert workload["changed_dimensions"] == {"input.feature_dim": {sibling: {"left": {"2": "1"}, "right": {"8": "1"}}}}
        assert workload["reason"] == "per_request_distribution_changed"
    else:
        assert workload["changed_dimensions"] == {}
        assert workload["reason"] == "actual_workload_evidence_incomplete"


def test_resource_scaling_pools_matching_distributions_across_resources(multi_case_sources):
    def repeated_work(directory, cpu, _scale, summary):
        multiplier = 1 if directory.name == "torch" and cpu == 1 else 2
        contract = summary["variants"][0]["contract"]
        other = copy.deepcopy(contract)
        other["input"]["feature_dim"] = 8
        summary.update(request_count=3 * multiplier, variants=[
            {"count": 2 * multiplier, "contract": contract}, {"count": multiplier, "contract": other}])
        return summary

    fixture = multi_case_sources(2, update_workload=repeated_work)
    report = compare_results(fixture.left, fixture.right, purpose="resource-scaling")
    workload = report["conditions"]["actual_workload"]
    assert report["status"] == "compatible"
    assert workload["status"] == "compatible"
    assert workload["request_counts_changed"] is True
    for scale in ("2", "4"):
        key = conditions.canonical(["off", scale])
        lhs, rhs = workload["left"][key], workload["right"][key]
        assert lhs["request_count"] == 9
        assert rhs["request_count"] == 12
        assert {variant["contract"]["input"]["feature_dim"]: (variant["count"], variant["proportion"])
                for variant in lhs["variants"]} == {2: (6, "2/3"), 8: (3, "1/3")}
        assert {key: case["request_count"] for key, case in lhs["resource_cases"].items()} == {
            conditions.canonical(["1", "4", "off", scale]): 3, conditions.canonical(["2", "4", "off", scale]): 6}


@pytest.mark.parametrize("cpu_count", [1, 4])
def test_analysis_prepares_each_input_scale_and_workload_case_once(multi_case_sources, monkeypatch, cpu_count):
    fixture = multi_case_sources(cpu_count)
    input_digests = []
    original_digest = conditions.digest

    def record_digest(value):
        if isinstance(value, list) and value and all(isinstance(item, dict) and "payload" in item for item in value):
            input_digests.append(value)
        return original_digest(value)

    workloads = Mock(wraps=conditions.workload_case_profile)
    monkeypatch.setattr(conditions, "digest", record_digest)
    monkeypatch.setattr(conditions, "workload_case_profile", workloads)

    model = load_analysis([fixture.left, fixture.right])

    assert len(model.configs) == 4 * cpu_count
    assert all(profile["status"] == "compatible" for config in model.configs
               for profile in config["comparison_profiles"].values())
    assert {"input_digests": len(input_digests), "workloads": workloads.call_count} == {
        "input_digests": 4, "workloads": 4 * cpu_count,
    }


def test_separate_sources_and_later_loads_keep_distinct_input_evidence(multi_case_sources):
    fixture = multi_case_sources(1)

    def profiles():
        return [config["comparison_profiles"]["resource-scaling"]
                for config in load_analysis([fixture.left, fixture.right]).configs]

    first = profiles()
    assert conditions.compare_profiles(first[0], first[2])["status"] == "compatible"
    fixture.change_json(fixture.right, "input_scale_plan.json", lambda plan:
                        plan["entries"][0]["payload"].update(features=[[9, 9], [8, 8]]))
    later = profiles()
    assert conditions.compare_profiles(later[0], later[2])["reasons"] == ["planned_inputs"]
    assert conditions.compare_profiles(later[1], later[3])["status"] == "compatible"
    assert first[0] == later[0]


def test_returned_workload_profiles_do_not_share_mutable_values(multi_case_sources):
    fixture = multi_case_sources(1)
    model = load_analysis([fixture.left])
    profiles = model.configs[0]["comparison_profiles"]
    before = copy.deepcopy(profiles["cross-hardware"])
    workload = next(iter(profiles["same-hardware"]["workload"].values()))
    workload["variants"][0]["fields"]["input.feature_dim"] = "999"
    workload["variants"][0]["controlled"].clear()
    assert profiles["cross-hardware"] == before


def test_input_index_keeps_equivalent_coordinates_raw_values_and_entry_order(multi_case_sources):
    fixture = multi_case_sources(1)
    snapshot = load_comparison_snapshot(fixture.left)
    entries = [
        {"input_scale": "2.0", "payload": {"value": "first"}},
        {"input_scale": 4, "payload": {"value": "other"}},
        {"input_scale": 2, "payload": {"value": "second"}},
    ]
    snapshot["planned_input_entries"] = entries
    prepared = conditions.PreparedComparison(snapshot)
    expected_json = '[{"input_scale":"2.0","payload":{"value":"first"}},{"input_scale":2,"payload":{"value":"second"}}]'
    input_digest = hashlib.sha256(expected_json.encode()).hexdigest()
    expected_token = hashlib.sha256(json.dumps(input_digest).encode()).hexdigest()

    for purpose in conditions.PURPOSES:
        profile = prepared.profile(purpose=purpose, case=("1", "4", "off", "2"))
        assert profile["checks"]["planned_inputs"]["value"] == expected_token
    assert prepared.profile(case=("1", "4", "off", "9"))["checks"]["planned_inputs"]["value"] is None
    assert snapshot["planned_input_entries"] == entries


def test_input_preparation_remains_lazy_and_rejects_invalid_selected_evidence(multi_case_sources):
    fixture = multi_case_sources(1)
    snapshot = load_comparison_snapshot(fixture.left)
    snapshot["planned_input_entries"].append({"input_scale": 9, "payload": {"unserializable": object()}})
    prepared = conditions.PreparedComparison(snapshot)
    assert prepared.profile(case=("1", "4", "off", "2"))["status"] == "compatible"
    with pytest.raises(TypeError, match="not JSON serializable"):
        prepared.profile(case=("1", "4", "off", "9"))

    snapshot["planned_input_entries"].append({"input_scale": "bad-scale", "payload": {}})
    prepared = conditions.PreparedComparison(snapshot)
    assert prepared.profile()["status"] == "compatible"
    for _ in range(2):
        with pytest.raises(ValueError, match="invalid input_scale"):
            prepared.profile(case=("1", "4", "off", "2"))


def test_prepared_profiles_preserve_snapshot_and_protect_cached_workload(multi_case_sources):
    fixture = multi_case_sources(1)
    snapshot = load_comparison_snapshot(fixture.left)
    before = copy.deepcopy(snapshot)
    prepared = conditions.PreparedComparison(snapshot)
    profile = prepared.profile(case=("1", "4", "off", "2"))
    sibling = prepared.profile(case=("1", "4", "off", "2"))

    def mutable_ids(value):
        if not isinstance(value, (dict, list)):
            return set()
        children = value.values() if isinstance(value, dict) else value
        return {id(value)}.union(*(mutable_ids(child) for child in children))

    assert mutable_ids(profile).isdisjoint(mutable_ids(sibling))
    expected = copy.deepcopy(profile)
    next(iter(profile["workload"].values()))["variants"][0]["controlled"].clear()
    profile["checks"].clear()
    assert prepared.profile(case=("1", "4", "off", "2")) == expected
    assert snapshot == before


@pytest.mark.parametrize("cpu_count", [1, 4])
def test_fixed_condition_tokens_are_prepared_once_per_source(multi_case_sources, monkeypatch, cpu_count):
    fixture = multi_case_sources(cpu_count)
    calls = []
    original = conditions.digest

    def record_digest(value):
        if value == "structured":
            calls.append(value)
        return original(value)

    monkeypatch.setattr(conditions, "digest", record_digest)
    model = load_analysis([fixture.left, fixture.right])

    assert len(model.configs) == 4 * cpu_count
    assert all(profile["status"] == "compatible" for config in model.configs
               for profile in config["comparison_profiles"].values())
    assert len(calls) == 2


def test_later_loads_refresh_fixed_conditions_without_changing_other_sources(multi_case_sources):
    fixture = multi_case_sources(1)
    before = [config["comparison_profiles"]["same-hardware"]
              for config in load_analysis([fixture.left, fixture.right]).configs]
    assert conditions.compare_profiles(before[0], before[2])["status"] == "compatible"

    fixture.change_json(fixture.right, "input_scale_plan.json", lambda plan:
                        plan["quality_constraints"].update(target="different"))
    after = [config["comparison_profiles"]["same-hardware"]
             for config in load_analysis([fixture.left, fixture.right]).configs]

    assert after[:2] == before[:2]
    assert conditions.compare_profiles(after[0], after[2])["reasons"] == ["quality_constraints"]
    assert conditions.compare_profiles(before[0], before[2])["status"] == "compatible"


@pytest.mark.parametrize(("value", "encoded"), [
    (1, b"1"), (1., b"1.0"), (True, b"true"), (-0., b"-0.0"),
    ([1, 2], b"[1,2]"), ([], None), ({}, None), (None, None),
    ("unknown", None), (["known", None], None),
])
def test_fixed_tokens_preserve_types_and_incomplete_evidence(multi_case_sources, value, encoded):
    fixture = multi_case_sources(2)
    snapshot = load_comparison_snapshot(fixture.left)
    snapshot["conditions"]["quality_constraints"] = value
    prepared = conditions.PreparedComparison(snapshot)
    expected = hashlib.sha256(encoded).hexdigest() if encoded is not None else None

    for cpu in ("1", "2"):
        for purpose in conditions.PURPOSES:
            profile = prepared.profile(purpose=purpose, case=(cpu, "4", "off", "2"))
            assert profile["checks"]["quality_constraints"] == {"value": expected, "difference": "incompatible"}
            assert ("quality_constraints" in profile["reasons"]) is (expected is None)


def test_condition_preparation_keeps_flattening_order_and_independent_results(multi_case_sources):
    fixture = multi_case_sources(1)
    snapshot = load_comparison_snapshot(fixture.left)
    snapshot["conditions"]["quality_constraints"] = {"target": {"value": 1, "missing": None}, "target.value": 2}
    prepared = conditions.PreparedComparison(snapshot)
    first = prepared.profile()
    sibling = prepared.profile()
    assert first["checks"]["quality_constraints.target.value"] == {
        "value": hashlib.sha256(b"2").hexdigest(), "difference": "incompatible"}
    assert first["checks"]["quality_constraints.target.missing"]["value"] is None
    first["checks"]["quality_constraints.target.value"]["value"] = "corrupted"
    assert prepared.profile() == sibling


def test_fixed_conditions_are_lazy_and_failed_preparation_is_not_reused(multi_case_sources):
    fixture = multi_case_sources(1)
    snapshot = load_comparison_snapshot(fixture.left)
    snapshot["conditions"]["quality_constraints"] = {"valid": 1, "invalid": object()}
    prepared = conditions.PreparedComparison(snapshot)
    with pytest.raises(ValueError, match="comparison purpose"):
        prepared.profile(purpose="invalid")
    for _ in range(2):
        with pytest.raises(TypeError, match="not JSON serializable"):
            prepared.profile()


def test_resource_and_runtime_thread_checks_still_follow_each_case(multi_case_sources):
    fixture = multi_case_sources(2)
    snapshot = load_comparison_snapshot(fixture.left)
    snapshot["hardware"]["runtime_threads"] = {"1c_4g_off": 1, "2c_4g_off": 2}
    prepared = conditions.PreparedComparison(snapshot)
    for cpu in ("1", "2"):
        for purpose in conditions.PURPOSES:
            profile = prepared.profile(purpose=purpose, case=(cpu, "4", "off", "2"))
            assert profile["checks"]["resources.cpus"] == {
                "value": hashlib.sha256(json.dumps(cpu).encode()).hexdigest(),
                "difference": "expected_difference" if purpose == "resource-scaling" else "incompatible"}
            thread_key = "runtime_threads.off" if purpose == "resource-scaling" else f"runtime_threads.{cpu}c_4g_off"
            encoded = f"[{cpu}]" if purpose == "resource-scaling" else cpu
            assert profile["checks"][thread_key] == {
                "value": hashlib.sha256(encoded.encode()).hexdigest(), "difference": "incompatible"}


@pytest.mark.parametrize(("cpu", "neighbor", "bound"), [
    ("1.5", "1", False),
    ("1.0000000000000001", "1", False),
    ("9007199254740993", "9007199254740992", True),
])
def test_hardware_evidence_binds_to_exact_measured_case(multi_case_sources, cpu, neighbor, bound):
    fixture = multi_case_sources(1)
    for directory in (fixture.left, fixture.right):
        path = directory / "result_all.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows({**row, "cpu_cores": value} for value in (neighbor, cpu) for row in rows)

        def update_hardware(payload):
            host = payload["cases"]["1c_4g_off"]
            payload["cases"] = {f"{neighbor}c_4g_off": host, f"{cpu}c_4g_off": {**host, "host_id": "other-host"}}

        def update_state(state):
            state["options"]["cpus"] = f"{neighbor},{cpu}"
            if not bound:
                del state["runtime"]  # Historical CSVs may have no recorded coverage plan.

        fixture.change_json(directory, "hardware_conditions.json", update_hardware)
        fixture.change_json(directory, "run_state.json", update_state)
        fixture.refresh(directory)

    snapshot = load_comparison_snapshot(fixture.left)
    assert snapshot["valid"]
    assert snapshot["metric_contracts"]["latency_app_s"]["all_windows"]
    assert snapshot["metric_availability"]["latency_app_s"]
    workload = snapshot["conditions"]["actual_workload"]
    assert {tuple(json.loads(key)) for key in workload} == {
        (value, "4", "off", scale) for value in (neighbor, cpu) for scale in ("2", "4")}
    assert all(case["request_count"] == 3 for case in workload.values())
    assert snapshot["hardware"]["host_id"] == {
        f"{neighbor}c_4g_off": "fixture-host", f"{cpu}c_4g_off": "other-host" if bound else None}

    statuses = {neighbor: "compatible", cpu: "compatible" if bound else "unknown"}
    model = load_analysis([fixture.left])
    for config in model.configs:
        profiles = config["comparison_profiles"]
        value = json.loads(next(iter(profiles["same-hardware"]["workload"])))[0]
        assert all(profile["status"] == statuses[value] for profile in profiles.values())
        assert config["metrics"]["latency_app_s"]["n"] == 1
    assert len(model.configs) == 4
    for purpose in conditions.PURPOSES:
        result = compare_results(fixture.left, fixture.right, purpose=purpose)
        assert result["status"] == statuses[cpu]
        assert result["conditions"]["hardware_host_id"]["status"] == statuses[cpu]
        assert result["conditions"]["actual_workload"]["status"] == "compatible"

    fixture.change_json(fixture.right, "hardware_conditions.json",
                        lambda data: data["cases"][f"{neighbor}c_4g_off"].update(host_id="different-host"))
    result = compare_results(fixture.left, fixture.right, purpose="same-hardware")
    assert result["conditions"]["hardware_host_id"]["status"] == "incompatible"
    assert result["status"] == "incompatible"


@pytest.mark.parametrize(("left_formal", "right_formal", "right_probe", "expected"), [
    (1, None, 1, "unknown"), (None, 1, 1, "unknown"),
    (1, 1, 2, "compatible"), (1, 2, 1, "incompatible"),
    (None, None, 1, "compatible"), (None, None, 2, "incompatible"),
])
def test_thread_counts_compare_within_their_recorded_source(multi_case_sources, left_formal, right_formal,
                                                            right_probe, expected):
    fixture = multi_case_sources(1)
    left, right = (load_comparison_snapshot(path) for path in (fixture.left, fixture.right))
    for snapshot, formal, probe in ((left, left_formal, 1), (right, right_formal, right_probe)):
        snapshot["hardware"]["runtime_threads"] = {"1c_4g_off": formal}
        snapshot["conditions"]["runtime_threads"] = {"off": probe}

    for purpose in conditions.PURPOSES:
        comparison = compare_snapshots(left, right, purpose=purpose)
        check = comparison["conditions"]["runtime_threads"]
        assert check["status"] == expected
        for side, formal, probe in (("left", left_formal, 1), ("right", right_formal, right_probe)):
            assert check[side] == ({"off": probe} if formal is None else {"1c_4g_off": formal})
        if (left_formal is None) != (right_formal is None):
            assert comparison["status"] == "unknown"


@pytest.mark.parametrize("invalid", [True, {}, "Infinity", 0, -1])
def test_invalid_recorded_plan_scale_has_a_diagnostic_and_keeps_observations(multi_case_sources, invalid):
    fixture = multi_case_sources(1)
    fixture.change_json(fixture.left, "input_scale_plan.json",
                        lambda plan: plan["entries"][0].update(input_scale=invalid))
    snapshot = load_comparison_snapshot(fixture.left)

    assert not snapshot["valid"]
    assert any("input_scale_plan.json" in issue and "invalid input_scale" in issue for issue in snapshot["issues"])
    assert snapshot["metric_availability"]["latency_app_s"]
    assert snapshot["conditions"]["planned_inputs"] is None
    assert all(config["metrics"]["latency_app_s"]["n"] == 1 for config in load_analysis([fixture.left]).configs)
