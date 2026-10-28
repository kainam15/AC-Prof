"""Comparison preparation is bounded by source evidence, not output purposes."""
import copy
import csv
import hashlib
import json
from unittest.mock import Mock

import pytest
from comparison_fixtures import ComparisonFixture

from acprof.analysis import conditions
from acprof.analysis.comparison import load_comparison_snapshot
from acprof.analysis.model import load_analysis


@pytest.fixture
def multi_case_sources(tmp_path):
    def build(cpu_count):
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
                    rows.append({**row, "cpu_cores": cpu, "input_scale": scale,
                                 "workload_contract": json.dumps({"schema_version": 1, "request_count": 3,
                                     "variants": [{"count": 3, "contract": contract}]})})
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


def test_input_index_keeps_float_equivalence_raw_values_and_entry_order(multi_case_sources):
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
        with pytest.raises(ValueError, match="could not convert string to float"):
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
