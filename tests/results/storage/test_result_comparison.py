"""跨后端比较条件与续跑身份独立；审计只读取已有产物。"""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from acprof.cli.audit import main
from acprof.config import CSV_FIELDS
from acprof.platform import Environment
from tests.support.comparison_fixtures import ComparisonFixture


def _case_workload(total, contract, observed=2):
    if total is None:
        return {"request_count": None, "observed_request_count": observed, "uncounted_windows": 1,
                "variants": [{"count": observed, "contract": contract, "proportion": None}]}
    variants = [{"count": total, "contract": contract, "proportion": "1"}] if total else []
    return {"request_count": total, "variants": variants}


class TestResultComparison(ComparisonFixture):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.build(tmp_path)

    def test_different_declared_workloads_are_not_pooled_despite_same_interface(self):
        for directory, task_id in ((self.left, 'translation_en_to_de'), (self.right, 'translation_en_to_fr')):
            self.change_json(directory, 'input_scale_plan.json', lambda plan: plan.update(task_workload={
                'task_id': task_id, 'interface': 'text2text-generation', 'fingerprint': task_id}))
        report = self.compare()
        assert report['conditions']['task_semantics']['status'] == 'incompatible'

    def test_wsl_cannot_enter_native_baseline_even_for_cross_hardware(self):
        from acprof.analysis.comparison import compare_results
        self.change_json(self.right, "static_meta.json", lambda meta: meta.update(Environment("wsl2").metadata()))
        path = self.right / "result_all.csv"
        path.write_text(path.read_text().replace("native_linux", "wsl2"))
        for purpose in ("same-hardware", "cross-hardware"):
            report = compare_results(self.left, self.right, purpose=purpose)
            assert (report["conditions"]["comparability_class"]["status"]) == ("incompatible")
            assert not (report["native_baseline_eligible"])
            assert (report["warnings"])
            assert (report["metric_comparability"]["cpu_energy_total_j"]["status"]) == ("not comparable")

    def test_legacy_environment_is_unknown_without_using_current_host(self):
        for directory in (self.left, self.right):
            self.change_json(directory, "static_meta.json", lambda meta: [meta.pop(name) for name in (
                "platform", "collection_tier", "comparability_class", "environment_class")])
            path = directory / "result_all.csv"
            path.write_text(path.read_text().replace("native_linux", "unknown"))
        report = self.compare()
        assert (report["status"]) == ("unknown")
        assert not (report["native_baseline_eligible"])

    def test_wsl_fabricated_zero_energy_is_invalid(self):
        from acprof.analysis.audit import audit_result
        self.change_json(self.right, "static_meta.json", lambda meta: meta.update(Environment("wsl2").metadata()))
        path = self.right / "result_all.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, row = reader.fieldnames, next(reader)
        row.update(environment_class="wsl2", cpu_energy_total_j="0")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerow(row)
        self.refresh(self.right)
        report = audit_result(self.right)
        assert not (report["valid"])
        assert (any(issue["code"] == "unsupported_wsl_metric" for issue in report["issues"]))

    def test_quality_and_measurement_evidence_are_distinct_from_comparability(self):
        from acprof.analysis.audit import audit_result
        from acprof.quality import loading_quality
        checks = loading_quality({"missing_keys": ["head.weight"]}, source="loader")
        self.write_json(self.right, "quality_checks.json", {"schema_version": 1, "checks": checks})
        report = self.compare()
        assert (report["status"]) == ("compatible")
        right = report["experiments"]["right"]
        assert (right["run_status"]) == ("complete")
        assert (right["measurement_status"]) == ("complete")
        assert (right["quality_status"]) == ("blocked")
        assert not (right["auto_selection_eligible"])
        assert ("weights_reinitialized") in (right["quality_reasons"])
        audit = audit_result(self.right)
        assert (audit["quality_status"]) == ("blocked")
        assert (audit["quality_checks"][0]["evidence"]["source"]) == ("loader")
        self.change_json(self.right, "run_state.json", lambda state: state["options"].update(repeat=2))
        assert (audit_result(self.right)["measurement_status"]) == ("incomplete")

    def test_expected_backend_identity_changes_do_not_prevent_comparison(self):
        report = self.compare()
        assert (report["status"]) == ("compatible")
        assert (report["conditions"]["planned_inputs"]["status"]) == ("compatible")
        assert ("runtime_backend") in (report["expected_differences"])
        assert ("image_id") in (report["expected_differences"])
        assert (report["experiments"]["left"]["run_id"]) != (report["experiments"]["right"]["run_id"])

    def test_malformed_hardware_case_is_reported_without_crashing(self):
        self.change_json(self.right, "hardware_conditions.json", lambda data: data["cases"].update({"1c_4g_off": []}))
        report = self.compare()
        assert not (report["valid"])
        assert (report["experiments"]["right"]["issues"])

    def test_comparison_metadata_uses_bounded_artifact_reader(self):
        original_read_text = Path.read_text
        metadata_names = {
            "run_state.json", "static_meta.json", "input_scale_plan.json",
            "hardware_conditions.json",
        }

        def guarded_read_text(path, *args, **kwargs):
            if path.name in metadata_names:
                raise AssertionError(f"unbounded metadata read: {path.name}")
            return original_read_text(path, *args, **kwargs)

        with patch.object(Path, "read_text", guarded_read_text):
            report = self.compare()

        assert report["status"] == "compatible"

    def test_input_order_change_is_detected_without_requiring_whole_plan_hash(self):
        self.change_json(self.right, "input_scale_plan.json", lambda plan: plan["entries"][0]["payload"]["features"].reverse())
        report = self.compare()
        assert (report["status"]) == ("incompatible")
        assert (report["conditions"]["planned_inputs"]["status"]) == ("incompatible")

    def test_resource_and_measurement_protocol_changes_are_detected(self):
        self.change_json(self.right, "run_state.json", lambda state: state["options"].update(cpus="2", profiling_mode="full"))
        report = self.compare()
        assert (report["conditions"]["resources"]["status"]) == ("incompatible")
        assert (report["conditions"]["measurement_protocol"]["status"]) == ("incompatible")

    @pytest.mark.parametrize(("field", "value"), [("sampling_policy", "skip"), ("raw_sample_limit", 500)])
    @pytest.mark.parametrize("purpose", ["same-hardware", "cross-hardware", "resource-scaling"])
    def test_sampling_and_retention_protocol_changes_block_comparison(self, field, value, purpose):
        from acprof.analysis.comparison import compare_results

        self.change_json(self.right, "run_state.json", lambda state: state["options"].update({field: value}))
        report = compare_results(self.left, self.right, purpose=purpose)
        assert report["conditions"]["measurement_protocol"]["status"] == "incompatible"
        assert report["status"] == "incompatible"
        assert report["metric_comparability"]["latency_app_s"]["status"] == "not comparable"

    @pytest.mark.parametrize("field", ["sampling_policy", "raw_sample_limit"])
    @pytest.mark.parametrize("missing_on_both", [False, True])
    def test_missing_historical_sampling_evidence_remains_unknown(self, field, missing_on_both):
        directories = (self.left, self.right) if missing_on_both else (self.right,)
        for directory in directories:
            self.change_json(directory, "run_state.json", lambda state: state["options"].pop(field))
        before = {directory: (directory / "run_state.json").read_bytes() for directory in directories}
        report = self.compare()
        protocol = report["conditions"]["measurement_protocol"]
        assert protocol["right"][field] is None
        assert protocol["status"] == "unknown"
        assert report["status"] == "unknown"
        assert {directory: (directory / "run_state.json").read_bytes() for directory in directories} == before

    def test_quality_threshold_change_is_not_hidden_by_protocol_validation(self):
        self.change_json(self.right, "input_scale_plan.json", lambda plan: plan["quality_constraints"].update(atol=0.1))
        report = self.compare()
        assert (report["conditions"]["quality_constraints"]["status"]) == ("incompatible")

    def test_missing_historical_fields_remain_unknown(self):
        for directory in (self.left, self.right):
            self.change_json(directory, "input_scale_plan.json", lambda plan: plan.pop("quality_constraints"))
        report = self.compare()
        assert (report["status"]) == ("unknown")
        assert (report["conditions"]["quality_constraints"]["status"]) == ("unknown")
        assert (report["conditions"]["planned_inputs"]["status"]) == ("compatible")

    def test_actual_change_does_not_change_preexecution_identity(self):
        before = (self.right / "run_state.json").read_bytes()
        contract = copy.deepcopy(self.contract)
        contract["input"]["actual_scale"] = 1
        self.write_rows(self.right, contract)
        report = self.compare()
        assert (report["conditions"]["planned_inputs"]["status"]) == ("compatible")
        assert (report["conditions"]["actual_workload"]["status"]) == ("incompatible")
        assert ((self.right / "run_state.json").read_bytes()) == (before)

    def test_auto_window_request_counts_do_not_change_per_request_work(self):
        self.write_rows(self.right, self.contract, count=9)
        assert (self.compare()["conditions"]["actual_workload"]["status"]) == ("compatible")

    def test_reversed_variant_distribution_is_not_equivalent(self):
        self.write_distribution(self.left, [99, 1])
        self.write_distribution(self.right, [1, 99])
        actual = self.compare()["conditions"]["actual_workload"]
        assert (actual["status"]) == ("incompatible")
        assert (actual["reason"]) == ("per_request_distribution_changed")
        assert ("input.actual_scale") in (actual["changed_dimensions"])
        case = next(iter(actual["left"].values()))
        assert (case["request_count"]) == (100)
        assert ([v["count"] for v in case["variants"]]) == ([99, 1])

    def test_missing_output_fact_does_not_hide_known_input_change(self):
        contract = copy.deepcopy(self.contract)
        contract["output"]["shape"] = None
        self.write_rows(self.left, contract)
        contract["input"]["actual_scale"] = 9
        self.write_rows(self.right, contract)
        actual = self.compare()["conditions"]["actual_workload"]
        assert (actual["status"]) == ("incompatible")
        assert ("input.actual_scale") in (actual["changed_dimensions"])

    def test_equal_incomplete_contracts_remain_unknown_with_counts_retained(self):
        contract = copy.deepcopy(self.contract)
        contract["output"]["shape"] = None
        self.write_rows(self.left, contract)
        self.write_rows(self.right, contract, count=9)
        actual = self.compare()["conditions"]["actual_workload"]
        assert (actual["status"]) == ("unknown")
        assert (actual["reason"]) == ("actual_workload_evidence_incomplete")
        assert (actual["request_counts_changed"])

    @pytest.mark.parametrize(("section", "field"), [
        (None, "task"), (None, "batch_size"), ("scenario", "type"), ("input", "planned_scale"),
        ("input", "actual_scale"), ("output", "type"), ("output", "shape"), ("output", "count"),
    ])
    def test_matching_contracts_missing_a_base_field_remain_unknown(self, section, field):
        contract: dict[str, Any] = copy.deepcopy(self.contract)
        target = contract if section is None else contract[section]
        target.pop(field)
        if section == "scenario":
            target["concurrency"] = 1
        self.write_rows(self.left, contract, count=2)
        self.write_rows(self.right, contract, count=3)
        report = self.compare()
        actual = report["conditions"]["actual_workload"]
        assert report["valid"]
        assert report["status"] == "unknown"
        assert actual["reason"] == "actual_workload_evidence_incomplete"
        assert actual["status"] == "unknown"
        assert actual["request_counts_changed"]
        case = next(iter(actual["left"].values()))
        assert case["request_count"] == 2
        assert case["variants"] == [{"contract": contract, "count": 2, "proportion": "1"}]

    def test_mixed_variants_keep_controlled_differences_when_a_base_field_is_absent(self):
        from acprof.analysis.conditions import compare_workload_profiles, workload_case_profile

        partial: dict[str, Any] = copy.deepcopy(self.contract)
        del partial["output"]["shape"]
        case = {"request_count": 3, "variants": [
            {"count": 2, "contract": self.contract}, {"count": 1, "contract": partial},
        ]}
        baseline = workload_case_profile(case)
        assert not baseline["complete"]
        assert [v["proportion"] for v in baseline["variants"]] == ["2/3", "1/3"]
        assert compare_workload_profiles({"case": baseline}, {"case": baseline}) == "unknown"
        partial["input"]["actual_scale"] = 9
        changed = workload_case_profile(case)
        assert compare_workload_profiles({"case": baseline}, {"case": changed}) == "incompatible"

    @pytest.mark.parametrize(("leaf", "complete", "status"), [(1, True, "compatible"), (None, False, "unknown")])
    def test_deep_workload_extension_preserves_known_and_unknown_evidence(self, leaf, complete, status):
        from acprof.analysis.conditions import workload_case_profile

        extension = leaf
        for _ in range(520):
            extension = [extension]
        contract = {**self.contract, "extension": extension}
        case = {"request_count": 1, "variants": [{"count": 1, "contract": contract}]}
        assert workload_case_profile(case)["complete"] is complete
        self.write_rows(self.left, contract)
        self.write_rows(self.right, contract)
        report = self.compare()
        assert report["valid"]
        assert report["status"] == status
        assert report["conditions"]["actual_workload"]["status"] == status

    def test_proportional_variant_counts_keep_distribution(self):
        self.write_distribution(self.left, [99, 1])
        self.write_distribution(self.right, [990, 10])
        actual = self.compare()["conditions"]["actual_workload"]
        assert (actual["status"]) == ("compatible")
        assert (actual["request_counts_changed"])
        assert (actual["changed_dimensions"]) == ({})

    @pytest.mark.parametrize("reserved_field", [False, True], ids=["ordinary-field", "joint-distribution-field"])
    def test_marginal_change_in_one_case_preserves_joint_change_in_another(self, reserved_field):
        from acprof.analysis.comparison import _workload_condition

        def observed(*shapes):
            variants = []
            proportion = "1" if len(shapes) == 1 else f"1/{len(shapes)}"
            for rows, features in shapes:
                contract: dict[str, Any] = copy.deepcopy(self.contract)
                contract["input"].update(rows=rows, feature_dim=features)
                variants.append({"contract": contract, "count": 1, "proportion": proportion})
            return {"request_count": len(variants), "variants": variants}

        marginal, joint = '["1","4","off","2"]', '["2","4","off","2"]'
        left = {marginal: observed((2, 2)), joint: observed((1, 2), (2, 4))}
        right = {marginal: observed((2, 6)), joint: observed((1, 4), (2, 2))}
        if reserved_field:
            for cases, value in ((left, 7), (right, 8)):
                cases[marginal]["variants"][0]["contract"]["joint_distribution"] = value
        condition = _workload_condition(left, right)
        assert condition["status"] == "incompatible"
        changed = condition["changed_dimensions"]
        assert changed["input.feature_dim"] == {marginal: {"left": {"2": "1"}, "right": {"6": "1"}}}
        assert set(changed) == {"input.feature_dim", "joint_distribution"}
        assert set(changed["joint_distribution"]) == ({marginal, joint} if reserved_field else {joint})
        if reserved_field:
            assert changed["joint_distribution"][marginal] == {"left": {"7": "1"}, "right": {"8": "1"}}
        for side, expected in (("left", {(1, 2), (2, 4)}), ("right", {(1, 4), (2, 2)})):
            histogram = changed["joint_distribution"][joint][side]
            shapes = [json.loads(raw)["input"] for raw in histogram]
            assert {(shape["rows"], shape["feature_dim"]) for shape in shapes} == expected
            assert set(histogram.values()) == {"1/2"}

    @pytest.mark.parametrize(("uncertain_left", "uncertain_right", "observed", "other_right", "changed"), [
        (None, 3, 2, 2, None), (None, 3, 3, 2, None), (None, None, 5, 2, None),
        (None, 3, 2, 5, True), (None, 3, 5, 2, True), (3, None, 5, 2, True),
        (0, 0, 0, 2, False), (0, 3, 0, 2, True),
    ])
    def test_request_count_changes_use_only_supported_totals(
            self, uncertain_left, uncertain_right, observed, other_right, changed):
        from acprof.analysis.comparison import _workload_condition

        left = {"partial": _case_workload(uncertain_left, self.contract, observed),
                "other": _case_workload(2, self.contract)}
        right = {"partial": _case_workload(uncertain_right, self.contract, observed),
                 "other": _case_workload(other_right, self.contract)}
        condition = _workload_condition(left, right)
        assert condition["request_counts_changed"] is changed
        assert condition["status"] == "unknown"
        assert condition["reason"] == "actual_workload_evidence_incomplete"
        assert not condition["changed_dimensions"]

    def test_counted_null_contract_preserves_counts_without_asserting_a_change(self):
        self.write_rows(self.left, None)
        condition = self.compare()["conditions"]["actual_workload"]
        assert condition["request_counts_changed"] is False
        assert condition["status"] == "unknown"
        assert condition["reason"] == "actual_workload_evidence_incomplete"
        assert not condition["changed_dimensions"]
        assert list(condition["left"].values()) == [
            {"request_count": 3, "variants": [{"count": 3, "contract": None, "proportion": "1"}]}]

    @pytest.mark.parametrize("partial_total", [None, 0, 3])
    def test_partial_case_does_not_hide_another_cases_observed_workload_difference(self, partial_total):
        from acprof.analysis.comparison import _workload_condition

        changed: dict[str, Any] = copy.deepcopy(self.contract)
        changed["input"]["actual_scale"] = 9
        incomplete_contract = self.contract if partial_total is None else None
        left = {"partial": _case_workload(partial_total, incomplete_contract),
                "other": _case_workload(2, self.contract)}
        right = {"partial": _case_workload(3, self.contract),
                 "other": _case_workload(2, changed)}
        condition = _workload_condition(left, right)
        assert condition["status"] == "incompatible"
        assert condition["reason"] == "per_request_distribution_changed"
        assert condition["changed_dimensions"]["input.actual_scale"] == {
            "other": {"left": {"2": "1"}, "right": {"9": "1"}}}
        assert all("partial" not in cases for cases in condition["changed_dimensions"].values())

    def test_variable_task_output_difference_requires_equivalence_evidence(self):
        self.write_distribution(self.left, [99, 1], output_only=True, task="text-generation")
        self.write_distribution(self.right, [1, 99], output_only=True, task="text-generation")
        actual = self.compare()["conditions"]["actual_workload"]
        assert (actual["status"]) == ("unknown")
        assert (actual["reason"]) == ("output_distribution_equivalence_unverified")
        assert ("output.count") in (actual["changed_dimensions"])

    def test_fixed_task_output_difference_is_incompatible(self):
        self.write_distribution(self.left, [99, 1], output_only=True)
        self.write_distribution(self.right, [1, 99], output_only=True)
        actual = self.compare()["conditions"]["actual_workload"]
        assert (actual["status"]) == ("incompatible")
        assert ("output.count") in (actual["changed_dimensions"])

    def test_equivalent_csv_numeric_spelling_uses_existing_measurement_keys(self):
        path = self.right / "result_all.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0].update(cpu_cores="1.0", input_scale="2.00", warmup="0.0")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        report = self.compare()
        assert (report["conditions"]["actual_workload"]["status"]) == ("compatible")
        assert (report["metric_comparability"]["latency_app_s"]["status"]) == ("comparable")

    def test_known_mismatch_is_reported_even_when_another_legacy_field_is_missing(self):
        self.change_json(self.right, "static_meta.json", lambda metadata: metadata.pop("cgroup_collection_mode"))
        self.change_json(self.right, "run_state.json", lambda state: state["options"].update(batch_size=2))
        report = self.compare()
        assert (report["conditions"]["resources"]["status"]) == ("incompatible")

    def test_effective_threads_are_compared_instead_of_backend_variable_names(self):
        for directory in (self.left, self.right):
            self.change_json(directory, "hardware_conditions.json",
                             lambda data: data["cases"]["1c_4g_off"].pop("runtime_threads"))
        self.change_json(self.left, "run_state.json", lambda state: state["options"]["measurement_environment"].pop("ACPROF_RUNTIME_THREADS"))
        self.change_json(self.left, "run_state.json", lambda state: state["options"]["measurement_environment"].update(TORCH_NUM_THREADS="1"))
        self.change_json(self.right, "run_state.json", lambda state: state["options"]["measurement_environment"].update(ACPROF_RUNTIME_THREADS="1"))
        report = self.compare()
        assert (report["conditions"]["runtime_threads"]["status"]) == ("compatible")
        assert (report["status"]) == ("unknown")
        self.change_json(self.right, "static_meta.json", lambda metadata: metadata["runtime_validation"]["devices"]["off"]["runtime_parameters"]["effective"].update(threads=2))
        assert (self.compare()["conditions"]["runtime_threads"]["status"]) == ("incompatible")

    def test_probe_threads_do_not_establish_default_server_threads(self):
        for directory in (self.left, self.right):
            self.change_json(directory, "hardware_conditions.json",
                             lambda data: data["cases"]["1c_4g_off"].pop("runtime_threads"))
        for directory in (self.left, self.right):
            self.change_json(directory, "run_state.json", lambda state: state["options"]["measurement_environment"].pop("ACPROF_RUNTIME_THREADS"))
        # Independent validation injects a quota-derived thread count. The
        # ordinary server does not, so equal probe counts do not prove equality.
        assert (self.compare()["conditions"]["runtime_threads"]["status"]) == ("unknown")

    def test_zero_thread_request_retains_unknown_runtime_default(self):
        for directory in (self.left, self.right):
            self.change_json(directory, "hardware_conditions.json",
                             lambda data: data["cases"]["1c_4g_off"].pop("runtime_threads"))
        self.change_json(self.left, "run_state.json", lambda state: state["options"]["measurement_environment"].update(ACPROF_RUNTIME_THREADS="0"))
        assert (self.compare()["conditions"]["runtime_threads"]["status"]) == ("unknown")

    def test_invalid_effective_thread_value_is_not_comparable(self):
        for directory in (self.left, self.right):
            self.change_json(directory, "hardware_conditions.json",
                             lambda data: data["cases"]["1c_4g_off"].pop("runtime_threads"))
        self.change_json(self.right, "static_meta.json", lambda metadata: metadata["runtime_validation"]["devices"]["off"]["runtime_parameters"]["effective"].update(threads=True))
        assert (self.compare()["conditions"]["runtime_threads"]["status"]) == ("unknown")

    def test_missing_effective_threads_are_unknown_even_if_request_was_recorded(self):
        for directory in (self.left, self.right):
            self.change_json(directory, "hardware_conditions.json",
                             lambda data: data["cases"]["1c_4g_off"].pop("runtime_threads"))
        self.change_json(self.right, "static_meta.json", lambda metadata: metadata.pop("runtime_validation"))
        assert (self.compare()["conditions"]["runtime_threads"]["status"]) == ("unknown")

    def test_malformed_metadata_produces_failed_audit_instead_of_crashing(self):
        self.change_json(self.right, "run_state.json", lambda state: state.update(options=[]))
        self.change_json(self.right, "static_meta.json", lambda metadata: metadata.update(runtime_validation=[1]))
        report = self.compare()
        assert not (report["valid"])
        assert (report["status"]) == ("incompatible")
        assert (report["experiments"]["right"]["issues"])

    def test_cli_exposes_comparison_and_unknown_requires_explicit_strict_mode(self):
        self.change_json(self.right, "input_scale_plan.json", lambda plan: plan.pop("quality_constraints"))
        snapshots = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = main([str(self.left), "--compare", str(self.right), "--json"])
        assert (code) == (0)
        assert (json.loads(output.getvalue())["comparison"]["status"]) == ("unknown")
        with contextlib.redirect_stdout(io.StringIO()):
            assert (main([str(self.left), "--compare", str(self.right), "--require-comparable"])) == (1)
        assert ({path: path.read_bytes() for path in snapshots}) == (snapshots)
