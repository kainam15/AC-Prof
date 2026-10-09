"""跨后端比较条件与续跑身份独立；审计只读取已有产物。"""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from comparison_fixtures import ComparisonFixture

from acprof.cli.audit import main
from acprof.config import CSV_FIELDS
from acprof.platform import Environment


class TestResultComparison(ComparisonFixture):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.build(tmp_path)

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

    def test_proportional_variant_counts_keep_distribution(self):
        self.write_distribution(self.left, [99, 1])
        self.write_distribution(self.right, [990, 10])
        actual = self.compare()["conditions"]["actual_workload"]
        assert (actual["status"]) == ("compatible")
        assert (actual["request_counts_changed"])
        assert (actual["changed_dimensions"]) == ({})

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
