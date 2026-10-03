"""HTML and CLI share condition evidence, including resource and hardware policy."""
import csv
import json
import unittest

import test_independent_comparison as independent_fixture
import test_result_comparison as comparison_fixture


class ComparisonProfileTests(unittest.TestCase):
    def setUp(self):
        self.fixture = comparison_fixture.ResultComparisonTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_html_reuses_strict_input_order_verdict_and_reason(self):
        from acprof.analysis.comparison import compare_profiles, compare_results
        from acprof.analysis.model import load_analysis
        fixture = self.fixture
        fixture.change_json(fixture.right, "input_scale_plan.json", lambda plan:
                            plan["entries"][0]["payload"]["features"].reverse())
        configs = load_analysis([fixture.left, fixture.right]).configs
        report = compare_profiles(*(c["comparison_profiles"]["same-hardware"] for c in configs))
        self.assertEqual(report["status"], compare_results(fixture.left, fixture.right)["status"])
        self.assertEqual(report["status"], "incompatible")
        self.assertIn("planned_inputs", report["reasons"])

    def test_resource_scaling_allows_quota_only_and_keeps_actual_workload(self):
        from acprof.analysis.comparison import compare_results
        from acprof.analysis.model import load_analysis
        fixture = self.fixture
        fixture.change_json(fixture.right, "run_state.json", lambda state: state["options"].update(cpus="2"))
        fixture.change_json(fixture.right, "hardware_conditions.json", lambda evidence:
                            evidence["cases"].update({"2c_4g_off": evidence["cases"].pop("1c_4g_off")}))
        path = fixture.right / "result_all.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, row = reader.fieldnames, next(reader)
        row["cpu_cores"] = "2"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerow(row)
        report = compare_results(fixture.left, fixture.right, purpose="resource-scaling")
        self.assertEqual(report["status"], "compatible")
        self.assertEqual(report["allowed_resource_dimensions"], ["cpu", "memory"])
        configs = load_analysis([fixture.left, fixture.right]).configs
        from acprof.analysis.comparison import compare_profiles
        self.assertEqual(compare_profiles(*(c["comparison_profiles"]["resource-scaling"] for c in configs))["status"], "compatible")
        summary = json.loads(row["workload_contract"])
        summary["variants"][0]["contract"]["input"]["feature_dim"] = 9
        row["workload_contract"] = json.dumps(summary)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerow(row)
        self.assertEqual(compare_results(fixture.left, fixture.right, purpose="resource-scaling")["status"], "incompatible")

    def test_html_projects_each_configuration_workload_in_a_multi_case_experiment(self):
        from acprof.analysis.conditions import compare_profiles
        from acprof.analysis.model import load_analysis
        fixture = self.fixture
        path = fixture.left / "result_all.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, row = reader.fieldnames, next(reader)
        other = {**row, "cpu_cores": "2"}
        workload = json.loads(other["workload_contract"])
        workload["variants"][0]["contract"]["input"]["feature_dim"] = 9
        other["workload_contract"] = json.dumps(workload)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows([row, other])
        fixture.change_json(fixture.left, "run_state.json", lambda state: state["options"].update(cpus="1,2"))
        fixture.change_json(fixture.left, "hardware_conditions.json", lambda evidence:
            evidence["cases"].update({"2c_4g_off": evidence["cases"]["1c_4g_off"]}))
        configs = load_analysis([fixture.left]).configs
        profiles = [c["comparison_profiles"]["resource-scaling"] for c in configs]
        self.assertTrue(all(p["status"] == "compatible" for p in profiles))
        self.assertEqual(compare_profiles(*profiles)["status"], "incompatible")
        self.assertEqual(compare_profiles(*profiles)["reasons"], ["actual_workload"])
        from acprof.analysis.independent_comparison import compare_experiments
        with self.assertRaisesRegex(ValueError, "one resource configuration"):
            compare_experiments([fixture.left], [fixture.right], metrics=["latency_app_s"], purpose="resource-scaling")


class IndependentHardwarePolicyTests(unittest.TestCase):
    def test_resource_scaling_pairs_distinct_resource_coordinates_without_pooling_cases(self):
        from acprof.analysis.independent_comparison import compare_experiments
        fixture = independent_fixture.IndependentComparisonTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        left, right = fixture.replicate("left", 0, [2]), fixture.replicate("right", 0, [3])
        fixture.fixture.change_json(right, "run_state.json", lambda state: state["options"].update(cpus="2"))
        fixture.fixture.change_json(right, "hardware_conditions.json", lambda evidence:
            evidence["cases"].update({"2c_4g_off": evidence["cases"].pop("1c_4g_off")}))
        path = right / "result_all.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, row = reader.fieldnames, next(reader)
        row["cpu_cores"] = "2"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerow(row)
        report = compare_experiments([left], [right], metrics=["latency_app_s"], purpose="resource-scaling", resamples=20)
        self.assertEqual(len(report["groups"]), 1)
        group = report["groups"][0]
        self.assertEqual(group["difference"], 1)
        self.assertEqual(group["resource_coordinates"], {"left": {"cpu_cores": 1., "mem_cap_gb": 4.},
                                                          "right": {"cpu_cores": 2., "mem_cap_gb": 4.}})

    def test_cross_hardware_requires_homogeneous_replicates_in_both_groups(self):
        from acprof.analysis.independent_comparison import compare_experiments
        fixture = independent_fixture.IndependentComparisonTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        left = [fixture.replicate("left", i, [2]) for i in range(2)]
        right = [fixture.replicate("right", i, [3]) for i in range(2)]
        for paths, side in ((left, "left"), (right, "right")):
            fixture.fixture.change_json(paths[1], "hardware_conditions.json", lambda data:
                data["cases"]["1c_4g_off"].update(host_id="other-host"))
            report = compare_experiments(left, right, metrics=["latency_app_s"], purpose="cross-hardware", resamples=20)
            self.assertEqual(report["status"], "incompatible")
            self.assertTrue(any(check.get("group") == side and check.get("within_group")
                                and check["status"] == "incompatible" for check in report["condition_checks"]))
            fixture.fixture.change_json(paths[1], "hardware_conditions.json", lambda data:
                data["cases"]["1c_4g_off"].update(host_id="fixture-host"))
