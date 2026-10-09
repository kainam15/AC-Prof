"""HTML and CLI share condition evidence, including resource and hardware policy."""
import csv
import json

import comparison_fixtures as comparison_fixture
import independent_comparison_fixtures as independent_fixture
import pytest


class TestComparisonProfile:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.fixture_root = tmp_path
        self.fixture = comparison_fixture.ComparisonFixture()
        self.fixture.build(self.fixture_root)

    def test_html_reuses_strict_input_order_verdict_and_reason(self):
        from acprof.analysis.comparison import compare_profiles, compare_results
        from acprof.analysis.model import load_analysis
        fixture = self.fixture
        fixture.change_json(fixture.right, "input_scale_plan.json", lambda plan:
                            plan["entries"][0]["payload"]["features"].reverse())
        configs = load_analysis([fixture.left, fixture.right]).configs
        report = compare_profiles(*(c["comparison_profiles"]["same-hardware"] for c in configs))
        assert (report["status"]) == (compare_results(fixture.left, fixture.right)["status"])
        assert (report["status"]) == ("incompatible")
        assert ("planned_inputs") in (report["reasons"])

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
        fixture.refresh(fixture.right)
        report = compare_results(fixture.left, fixture.right, purpose="resource-scaling")
        assert (report["status"]) == ("compatible")
        assert (report["allowed_resource_dimensions"]) == (["cpu", "memory"])
        configs = load_analysis([fixture.left, fixture.right]).configs
        from acprof.analysis.comparison import compare_profiles
        assert (compare_profiles(*(c["comparison_profiles"]["resource-scaling"] for c in configs))["status"]) == ("compatible")
        summary = json.loads(row["workload_contract"])
        summary["variants"][0]["contract"]["input"]["feature_dim"] = 9
        row["workload_contract"] = json.dumps(summary)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerow(row)
        fixture.refresh(fixture.right)
        assert (compare_results(fixture.left, fixture.right, purpose="resource-scaling")["status"]) == ("incompatible")

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
        fixture.refresh(fixture.left)
        configs = load_analysis([fixture.left]).configs
        profiles = [c["comparison_profiles"]["resource-scaling"] for c in configs]
        assert (all(p["status"] == "compatible" for p in profiles))
        assert (compare_profiles(*profiles)["status"]) == ("incompatible")
        assert (compare_profiles(*profiles)["reasons"]) == (["actual_workload"])
        from acprof.analysis.independent_comparison import compare_experiments
        with pytest.raises(ValueError, match="one resource configuration"):
            compare_experiments([fixture.left], [fixture.right], metrics=["latency_app_s"], purpose="resource-scaling")


class TestIndependentHardwarePolicy:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.fixture_root = tmp_path
    def test_resource_scaling_pairs_distinct_resource_coordinates_without_pooling_cases(self):
        from acprof.analysis.independent_comparison import compare_experiments
        fixture = independent_fixture.IndependentComparisonFixture()
        fixture.build(self._request, self.fixture_root)
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
        fixture.fixture.refresh(right)
        report = compare_experiments([left], [right], metrics=["latency_app_s"], purpose="resource-scaling", resamples=20)
        assert (len(report["groups"])) == (1)
        group = report["groups"][0]
        assert (group["difference"]) == (1)
        assert (group["resource_coordinates"]) == ({"left": {"cpu_cores": 1., "mem_cap_gb": 4.},
                                                          "right": {"cpu_cores": 2., "mem_cap_gb": 4.}})

    def test_cross_hardware_requires_homogeneous_replicates_in_both_groups(self):
        from acprof.analysis.independent_comparison import compare_experiments
        fixture = independent_fixture.IndependentComparisonFixture()
        fixture.build(self._request, self.fixture_root)
        left = [fixture.replicate("left", i, [2]) for i in range(2)]
        right = [fixture.replicate("right", i, [3]) for i in range(2)]
        for paths, side in ((left, "left"), (right, "right")):
            fixture.fixture.change_json(paths[1], "hardware_conditions.json", lambda data:
                data["cases"]["1c_4g_off"].update(host_id="other-host"))
            report = compare_experiments(left, right, metrics=["latency_app_s"], purpose="cross-hardware", resamples=20)
            assert (report["status"]) == ("incompatible")
            assert (any(check.get("group") == side and check.get("within_group")
                                and check["status"] == "incompatible" for check in report["condition_checks"]))
            fixture.fixture.change_json(paths[1], "hardware_conditions.json", lambda data:
                data["cases"]["1c_4g_off"].update(host_id="fixture-host"))
