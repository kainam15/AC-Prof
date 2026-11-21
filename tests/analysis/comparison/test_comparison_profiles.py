"""HTML and CLI share condition evidence, including resource and hardware policy."""
import csv
import json

import pytest

import tests.support.comparison_fixtures as comparison_fixture
import tests.support.independent_comparison_fixtures as independent_fixture


class TestComparisonProfile:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.fixture_root = tmp_path
        self.fixture = comparison_fixture.ComparisonFixture()
        self.fixture.build(self.fixture_root)

    @pytest.mark.parametrize("known_count", [0, 3])
    def test_unknown_request_contract_keeps_its_weight_without_comparable_facts(self, known_count):
        from acprof.analysis.conditions import (
            canonical,
            compare_workload_profiles,
            digest,
            workload_case_profile,
        )

        contract = self.fixture.contract
        variants: list[dict[str, object]] = [{"contract": None, "count": 1}]
        if known_count:
            variants.append({"contract": contract, "count": known_count})
        profile = workload_case_profile({"request_count": 1 + known_count, "variants": variants})
        assert not profile["complete"]
        unknown_weight = "1/4" if known_count else "1"
        assert profile["variants"][0] == {"proportion": unknown_weight, "fields": {}, "controlled": []}
        distribution = {"null": unknown_weight}
        if known_count:
            known = profile["variants"][1]
            assert known["proportion"] == "3/4"
            assert known["fields"]["input.actual_scale"] == "2"
            distribution[canonical(contract)] = "3/4"
        assert profile["observed"] == digest(distribution)
        assert compare_workload_profiles({"partial": profile}, {"partial": profile}) == "unknown"

    def test_unknown_request_in_one_case_keeps_proven_difference_in_another(self):
        from acprof.analysis.conditions import compare_workload_profiles, workload_case_profile

        def profile(contract):
            return workload_case_profile({"request_count": 1, "variants": [{"count": 1, "contract": contract}]})

        partial = profile(None)
        known = profile(self.fixture.contract)
        changed = profile({**self.fixture.contract, "batch_size": 2})
        assert compare_workload_profiles({"partial": partial, "measured": known},
                                         {"partial": partial, "measured": changed}) == "incompatible"

    @pytest.mark.parametrize("request_count", [None, 0])
    def test_unavailable_distribution_is_ambiguous_and_its_cached_profile_is_isolated(self, request_count):
        from acprof.analysis.comparison import load_comparison_snapshot
        from acprof.analysis.conditions import PreparedComparison, compare_workload_profiles

        snapshot = load_comparison_snapshot(self.fixture.left)
        workload = snapshot["conditions"]["actual_workload"]
        key = next(iter(workload))
        workload[key]["request_count"] = request_count
        if request_count == 0:
            workload[key]["variants"] = []
        prepared = PreparedComparison(snapshot)
        projected = prepared.profile()["workload"]
        assert projected == {key: {"ambiguous": True}}
        assert compare_workload_profiles(projected, projected) == "unknown"
        projected[key]["ambiguous"] = False
        assert prepared.profile()["workload"] == {key: {"ambiguous": True}}

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

    @pytest.mark.parametrize("missing", [False, True], ids=["different-input", "missing-window-count"])
    def test_html_projects_each_configuration_workload_in_a_multi_case_experiment(self, missing):
        from acprof.analysis.conditions import compare_profiles
        from acprof.analysis.model import load_analysis
        fixture = self.fixture
        path = fixture.left / "result_all.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, row = reader.fieldnames, next(reader)
        other = {**row, "cpu_cores": "2"}
        if missing:
            other.update(workload_contract="N/A", repeat_in_window="nan")
        else:
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
        expected = {1: "compatible", 2: "unknown" if missing else "compatible"}
        for policy in ("same-hardware", "resource-scaling"):
            assert {c["cpu"]: c["comparison_profiles"][policy]["status"] for c in configs} == expected
        assert compare_profiles(*profiles)["status"] == ("unknown" if missing else "incompatible")
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
