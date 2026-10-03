"""Only distinct runs count as independent replicates; failed samples remain visible."""
import csv
import json
import shutil
from unittest.mock import patch

import pytest
from independent_comparison_fixtures import IndependentComparisonFixture


class TestIndependentComparison(IndependentComparisonFixture):
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path):
        self.build(request, tmp_path)

    def test_constant_ratio_and_difference_use_run_means(self):
        left = [self.replicate("left", i, [2, 2]) for i in range(3)]
        right = [self.replicate("right", i, [3, 3]) for i in range(3)]
        report = self.compare(left, right)
        group = report["groups"][0]
        assert (group["left"]["n_runs"]) == (3)
        assert (group["difference"]) == (1)
        assert (group["ratio"]) == (1.5)
        assert (group["difference_ci"]) == ([1, 1])
        assert (group["ratio_ci"]) == ([1.5, 1.5])

    def test_many_windows_in_one_run_do_not_create_independent_interval(self):
        report = self.compare([self.replicate("left", 0, [2] * 12)], [self.replicate("right", 0, [3] * 12)])
        group = report["groups"][0]
        assert (group["left"]["n_runs"]) == (1)
        assert (group["difference_ci"]) is None
        assert (group["reason"]) == ("insufficient_independent_runs")

    def test_copy_of_same_run_cannot_be_a_new_replicate(self):
        run = self.replicate("left", 0, [2])
        duplicate = self.fixture.root / "copied"
        shutil.copytree(run, duplicate)
        with pytest.raises(ValueError, match="duplicate.*run"):
            self.compare([run, duplicate], [self.replicate("right", 0, [3])])

    def test_failed_windows_remain_visible_and_do_not_enter_means(self):
        left = [self.replicate("left", 0, [2, 999], failed=True)]
        right = [self.replicate("right", 0, [3, 3])]
        report = self.compare(left, right)
        group = report["groups"][0]
        assert (group["left"]["failed_windows"]) == (1)
        assert (group["left"]["mean"]) == (2)

    def test_changed_actual_workload_blocks_performance_claim(self):
        left = self.replicate("left", 0, [2])
        right = self.replicate("right", 0, [3])
        with (right / "result_all.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, row = reader.fieldnames, next(reader)
        summary = json.loads(row["workload_contract"])
        summary["variants"][0]["contract"]["output"]["count"] = 9
        row["workload_contract"] = json.dumps(summary)
        with (right / "result_all.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerow(row)
        report = self.compare([left], [right])
        assert (report["status"]) == ("incompatible")
        assert (report["groups"][0]["difference"]) is None

    def test_quality_reaches_run_and_side_without_changing_statistics(self):
        from acprof.quality import loading_quality
        left = self.replicate("left", 0, [2])
        right = self.replicate("right", 0, [3])
        checks = loading_quality({"missing_keys": ["head.weight"]}, source="loader")
        self.fixture.write_json(right, "quality_checks.json", {"schema_version": 1, "checks": checks})
        report = self.compare([left], [right])
        assert (report["status"]) == ("compatible")
        side = report["groups"][0]["right"]
        assert (side["mean"]) == (3)
        assert (side["quality_status"]) == ("blocked")
        assert not (side["auto_selection_eligible"])
        assert (side["runs"][0]["quality_status"]) == ("blocked")
        assert (report["quality"]["right"]["quality_checks"][0]["code"]) == ("weights_reinitialized")
        assert (str(right / "quality_checks.json")) in (report["source_sha256"])

    def test_quality_change_during_comparison_invalidates_snapshot(self):
        from acprof.analysis.comparison import compare_results
        left = self.replicate("left", 0, [2])
        right = self.replicate("right", 0, [3])
        self.fixture.write_json(right, "quality_checks.json", {"schema_version": 1, "checks": []})

        def change_after_read(a, b, **kwargs):
            result = compare_results(a, b, **kwargs)
            if b == right:
                self.fixture.write_json(right, "quality_checks.json", {"schema_version": 1, "checks": [], "changed": True})
            return result

        with patch("acprof.analysis.independent_comparison.compare_results", side_effect=change_after_read):
            with pytest.raises(ValueError, match="experiment changed.*quality_checks"):
                self.compare([left], [right])
