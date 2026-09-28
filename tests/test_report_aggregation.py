"""A green shard is insufficient unless the entire discovered suite is accounted for."""
import copy
import hashlib
import unittest


class ReportAggregationTests(unittest.TestCase):
    def reports(self):
        ids = ["a.test", "b.test", "c.test", "d.test"]
        digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        return [{"schema_version": 1, "python": "3.12.7", "successful": True,
                 "patterns": ["test_*.py"], "discovery_counts": {"test_*.py": 4},
                 "shard": {"index": index, "count": 2, "discovered": 4, "selected": 2, "suite_sha256": digest},
                 "counts": {"run": 2, "passed": 2, "failed": 0, "errors": 0, "skipped": 0,
                            "expected_failures": 0, "unexpected_successes": 0},
                 "tests": [{"id": value, "outcome": "passed"} for value in ids[index::2]]} for index in range(2)]

    def summarize(self, reports):
        from scripts.aggregate_test_reports import aggregate_reports
        return aggregate_reports(reports, python_versions=["3.12"], shard_count=2)

    def test_complete_suite_is_counted_once(self):
        report = self.summarize(self.reports())
        self.assertTrue(report["successful"], report["errors"])
        self.assertEqual(report["versions"]["3.12"]["counts"]["passed"], 4)

    def test_missing_duplicate_and_unknown_shards_fail(self):
        reports = self.reports()
        for broken in (reports[:1], [*reports, reports[0]], [*reports, {**reports[0], "python": "3.10.9"}]):
            with self.subTest(broken=len(broken)):
                self.assertFalse(self.summarize(broken)["successful"])

    def test_forged_green_reports_cannot_hide_coverage_or_failures(self):
        for change in (lambda rows: rows[1]["tests"].pop(),
                       lambda rows: rows[1]["tests"][0].update(id="a.test"),
                       lambda rows: rows[1]["tests"][0].update(id="unseen.test"),
                       lambda rows: rows[1]["tests"][0].update(outcome="error"),
                       lambda rows: rows[1]["shard"].update(suite_sha256="different"),
                       lambda rows: rows[1]["counts"].update(passed=1),
                       lambda rows: rows[1].update(successful=False)):
            rows = copy.deepcopy(self.reports())
            change(rows)
            self.assertFalse(self.summarize(rows)["successful"])
