"""A green shard is insufficient unless the entire discovered suite is accounted for."""
import copy
import hashlib

import pytest


class TestReportAggregation:
    def reports(self):
        ids = ["a.test", "b.test", "c.test", "d.test"]
        digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        return [{"schema_version": 1, "python": "3.12.7", "successful": True,
                 "packages": {"pytest": "9.0.0", "numpy": "2.4.4"},
                 "provenance": {"schema_version": 1, "source_sha256": "a" * 64,
                                "tests_sha256": "b" * 64, "locks_sha256": "c" * 64},
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
        assert (report["successful"]), report["errors"]
        assert (report["versions"]["3.12"]["counts"]["passed"]) == (4)

    @pytest.mark.parametrize('field', ('source_sha256', 'tests_sha256', 'locks_sha256', 'packages', 'missing'))
    def test_mixed_or_missing_evidence_identity_is_rejected(self, field):
        reports = self.reports()
        if field == 'packages':
            reports[1]['packages']['numpy'] = '2.2.6'
        elif field == 'missing':
            reports[1].pop('provenance')
        else:
            reports[1]['provenance'][field] = 'd' * 64
        result = self.summarize(reports)
        assert not result['successful']
        assert any('identity' in error or 'provenance' in error for error in result['errors'])

    def test_python_versions_may_change_packages_but_not_source_identity(self):
        from scripts.aggregate_test_reports import aggregate_reports
        reports = self.reports()
        other = copy.deepcopy(reports)
        for row in other:
            row['python'] = '3.10.21'
            row['packages']['numpy'] = '2.2.6'
        combined = [*reports, *other]
        assert aggregate_reports(combined, python_versions=['3.10', '3.12'], shard_count=2)['successful']
        for row in other:
            row['provenance']['source_sha256'] = 'd' * 64
        result = aggregate_reports(combined, python_versions=['3.10', '3.12'], shard_count=2)
        assert not result['successful']
        assert 'source or lock identity differs across Python versions' in result['errors']

    @pytest.mark.parametrize('broken_case', range(3), ids=['reports[:1]', '[*reports, reports[0]]', "[*reports, {**reports[0], 'python': '3.10.9'}]"])
    def test_missing_duplicate_and_unknown_shards_fail(self, broken_case):
        reports = self.reports()
        broken = tuple((reports[:1], [*reports, reports[0]], [*reports, {**reports[0], 'python': '3.10.9'}]))[broken_case]
        assert not (self.summarize(broken)["successful"])

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
            assert not (self.summarize(rows)["successful"])
