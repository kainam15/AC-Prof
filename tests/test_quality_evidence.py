"""Quality policy preserves evidence and refuses missing or unreviewed checks."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from acprof.quality import (
    QualityCheck,
    combine_quality,
    loading_quality,
    read_quality,
    summarize_quality,
)


class QualityEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_absent_explicit_empty_benign_and_output_checks_have_distinct_states(self):
        self.assertEqual(read_quality(self.root)["quality_status"], "unknown")
        self.assertEqual(summarize_quality([])["quality_status"], "passed")
        benign = summarize_quality(loading_quality({"unexpected_keys": ["unused"]}, source="loader"))
        self.assertEqual(benign["quality_status"], "warning")
        self.assertTrue(benign["auto_selection_eligible"])
        for info in ({"missing_keys": ["head"]}, {"mismatched_keys": ["head"]}):
            blocked = summarize_quality(loading_quality(info, source="loader"))
            self.assertEqual(blocked["quality_status"], "blocked")
            self.assertFalse(blocked["auto_selection_eligible"])
        unknown = summarize_quality([QualityCheck("new_warning", "warning", 1, 0,
            "No approved interpretation", {"source": "custom"}).to_dict()])
        self.assertEqual(unknown["quality_status"], "unknown")
        self.assertFalse(unknown["auto_selection_eligible"])
        self.assertIn("new_warning", unknown["quality_reasons"])

    def test_partial_quality_evidence_cannot_become_passed_when_merged(self):
        result = combine_quality([summarize_quality([]), summarize_quality(None)])
        self.assertEqual(result["quality_status"], "unknown")
        self.assertFalse(result["auto_selection_eligible"])
        self.assertIn("quality_evidence_missing", result["quality_reasons"])

    def test_runtime_device_evidence_and_original_source_are_preserved(self):
        checks = loading_quality({"missing_keys": ["head"]}, source="loader")
        (self.root / "runtime_validation.json").write_text(json.dumps({"devices": {
            "off": {"quality_checks": []}, "on": {"quality_checks": checks}}}))
        self.assertEqual(read_quality(self.root, device="off")["quality_status"], "passed")
        gpu = read_quality(self.root, device="on")
        self.assertEqual(gpu["quality_status"], "blocked")
        self.assertEqual(gpu["quality_checks"][0]["evidence"]["device"], "on")
        self.assertEqual(gpu["quality_checks"][0]["evidence"]["source"], "loader")
        self.assertTrue(gpu["quality_checks"][0]["evidence"]["artifact"].endswith("runtime_validation.json"))

    def test_malformed_evidence_is_unknown_and_does_not_fall_back_to_success(self):
        path = self.root / "quality_checks.json"
        for payload in ('{broken', '[]', '{"schema_version":1}',
                        '{"schema_version":1,"checks":[{"code":"x"}]}'):
            with self.subTest(payload=payload):
                path.write_text(payload)
                report = read_quality(self.root)
                self.assertEqual(report["quality_status"], "unknown")
                self.assertFalse(report["auto_selection_eligible"])
                self.assertEqual(report["quality_checks"][0]["code"], "quality_evidence_invalid")

    def test_compatibility_csv_keeps_attempt_and_unknown_quality_evidence(self):
        from acprof.analysis.compatibility import write_compatibility_report
        checks = loading_quality({"unexpected_keys": ["unused"]}, source="loader")
        row = {"model_id": "fixture/model", "quality_status": "unknown", "quality_checks": checks,
               "attempt_id": 2, "attempt_path": "attempts/0002/model", "configuration_sha256": "new-config",
               "failure": {"stage": "validation", "reason_code": "runtime_validation_failed", "evidence": {"log": "original-failure"}}}
        write_compatibility_report(self.root, [row])
        with (self.root / "models.csv").open() as stream:
            result = next(csv.DictReader(stream))
        self.assertEqual(result["quality_status"], "unknown")
        self.assertEqual(json.loads(result["quality_checks"])[0]["evidence"]["source"], "loader")
        self.assertEqual(result["attempt_id"], "2")
        self.assertEqual(result["attempt_path"], "attempts/0002/model")
        self.assertEqual(result["configuration_sha256"], "new-config")
