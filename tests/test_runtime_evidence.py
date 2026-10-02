import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from acprof.analysis.audit import audit_result
from acprof.failures import Failure, RuntimeFailure
from acprof.tui.progress import RunProgressTracker


class RuntimeEvidenceTests(unittest.TestCase):
    def test_actual_dtype_reports_mixed_parameters_instead_of_primary_property(self):
        from acprof.container.load_policy import actual_dtype
        parameters = [SimpleNamespace(dtype=dtype, is_floating_point=lambda: True) for dtype in ("torch.float16", "torch.float32")]
        model = SimpleNamespace(dtype="torch.float16", parameters=lambda: iter(parameters))
        self.assertEqual(actual_dtype({"model": model}), "mixed[torch.float16, torch.float32]")

    def test_all_consumers_preserve_the_same_reason_code(self):
        from acprof.analysis.compatibility import write_compatibility_report
        from acprof.tui.reports import read_report
        failure = Failure("preflight", "runtime_task_unsupported", "本地化说明", runtime_profile="cv-transformers560-cpu")
        row = {"model_id": "fixture/model", "failure": failure.to_dict()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_compatibility_report(root, [row])
            with (root / "models.csv").open() as stream:
                self.assertEqual(next(csv.DictReader(stream))["reason_code"], failure.reason_code)
            self.assertIn(failure.reason_code, (root / "REPORT.md").read_text())
            source = root / "coverage.json"
            source.write_text(json.dumps({"schema_version": 1, "scope": "selected_sample_only; no_formal_measurement", "rows": [row]}))
            self.assertEqual(read_report(source).rows[0].cells[2], failure.reason_code)
        tracker = RunProgressTracker()
        for line in str(RuntimeFailure(failure)).splitlines():
            tracker.feed(line)
        self.assertEqual(tracker.snapshot.failure, failure.to_dict())

    def test_budget_uses_selected_files_and_never_claims_measured_oom(self):
        from acprof.analysis.compatibility import result_status
        from acprof.resource_budget import assess_resource_budget
        small = assess_resource_budget(plan={"selected_bytes": 50, "repository_bytes": 5000}, max_download_bytes=100)
        self.assertEqual(small["status"], "within_budget")
        for plan in ({"total_selected_bytes": 150}, {"selected_bytes": None}):
            with self.subTest(plan=plan):
                result = assess_resource_budget(plan=plan, max_download_bytes=100)
                self.assertEqual(result_status(result), "unverified")
                self.assertFalse(result["failure"]["evidence"]["measured_oom"])

    def test_recorded_result_report_preserves_warnings_and_legacy_unknown(self):
        from acprof.analysis.compatibility import report_results, result_status
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [root / "current", root / "legacy"]
            for source in sources:
                source.mkdir()
                (source / "static_meta.json").write_text(json.dumps({"model_id": "fixture/model", "model_revision": "a" * 40}))
                (source / "capability_report.json").write_text('{"full_profile_complete": true}')
            (sources[0] / "quality_checks.json").write_text(json.dumps({"schema_version": 1, "checks": [{"code": "cpu_idle_baseline_unstable", "severity": "warning"}]}))
            before = {path: path.read_bytes() for source in sources for path in source.iterdir()}
            report = report_results(sources, root / "report")
            self.assertEqual(result_status(report["rows"][0]), "full_success_with_warnings")
            self.assertEqual(result_status(report["rows"][1]), "full_success_quality_unknown")
            self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_quality_does_not_revoke_verified_capability(self):
        from acprof.analysis.compatibility import result_status
        from acprof.capabilities import Capability, CapabilityReport, apply_runtime_validation
        check = {"code": "weights_reinitialized", "severity": "warning"}
        report = CapabilityReport("full", measurement={"latency": Capability("verified")}, requested={"latency"},
                                  collection_complete=True, identity={"comparability_class": "native_linux"})
        apply_runtime_validation(report, {"devices": {"off": {"status": "ok", "quality_checks": [check]}}})
        snapshot = report.to_dict()
        self.assertTrue(snapshot["full_profile_complete"])
        self.assertNotIn("weights_reinitialized", json.dumps(snapshot))
        self.assertEqual(result_status({**snapshot, "quality_checks": [check]}), "full_success_with_warnings")

    def test_server_failure_survives_client_and_is_written_after_request(self):
        from acprof.host.client import ClientRunner
        from acprof.host.client_config import ClientConfig
        failure = Failure("predict", "inference_failed", "arbitrary detail", evidence={"model_loaded": True})
        runner = ClientRunner.__new__(ClientRunner)
        runner.runtime_failures = []
        runner.config = ClientConfig()
        response = SimpleNamespace(status_code=500, json=lambda: {"error": "different text", "failure": failure.to_dict()})
        with patch("acprof.host.client.requests.post", return_value=response):
            with self.assertRaises(RuntimeFailure) as caught:
                runner._one_request(1, "case_w0:0", {"input_scale": 1})
        failure = caught.exception.failure
        self.assertEqual(failure.reason_code, "inference_failed")
        self.assertEqual(failure.evidence["input_scale"], 1)
        self.assertEqual(failure.evidence["request_id"], "case_w0:0")
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            runner.config.out_csv = str(Path(directory, "case.csv"))
            runner.main = Mock(side_effect=caught.exception)
            with self.assertRaises(SystemExit):
                runner.run_cli()
            saved = json.loads(Path(directory, "case.csv.runtime_failures.json").read_text())
        self.assertEqual(saved["failures"], [failure.to_dict()])

    def test_tui_keeps_typed_failure_instead_of_classifying_detail(self):
        failure = Failure("preflight", "runtime_dependency_missing", "任意本地化文本", evidence={"module": "skimage"})
        tracker = RunProgressTracker()
        for line in str(RuntimeFailure(failure)).splitlines():
            tracker.feed(line)
        self.assertEqual(tracker.snapshot.failure["reason_code"], "runtime_dependency_missing")

    def test_audit_consumes_quality_and_failures_without_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "result_all.csv").write_text("status,warmup\nok,0\n")
            failure = Failure("load", "processor_incompatible", "localized").to_dict()
            (root / "runtime_validation.json").write_text(json.dumps({"devices": {"off": {"failure": failure}}}))
            check = {"code": "weights_reinitialized", "severity": "warning", "observed": ["head.weight"],
                     "threshold": 0, "detail": "loading info", "evidence": {"source": "from_pretrained"}}
            (root / "quality_checks.json").write_text(json.dumps({"schema_version": 1, "checks": [check]}))
            report = audit_result(root)
        self.assertEqual(report.get("failures"), [failure])
        self.assertEqual(report.get("quality_checks"), [check])


if __name__ == "__main__":
    unittest.main()
