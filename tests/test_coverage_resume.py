"""Coverage recovery preserves frozen samples, successful models and failed attempts."""
import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_resolution_decisions import candidate

from acprof.cli.coverage import main
from acprof.failures import Failure, RuntimeFailure
from acprof.host.model_coverage import run_sample


class CoverageResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "coverage"
        self.sample = {"schema_version": 1, "sampling": "fixed", "weight_basis": "uniform",
                       "models": [{"model_id": f"example/model-{index}", "revision": "a" * 40,
                                   "weight": 1} for index in range(3)]}
        self.host = {"source_sha256": "source", "machine_id_sha256": "machine",
                     "environment_class": "native_linux", "python": "3.12", "packages_sha256": "packages"}
        for target, value in (("acprof.host.run_state.host_identity", self.host),
                              ("acprof.host.automation.check_repository_access", None)):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def task(self, model_id, *, revision):
        task = candidate(tag="text-generation")
        task.model_id = model_id
        task.model_revision = revision
        return task

    def run_batch(self, **kwargs):
        with patch("acprof.host.detect.detect_task", side_effect=self.task):
            return run_sample(self.sample, self.root, **kwargs)

    def failure(self, code="request_timeout", stage="predict"):
        return RuntimeFailure(Failure(stage, code, "original failure", retryability="higher_budget",
                                      evidence={"original_budget": 60}))

    def test_resume_after_interruption_skips_completed_models_and_keeps_original_attempt(self):
        seen = []

        def interrupted(model_id, *, revision):
            seen.append(model_id)
            if model_id.endswith("-1"):
                raise KeyboardInterrupt
            return self.task(model_id, revision=revision)

        with patch("acprof.host.detect.detect_task", side_effect=interrupted), self.assertRaises(KeyboardInterrupt):
            run_sample(self.sample, self.root)
        first_attempt = self.root / "attempts/attempt-000001/attempt.json"
        original = first_attempt.read_bytes()
        with patch("acprof.host.detect.detect_task", side_effect=self.task) as detect:
            report = run_sample(self.sample, self.root, resume=True)
        self.assertEqual([call.args[0] for call in detect.call_args_list], ["example/model-1", "example/model-2"])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["summary"]["completed_count"], 3)
        self.assertEqual(len(report["attempts"]), 2)
        self.assertEqual(first_attempt.read_bytes(), original)
        self.assertEqual([row["attempt_id"] for row in report["rows"]], [1, 2, 2])
        with patch("acprof.host.detect.detect_task") as detect:
            self.assertEqual(run_sample(self.sample, self.root, resume=True), report)
        detect.assert_not_called()

    def test_changed_sample_device_and_configuration_fail_before_resolution(self):
        self.run_batch()
        original = (self.root / "coverage.json").read_bytes()
        changed = copy.deepcopy(self.sample)
        changed["models"][0]["revision"] = "b" * 40
        for sample, options in ((changed, {}), (self.sample, {"cpus": 4}),
                                (self.sample, {"gpu": True}), (self.sample, {"timeout_seconds": 600})):
            with self.subTest(options=options), patch("acprof.host.detect.detect_task") as detect:
                with self.assertRaisesRegex(ValueError, "sample|configuration|配置|样本"):
                    run_sample(sample, self.root, resume=True, **options)
                detect.assert_not_called()
                self.assertEqual((self.root / "coverage.json").read_bytes(), original)

    def test_retry_filters_preserve_old_budget_and_failure_in_an_independent_attempt(self):
        with patch("acprof.host.model_inspection.probe_model_contract", side_effect=[
                {"status": "ok"}, self.failure(), self.failure("resource_limit", "resource")]):
            before = self.run_batch(probe="full", timeout_seconds=60)
        original = (self.root / "attempts/attempt-000001/attempt.json").read_bytes()
        with patch("acprof.host.model_inspection.probe_model_contract", return_value={"status": "ok"}) as probe:
            after = self.run_batch(probe="full", timeout_seconds=600, resume=True,
                                   retry_stages=["predict"], retry_reasons=["request_timeout"])
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(probe.call_args.args[0].model_id, "example/model-1")
        self.assertEqual(after["resources"]["timeout_seconds"], 60)
        self.assertEqual(after["attempts"][1]["configuration"]["resources"]["timeout_seconds"], 600)
        self.assertEqual(after["attempts"][1]["configuration_changes"]["resources.timeout_seconds"],
                         {"before": 60, "after": 600})
        self.assertEqual(after["rows"][1]["runtime_status"], "ok")
        self.assertEqual(after["rows"][2]["failure"], before["rows"][2]["failure"])
        self.assertEqual((self.root / "attempts/attempt-000001/attempt.json").read_bytes(), original)
        self.assertTrue(after["mixed_configurations"])

    def test_resume_reads_durable_success_after_summary_write_was_interrupted(self):
        self.run_batch()
        summary = json.loads((self.root / "coverage.json").read_text())
        summary["rows"] = summary["rows"][:1]
        summary["status"] = "running"
        (self.root / "coverage.json").write_text(json.dumps(summary))
        with patch("acprof.host.detect.detect_task") as detect:
            report = run_sample(self.sample, self.root, resume=True)
        detect.assert_not_called()
        self.assertEqual(len(report["rows"]), 3)

    def test_resume_rejects_unknown_legacy_budget_evidence_without_rewriting_history(self):
        self.root.mkdir()
        (self.root / "sample.json").write_text(json.dumps(self.sample))
        legacy = {"schema_version": 1, "rows": [], "probe": "none", "resources": {"cpus": 2}}
        (self.root / "coverage.json").write_text(json.dumps(legacy))
        with self.assertRaisesRegex(ValueError, "schema|history|历史"):
            self.run_batch(resume=True)
        self.assertEqual(json.loads((self.root / "coverage.json").read_text()), legacy)

    def test_retry_requires_resume_and_unknown_selectors_do_not_create_attempts(self):
        with self.assertRaisesRegex(ValueError, "resume"):
            self.run_batch(retry_failed=True)
        self.assertFalse(self.root.exists())
        with patch("acprof.host.model_inspection.probe_model_contract", side_effect=self.failure()):
            self.run_batch(probe="full")
        with self.assertRaisesRegex(ValueError, "matching|匹配"):
            self.run_batch(probe="full", resume=True, retry_reasons=["resource_limit"])
        self.assertEqual(len(list((self.root / "attempts").iterdir())), 1)

    def test_cli_resume_and_retry_flags_reach_shared_runner(self):
        source = Path(self.temporary.name) / "sample.json"
        source.write_text(json.dumps(self.sample))
        with patch("acprof.host.env_utils.bootstrap_project_env"), patch(
                "acprof.host.model_coverage.run_sample", return_value={"summary": {}}) as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["run", str(source), "--output-dir", str(self.root), "--resume",
                                   "--retry-stage", "predict", "--retry-reason", "request_timeout"]), 0)
        self.assertTrue(run.call_args.kwargs["resume"])
        self.assertEqual(run.call_args.kwargs["retry_stages"], ["predict"])
        self.assertEqual(run.call_args.kwargs["retry_reasons"], ["request_timeout"])

    def test_parameter_budget_retry_does_not_replace_initial_budget(self):
        def bounded_task(model_id, *, revision):
            task = self.task(model_id, revision=revision)
            task.parameter_count = 12
            return task

        with patch("acprof.host.detect.detect_task", side_effect=bounded_task):
            initial = run_sample(self.sample, self.root, max_parameters=10)
            retried = run_sample(self.sample, self.root, max_parameters=20, resume=True,
                                 retry_reasons=["resource_limit"])
        self.assertEqual(initial["rows"][0]["failure"]["reason_code"], "resource_limit")
        self.assertEqual(retried["budgets"]["max_parameters"], 10)
        self.assertEqual(retried["attempts"][1]["configuration"]["budgets"]["max_parameters"], 20)
        self.assertNotIn("failure", retried["rows"][0])

    def test_resume_of_interrupted_retry_only_continues_that_attempt_selection(self):
        with patch("acprof.host.model_inspection.probe_model_contract", side_effect=[
                {"status": "ok"}, self.failure(), self.failure()]):
            self.run_batch(probe="full", timeout_seconds=60)
        with patch("acprof.host.model_inspection.probe_model_contract", side_effect=[
                {"status": "ok"}, KeyboardInterrupt]), self.assertRaises(KeyboardInterrupt):
            self.run_batch(probe="full", timeout_seconds=600, resume=True, retry_failed=True)
        with patch("acprof.host.model_inspection.probe_model_contract", return_value={"status": "ok"}) as probe:
            result = self.run_batch(probe="full", timeout_seconds=600, resume=True)
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(probe.call_args.args[0].model_id, "example/model-2")
        self.assertEqual([row["attempt_id"] for row in result["rows"]], [1, 2, 3])

    def test_missing_attempt_evidence_and_modified_frozen_copy_are_rejected(self):
        self.run_batch()
        path = self.root / "attempts/attempt-000001/attempt.json"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing|evidence"):
            self.run_batch(resume=True)
        path.write_bytes(original)
        frozen = copy.deepcopy(self.sample)
        frozen["models"][0]["weight"] = 7
        (self.root / "sample.json").write_text(json.dumps(frozen))
        with self.assertRaisesRegex(ValueError, "sample"):
            self.run_batch(resume=True)

    def test_directory_lock_prevents_two_recovery_writers(self):
        from acprof.host.run_state import ResultDirectoryLock, RunStateError
        self.run_batch()
        with ResultDirectoryLock(self.root), patch("acprof.host.detect.detect_task") as detect:
            with self.assertRaises(RunStateError):
                self.run_batch(resume=True)
        detect.assert_not_called()

    def test_resume_checks_physical_gpu_uuid_before_running_a_probe(self):
        with patch("acprof.host.gpu_device.pin_gpu_device", return_value={"uuid": "GPU-first"}), patch(
                "acprof.host.model_inspection.probe_model_contract", return_value={"status": "ok"}):
            self.run_batch(probe="full", gpu=True)
        with patch("acprof.host.gpu_device.pin_gpu_device", return_value={"uuid": "GPU-replacement"}), patch(
                "acprof.host.model_inspection.probe_model_contract") as probe:
            with self.assertRaisesRegex(ValueError, "configuration"):
                self.run_batch(probe="full", gpu=True, resume=True)
        probe.assert_not_called()

    def test_cleanup_failure_stops_batch_after_preserving_run_and_cleanup_evidence(self):
        original_failure = self.failure().failure.to_dict()
        cleanup = {"schema_version": 1, "status": "incomplete", "container_id": "c" * 64,
                   "final_state": "unknown", "run_error": {"detail": "original timeout"}}

        def incomplete_cleanup(task, output, **kwargs):
            validation = {"status": "error", "cleanup_status": "incomplete", "devices": {
                "off": {"status": "inconclusive", "failure": original_failure,
                        "cleanup_error": cleanup, "quality_checks": []}}}
            (output / "runtime_validation.json").write_text(json.dumps(validation))
            raise RuntimeError("owned container cleanup incomplete")

        with patch("acprof.host.model_inspection.probe_model_contract", side_effect=incomplete_cleanup) as probe:
            with self.assertRaisesRegex(RuntimeError, "cleanup"):
                self.run_batch(probe="full")
        self.assertEqual(probe.call_count, 1)
        report = json.loads((self.root / "coverage.json").read_text())
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(len(report["rows"]), 1)
        row = report["rows"][0]
        self.assertEqual(row["failure"], original_failure)
        self.assertEqual(row["cleanup_status"], "incomplete")
        self.assertEqual(row["cleanup_errors"], [cleanup])
        original = (self.root / "attempts/attempt-000001/attempt.json").read_bytes()
        with patch("acprof.host.model_inspection.probe_model_contract") as probe:
            with self.assertRaisesRegex(ValueError, "cleanup"):
                self.run_batch(probe="full", resume=True)
        probe.assert_not_called()
        with patch("acprof.host.model_inspection.probe_model_contract", return_value={"status": "ok"}) as probe, patch(
                "acprof.host.coverage_state.confirm_previous_cleanup", return_value={"status": "complete"}) as confirm:
            retried = self.run_batch(probe="full", resume=True, retry_stages=["cleanup"])
        confirm.assert_called_once()
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(len(retried["rows"]), 1)
        self.assertEqual((self.root / "attempts/attempt-000001/attempt.json").read_bytes(), original)

    def test_cleanup_recovery_failure_without_device_result_still_stops_batch(self):
        cleanup = {"status": "incomplete", "final_state": "present", "container_id": "d" * 64}

        def incomplete_cleanup(task, output, **kwargs):
            (output / "runtime_validation.json").write_text(json.dumps({
                "status": "error", "cleanup_status": "incomplete", "cleanup_error": cleanup, "devices": {}}))
            raise RuntimeError("recovery cleanup incomplete")

        with patch("acprof.host.model_inspection.probe_model_contract", side_effect=incomplete_cleanup) as probe:
            with self.assertRaisesRegex(RuntimeError, "cleanup"):
                self.run_batch(probe="full")
        self.assertEqual(probe.call_count, 1)
        report = json.loads((self.root / "coverage.json").read_text())
        self.assertEqual(report["rows"][0]["cleanup_errors"], [cleanup])

    def test_resume_finishes_a_retry_killed_after_its_last_durable_row(self):
        with patch("acprof.host.model_inspection.probe_model_contract", side_effect=self.failure()):
            self.run_batch(probe="full", timeout_seconds=60)
        with patch("acprof.host.model_inspection.probe_model_contract", return_value={"status": "ok"}):
            self.run_batch(probe="full", timeout_seconds=600, resume=True, retry_failed=True)
        path = self.root / "attempts/attempt-000002/attempt.json"
        interrupted = json.loads(path.read_text())
        interrupted["status"] = "running"
        interrupted.pop("finished_at")
        path.write_text(json.dumps(interrupted))
        original = path.read_bytes()
        with patch("acprof.host.model_inspection.probe_model_contract") as probe:
            report = self.run_batch(probe="full", timeout_seconds=600, resume=True)
        probe.assert_not_called()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(report["attempts"][-1]["mode"], "resume")
        self.assertEqual(report["attempts"][-1]["status"], "complete")
        self.assertEqual(report["attempts"][-1]["model_indices"], [])

    def test_resume_rejects_rows_bound_to_another_attempt_or_configuration(self):
        self.run_batch()
        path = self.root / "attempts/attempt-000001/attempt.json"
        original = path.read_bytes()
        for field, value in (("configuration_sha256", "different"), ("attempt_path", "attempts/attempt-000002")):
            with self.subTest(field=field):
                damaged = json.loads(original)
                damaged["rows"][0][field] = value
                path.write_text(json.dumps(damaged))
                with patch("acprof.host.detect.detect_task") as detect:
                    with self.assertRaisesRegex(ValueError, "attempt|configuration"):
                        self.run_batch(resume=True)
                detect.assert_not_called()
        path.write_bytes(original)

    def test_changed_effective_download_source_requires_retry_and_records_the_difference(self):
        with patch.dict(os.environ, {"HF_DOWNLOAD_MODE": "mirror-only", "HF_ENDPOINT": "https://first.invalid",
                                    "HF_FALLBACK_ENDPOINTS": ""}):
            with patch("acprof.host.model_inspection.probe_model_contract", side_effect=self.failure()):
                self.run_batch(probe="full")
            os.environ["HF_ENDPOINT"] = "https://second.invalid"
            with self.assertRaisesRegex(ValueError, "configuration"):
                self.run_batch(probe="full", resume=True)
            with patch("acprof.host.model_inspection.probe_model_contract", return_value={"status": "ok"}):
                report = self.run_batch(probe="full", resume=True, retry_failed=True)
        self.assertEqual(report["attempts"][-1]["configuration_changes"]["preparation_sources.endpoints"],
                         {"before": ["https://first.invalid"], "after": ["https://second.invalid"]})

    def test_finishing_interrupted_retry_does_not_adopt_other_unfinished_models(self):
        with patch("acprof.host.model_inspection.probe_model_contract", side_effect=[self.failure(), KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                self.run_batch(probe="full", timeout_seconds=60)
        with patch("acprof.host.model_inspection.probe_model_contract", return_value={"status": "ok"}):
            self.run_batch(probe="full", timeout_seconds=600, resume=True, retry_failed=True)
        path = self.root / "attempts/attempt-000002/attempt.json"
        interrupted = json.loads(path.read_text())
        interrupted["status"] = "running"
        path.write_text(json.dumps(interrupted))
        with patch("acprof.host.model_inspection.probe_model_contract") as probe:
            report = self.run_batch(probe="full", timeout_seconds=600, resume=True)
        probe.assert_not_called()
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(len(report["rows"]), 1)
        with patch("acprof.host.model_inspection.probe_model_contract", return_value={"status": "ok"}) as probe:
            report = self.run_batch(probe="full", timeout_seconds=60, resume=True)
        self.assertEqual(probe.call_count, 2)
        self.assertEqual(report["status"], "complete")

    def test_cli_reports_cleanup_stop_without_losing_the_persisted_evidence_location(self):
        from acprof.host.model_coverage import CoverageCleanupError
        source = Path(self.temporary.name) / "sample.json"
        source.write_text(json.dumps(self.sample))
        stderr = io.StringIO()
        with patch("acprof.host.env_utils.bootstrap_project_env"), patch(
                "acprof.host.model_coverage.run_sample", side_effect=CoverageCleanupError("cleanup: attempts/attempt-000001/attempt.json")), contextlib.redirect_stderr(stderr):
            self.assertEqual(main(["run", str(source), "--output-dir", str(self.root)]), 2)
        self.assertIn("attempts/attempt-000001/attempt.json", stderr.getvalue())

    def seed_cleanup_debt(self):
        cleanup = {"status": "incomplete", "container_id": "c" * 64, "final_state": "unknown"}
        result = {"status": "error", "cleanup_status": "incomplete", "cleanup_error": cleanup,
                  "devices": {"off": {"failure": self.failure().failure.to_dict()}}}
        with patch("acprof.host.model_inspection.probe_model_contract", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "cleanup"):
                self.run_batch(probe="full")
        return (self.root / "attempts/attempt-000001/attempt.json").read_bytes(), cleanup

    def test_cleanup_retry_cannot_erase_debt_when_previous_container_is_unconfirmed(self):
        original, cleanup = self.seed_cleanup_debt()
        with patch("acprof.host.coverage_state.confirm_previous_cleanup", create=True,
                   side_effect=RuntimeError("Docker did not confirm absence")) as confirm, patch(
                "acprof.host.detect.detect_task", side_effect=RuntimeError("Hub unavailable")) as detect, patch(
                "acprof.host.model_inspection.probe_model_contract") as probe:
            with self.assertRaisesRegex(RuntimeError, "cleanup"):
                run_sample(self.sample, self.root, probe="full", resume=True, retry_stages=["cleanup"])
        confirm.assert_called_once()
        detect.assert_not_called()
        probe.assert_not_called()
        report = json.loads((self.root / "coverage.json").read_text())
        self.assertEqual(report["rows"][0]["cleanup_status"], "incomplete")
        self.assertIn(cleanup, report["rows"][0]["cleanup_errors"])
        self.assertEqual(report["rows"][0]["cleanup_recovery"]["status"], "incomplete")
        self.assertEqual((self.root / "attempts/attempt-000001/attempt.json").read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "cleanup"):
            self.run_batch(probe="full", resume=True)

    def test_cleanup_confirmed_absent_is_preserved_even_if_new_resolution_fails(self):
        original, _ = self.seed_cleanup_debt()
        proof = {"status": "complete", "verified_container_ids": ["c" * 64], "recovered_container_ids": []}
        with patch("acprof.host.coverage_state.confirm_previous_cleanup", create=True, return_value=proof) as confirm, patch(
                "acprof.host.detect.detect_task", side_effect=RuntimeError("Hub unavailable")), patch(
                "acprof.host.model_inspection.probe_model_contract") as probe:
            report = run_sample(self.sample, self.root, probe="full", resume=True, retry_stages=["cleanup"])
        confirm.assert_called_once()
        probe.assert_not_called()
        self.assertEqual(report["rows"][0]["cleanup_recovery"], proof)
        self.assertNotEqual(report["rows"][0].get("cleanup_status"), "incomplete")
        self.assertEqual(report["rows"][0]["failed_stage"], "resolution")
        self.assertEqual((self.root / "attempts/attempt-000001/attempt.json").read_bytes(), original)

    def test_runtime_build_overrides_require_explicit_retry_and_keep_original_conditions(self):
        changes = {"ACPROF_NLP_TORCH_INDEX_URL": ("https://download.pytorch.org/whl/cu124", "https://download.pytorch.org/whl/cpu"),
                   "ACPROF_NLP_TORCH_SPEC": ("torch==2.6.0", "torch==2.7.0"),
                   "ACPROF_HOST_CUDA_VERSION": ("12.4", "12.8")}
        for index, (name, (before, after)) in enumerate(changes.items()):
            with self.subTest(override=name), patch.dict(os.environ, {name: before}):
                self.root = Path(self.temporary.name) / f"overrides-{index}"
                with patch("acprof.host.model_inspection.probe_model_contract", side_effect=self.failure()):
                    self.run_batch(probe="full")
                original = (self.root / "attempts/attempt-000001/attempt.json").read_bytes()
                os.environ[name] = after
                with patch("acprof.host.detect.detect_task") as detect:
                    with self.assertRaisesRegex(ValueError, "configuration"):
                        run_sample(self.sample, self.root, probe="full", resume=True)
                detect.assert_not_called()
                with patch("acprof.host.model_inspection.probe_model_contract", return_value={"status": "ok"}):
                    report = self.run_batch(probe="full", resume=True, retry_failed=True)
                self.assertEqual(report["attempts"][-1]["configuration_changes"][f"runtime_build_overrides.{name}"],
                                 {"before": before, "after": after})
                self.assertEqual((self.root / "attempts/attempt-000001/attempt.json").read_bytes(), original)
                self.assertTrue(report["mixed_configurations"])

    def test_repeated_cleanup_retry_cannot_rebind_debt_to_another_host_or_docker_context(self):
        self.seed_cleanup_debt()
        for environment, host, reason in (({}, {**self.host, "machine_id_sha256": "another-host"}, "original host"),
                                         ({"DOCKER_CONTEXT": "another-daemon"}, self.host, "endpoint/context")):
            for _ in range(2):
                with self.subTest(reason=reason), patch.dict(os.environ, environment), patch(
                        "acprof.host.run_state.host_identity", return_value=host), patch(
                        "acprof.host.detect.detect_task") as detect:
                    with self.assertRaisesRegex(RuntimeError, "cleanup"):
                        run_sample(self.sample, self.root, probe="full", resume=True, retry_stages=["cleanup"])
                detect.assert_not_called()
                row = json.loads((self.root / "coverage.json").read_text())["rows"][0]
                self.assertEqual(row["cleanup_origin_attempt_id"], 1)
                self.assertIn(reason, row["cleanup_recovery"]["error"]["detail"])

    def test_legacy_retry_without_cleanup_confirmation_cannot_hide_prior_debt(self):
        self.seed_cleanup_debt()
        with patch("acprof.host.coverage_state.confirm_previous_cleanup", side_effect=RuntimeError("Docker unavailable")):
            with self.assertRaisesRegex(RuntimeError, "cleanup"):
                self.run_batch(probe="full", resume=True, retry_stages=["cleanup"])
        path = self.root / "attempts/attempt-000002/attempt.json"
        data = json.loads(path.read_text())
        for name in ("cleanup_status", "cleanup_errors", "cleanup_recovery", "cleanup_origin_attempt_id"):
            data["rows"][0].pop(name, None)
        data["status"] = "complete"
        path.write_text(json.dumps(data))
        with patch("acprof.host.detect.detect_task") as detect:
            with self.assertRaisesRegex(ValueError, "cleanup"):
                run_sample(self.sample, self.root, probe="full", resume=True)
        detect.assert_not_called()

    def test_cleanup_retry_cannot_rebind_an_unknown_container_to_another_owner_uid(self):
        self.seed_cleanup_debt()
        original = json.loads((self.root / "coverage.json").read_text())["configuration"]
        changed = {**original, "container_owner_uid": os.getuid() + 1}
        for _ in range(2):
            with patch("acprof.host.coverage_state.coverage_configuration", return_value=changed), patch(
                    "acprof.host.detect.detect_task") as detect:
                with self.assertRaisesRegex(RuntimeError, "cleanup"):
                    run_sample(self.sample, self.root, probe="full", resume=True, retry_stages=["cleanup"])
            detect.assert_not_called()
            row = json.loads((self.root / "coverage.json").read_text())["rows"][0]
            self.assertEqual(row["cleanup_origin_attempt_id"], 1)
            self.assertIn("original owner UID", row["cleanup_recovery"]["error"]["detail"])
