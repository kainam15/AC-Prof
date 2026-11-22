"""Coverage keeps a frozen denominator and does not count resource failures as semantic errors."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.cli.main import main
from tests.execution.planning.test_resolution_decisions import candidate


def test_runtime_failures_keep_semantic_review_and_frozen_denominator():
    from acprof.host.preparation.model_coverage import run_sample
    from acprof.host.preparation.model_inspection import ProbePreparationError
    task = candidate(tag="text-generation")
    manifest = {"schema_version": 1, "sampling": "fixed", "weight_basis": "uniform",
                "models": [{"model_id": f"example/model-{index}", "revision": "a" * 40, "weight": 1,
                            "semantic_reference": {"task": "text-generation", "source": "manual_fixture"}}
                           for index in range(3)]}
    with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task", return_value=task), patch(
            "acprof.host.automation.check_repository_access"), patch(
        "acprof.host.preparation.model_inspection.validate_model_runtime", side_effect=[
                {"status": "ok"}, {"status": "resource_limited"}, ProbePreparationError("build failed")]):
        report = run_sample(manifest, Path(directory, "report"), probe="full")
        summary = report["summary"]
        assert (summary["weighted_runtime_success_rate"]) == (1 / 3)
        assert (summary["weighted_resource_limited_rate"]) == (1 / 3)
        assert (summary["semantic_accuracy_on_reviewed_selections"]) == (1)
        assert (summary["semantic_wrong_selections"]) == (0)
        assert (summary["failure_stages"]) == ({"resource": 1, "environment": 1})


def test_runtime_failure_with_corrupt_validation_artifact_preserves_primary_failure(tmp_path):
    from acprof.host.preparation.model_coverage import run_sample
    from acprof.host.preparation.model_inspection import ProbePreparationError

    task = candidate(tag="text-generation")
    manifest = {
        "schema_version": 1,
        "sampling": "fixed",
        "weight_basis": "uniform",
        "models": [{"model_id": "example/model", "revision": "a" * 40, "weight": 1}],
    }

    def fail_runtime(_task, output, **_kwargs):
        Path(output, "runtime_validation.json").write_text('{"status":', encoding="utf-8")
        raise ProbePreparationError("build failed")

    with patch("acprof.host.detect.detect_task", return_value=task), patch(
        "acprof.host.automation.check_repository_access"
    ), patch(
        "acprof.host.preparation.model_inspection.validate_model_runtime",
        side_effect=fail_runtime,
    ):
        report = run_sample(manifest, tmp_path / "report", probe="full")

    row = report["rows"][0]
    assert row["error_type"] == "ProbePreparationError"
    assert row["runtime_status"] == "error"


def test_runtime_coverage_serializes_with_formal_measurement_but_static_does_not(tmp_path):
    from acprof.host import run_state
    from acprof.host.preparation.model_coverage import run_sample

    task = candidate(tag="text-generation")
    manifest = {
        "schema_version": 1,
        "sampling": "fixed",
        "weight_basis": "uniform",
        "models": [{"model_id": "example/model", "revision": "a" * 40, "weight": 1}],
    }

    def competing_process():
        script = """
import sys
from pathlib import Path
from acprof.host import run_state
run_state.MEASUREMENT_LOCK_ROOT = Path(sys.argv[1])
run_state.MEASUREMENT_LOCK_NAMESPACE = bytes.fromhex(sys.argv[2])
try:
    with run_state.MeasurementLock():
        pass
except run_state.RunStateError as exc:
    print(exc)
    sys.exit(1)
"""
        return subprocess.run(
            [sys.executable, "-c", script, str(run_state.MEASUREMENT_LOCK_ROOT),
             run_state.MEASUREMENT_LOCK_NAMESPACE.hex()],
            capture_output=True, text=True, timeout=10, check=False,
        )

    def validate(**kwargs):
        attempt = competing_process()
        assert attempt.returncode == 1, attempt.stderr
        assert "测量锁" in attempt.stdout
        with pytest.raises(run_state.RunStateError), run_state.ResultDirectoryLock(kwargs["output_dir"]):
            pytest.fail("model output must stay locked during validation")
        return {"status": "ok", "devices": {}}

    with patch("acprof.host.detect.detect_task", return_value=task), patch(
        "acprof.host.automation.check_repository_access"
    ), patch("acprof.host.preflight.require_collection_host"), patch("acprof.host.preflight.require_native_docker"), patch(
        "acprof.host.images.runtime.prepare_image"
    ), patch("acprof.host.runtime_validation.validate_runtime", side_effect=validate) as validated:
        report = run_sample(manifest, tmp_path / "runtime", probe="full")
    assert report["rows"][0]["runtime_status"] == "ok", report["rows"][0].get("failure")
    validated.assert_called_once()
    assert competing_process().returncode == 0

    with patch("acprof.host.detect.detect_task", return_value=task), patch(
        "acprof.host.run_state.MeasurementLock",
        side_effect=AssertionError("static coverage must not take measurement lock"),
    ):
        run_sample(manifest, tmp_path / "static", probe="none")


def test_busy_coverage_cli_can_retry_the_same_new_output_directory(tmp_path, capsys):
    from acprof.host.run_state import MeasurementLock

    output = tmp_path / "report"
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({
        "schema_version": 1, "sampling": "fixed", "weight_basis": "uniform",
        "models": [{"model_id": "example/model", "revision": "a" * 40, "weight": 1}],
    }))
    args = ["coverage", "run", str(sample), "--output-dir", str(output), "--validate-runtime"]
    with patch("acprof.host.env_utils.bootstrap_project_env"), patch(
        "acprof.host.detect.detect_task", return_value=candidate(tag="text-generation")
    ) as detect, patch("acprof.host.automation.check_repository_access"), patch(
        "acprof.host.preparation.model_inspection.validate_model_runtime", return_value={"status": "ok"}
    ) as validate:
        with MeasurementLock():
            assert main(args) == 2
            assert "测量锁" in capsys.readouterr().err
            assert not output.exists()
            detect.assert_not_called()
        assert main(args) == 0
    validate.assert_called_once()
    report = json.loads((output / "coverage.json").read_bytes())
    assert report["rows"][0]["runtime_status"] == "ok"


def test_static_coverage_keeps_unmeasured_separate_and_requires_reference():
    first = candidate(tag="text-generation")
    second = candidate(tag="fill-mask", hub={"transformers_info": {"pipeline_tag": "text-generation"}})
    manifest = {"schema_version": 1, "sampling": "fixed_regression", "weight_basis": "fixed_test_weights",
                "models": [{"model_id": "example/first", "revision": "a" * 40, "weight": 3,
                            "semantic_reference": {"task": "text-generation", "source": "manual_fixture"}},
                           {"model_id": "example/second", "revision": "a" * 40, "weight": 1}]}
    with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task", side_effect=[first, second]), patch(
            "acprof.host.env_utils.bootstrap_project_env"), contextlib.redirect_stdout(io.StringIO()):
        root = Path(directory)
        source = root / "models.json"
        source.write_text(json.dumps(manifest))
        assert (main(["coverage", "run", str(source), "--output-dir", str(root / "report")])) == (0)
        report = json.loads((root / "report/coverage.json").read_text())
        assert (report["summary"]["sample_weight"]) == (4)
        assert (report["summary"]["weighted_resolution_rate"]) == (0.75)
        assert (report["summary"]["weighted_abstain_rate"]) == (0.25)
        assert (report["summary"]["weighted_runtime_success_rate"]) is None
        assert (report["summary"]["weighted_access_denied_rate"]) is None
        assert (report["summary"]["semantic_accuracy_on_reviewed_selections"]) == (1.0)
        assert (report["rows"][1]["runtime_status"]) == ("not_requested")

@pytest.mark.parametrize(("status", "reason_code"), [
    ("blocked", "runtime_unavailable"),
    ("selection_required", "task_selection_required"),
])
def test_static_blockers_are_not_counted_as_resolved_or_runtime_failures(tmp_path, status, reason_code):
    from acprof.failures import Failure
    from acprof.host.preparation.model_coverage import run_sample

    task = candidate(tag="text-generation")
    task.model_resolution.update(status=status, failure=Failure(
        "resolution", reason_code, "fixture requires a new selection",
        retryability="after_configuration", evidence={"next_action": "Select a compatible registered path."},
    ).to_dict())
    manifest = {"schema_version": 1, "sampling": "fixed", "weight_basis": "uniform",
                "models": [{"model_id": task.model_id, "revision": task.model_revision, "weight": 1}]}
    with patch("acprof.host.detect.detect_task", return_value=task), patch(
        "acprof.host.preparation.model_inspection.validate_model_runtime",
    ) as runtime:
        report = run_sample(manifest, tmp_path / "report", probe="full")

    row = report["rows"][0]
    assert row["inspection_status"] == status
    assert row["resolved"] is False
    assert row["supported"] is False
    assert row["runtime_status"] == "not_run"
    assert row["reason_code"] == reason_code
    assert row["next_action"] == "Select a compatible registered path."
    assert report["summary"]["weighted_resolution_rate"] == 0
    assert report["summary"]["weighted_support_rate"] == 0
    assert report["summary"]["weighted_runtime_success_rate"] == 0
    runtime.assert_not_called()


@pytest.mark.parametrize(("files", "compatibility", "supported"), [
    ((), "unknown", False),
    (("config.json", "model.safetensors"), "supported", True),
])
def test_static_coverage_requires_positive_compatibility_evidence(tmp_path, files, compatibility, supported):
    from acprof.host.preparation.model_coverage import run_sample

    task = candidate(tag="fill-mask", config={"model_type": "bert", "architectures": ["BertForMaskedLM"]})
    task.repository_files = files
    manifest = {"schema_version": 1, "sampling": "fixed", "weight_basis": "uniform",
                "models": [{"model_id": task.model_id, "revision": task.model_revision, "weight": 1}]}
    with patch("acprof.host.detect.detect_task", return_value=task), patch(
        "acprof.host.preparation.model_inspection.validate_model_runtime",
    ) as runtime:
        report = run_sample(manifest, tmp_path / "report")

    row = report["rows"][0]
    assert row["inspection_status"] == "candidate"
    assert row["resolved"] is True
    assert row["compatibility"]["status"] == compatibility
    assert row["compatibility"]["checkpoint_verified"] is False
    assert row["supported"] is supported
    assert row["runtime_status"] == "not_requested"
    assert row["next_action"]
    assert report["summary"]["weighted_support_rate"] == (1 if supported else 0)
    assert report["summary"]["weighted_runtime_success_rate"] is None
    runtime.assert_not_called()


def test_support_failure_updates_the_final_static_decision(tmp_path):
    from acprof.host.preparation.model_coverage import run_sample

    task = candidate(tag="fill-mask", config={"model_type": "bert", "architectures": ["BertForMaskedLM"]})
    task.repository_files = ("config.json", "model.safetensors")
    task.runtime_profile_id = "missing-runtime"
    manifest = {"schema_version": 1, "sampling": "fixed", "weight_basis": "uniform",
                "models": [{"model_id": task.model_id, "revision": task.model_revision, "weight": 1}]}
    with patch("acprof.host.detect.detect_task", return_value=task):
        report = run_sample(manifest, tmp_path / "report")

    row = report["rows"][0]
    assert row["inspection_status"] == "blocked"
    assert row["resolution_status"] == "blocked"
    assert row["resolved"] is False
    assert row["supported"] is False
    assert row["reason_code"] == "runtime_unavailable"
    assert row["runtime_status"] == "not_requested"
    assert report["summary"]["weighted_resolution_rate"] == 0


def test_snapshot_freezes_sha_and_weights_without_creating_semantic_oracles():
    model = {"id": "example/model", "sha": "a" * 40, "downloads": 7, "pipeline_tag": "text-generation",
             "library_name": "transformers"}
    wrapper = {"id": "timm/wrapper", "sha": "b" * 40, "downloads": 9, "pipeline_tag": "text-generation",
               "library_name": "timm"}
    with tempfile.TemporaryDirectory() as directory, patch("huggingface_hub.utils.paginate", return_value=[wrapper, model]), patch(
            "acprof.host.env_utils.bootstrap_project_env"), contextlib.redirect_stdout(io.StringIO()):
        path = Path(directory, "sample.json")
        assert (main(["coverage", "snapshot", "--stratum", "text-generation:transformers",
                               "--output", str(path), "--limit", "1"])) == (0)
        data = json.loads(path.read_text())
        assert (data["models"][0]["revision"]) == ("a" * 40)
        assert (data["models"][0]["weight"]) == (7)
        assert (len(data["models"])) == (1)
        assert ("semantic_reference") not in (data["models"][0])

def test_unpinned_manifest_is_rejected_before_network():
    with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task") as detect, patch(
            "acprof.host.env_utils.bootstrap_project_env"), contextlib.redirect_stderr(io.StringIO()):
        path = Path(directory, "models.json")
        path.write_text(json.dumps({"schema_version": 1, "sampling": "fixed", "weight_basis": "uniform",
                                    "models": [{"model_id": "example/model", "revision": "main", "weight": 1}]}))
        assert (main(["coverage", "run", str(path), "--output-dir", str(Path(directory, "report"))])) == (2)
        detect.assert_not_called()
