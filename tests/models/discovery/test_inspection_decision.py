"""Static inspection reports actionable decisions without inventing runtime evidence."""
from __future__ import annotations

import json

import pytest

from acprof.failures import Failure
from acprof.host.detect import TaskInfo
from acprof.host.preparation.model_inspection import explain_resolution
from acprof.host.preparation.task_support import TaskSupportError, require_task_support
from acprof.models.contract import write_model_resolution


def _task(**changes):
    values = {
        "model_id": "fixture/static-inspection",
        "pipeline_tag": "fill-mask",
        "task_family": "nlp",
        "runtime_backend": "transformers_pipeline",
        "library_name": "transformers",
        "model_revision": "a" * 40,
        "detection_method": "manual",
        "model_config": {"model_type": "bert", "architectures": ["BertForMaskedLM"]},
        "repository_files": ("config.json", "model.safetensors"),
        "model_resolution": {"status": "candidate"},
    }
    values.update(changes)
    return TaskInfo(**values)


def test_missing_runtime_is_blocked_in_exception_display_and_export(tmp_path):
    task = _task(runtime_profile_id="missing-runtime")
    with pytest.raises(TaskSupportError, match="No registered runtime") as caught:
        require_task_support(task)
    assert caught.value.failure.reason_code == "runtime_unavailable"
    assert task.model_resolution["status"] == "blocked"
    assert task.model_resolution["failure"]["evidence"]["next_action"]
    output = explain_resolution(task)
    assert "Static Resolution: blocked" in output
    assert "Reason: runtime_unavailable" in output
    assert "Runtime Validation: not_started" in output
    saved = json.loads(write_model_resolution(task, tmp_path).read_text())
    assert saved["failure"]["reason_code"] == "runtime_unavailable"
    assert saved["status"] == "blocked"


@pytest.mark.parametrize(("status", "reason_code"), [
    ("selection_required", "task_selection_required"),
    ("ambiguous", "evidence_conflict"),
    ("needs_configuration", "model_contract_required"),
])
def test_unresolved_decisions_keep_their_specific_reason(status, reason_code):
    task = _task(model_resolution={"status": status, "missing": ["review required"]})
    with pytest.raises(TaskSupportError) as caught:
        require_task_support(task)
    assert caught.value.failure.reason_code == reason_code
    assert task.model_resolution["status"] == status
    assert caught.value.failure.evidence["next_action"]


def test_metadata_error_is_not_classified_as_an_unsupported_task():
    task = _task(metadata_errors=("config.json: invalid JSON",))
    with pytest.raises(TaskSupportError, match="config.json") as caught:
        require_task_support(task)
    assert caught.value.failure.reason_code == "metadata_invalid"
    assert task.model_resolution["status"] == "blocked"


def test_original_typed_metadata_failure_and_raw_digest_survive_preflight():
    original = Failure("resolution", "metadata_invalid", "config.json /limit: Infinity",
                       evidence={"raw_sha256": "b" * 64, "json_pointer": "/limit"})
    task = _task(metadata_errors=("invalid",), model_resolution={
        "status": "blocked", "failure": original.to_dict(),
    })
    with pytest.raises(TaskSupportError, match="/limit") as caught:
        require_task_support(task)
    assert caught.value.failure.evidence["raw_sha256"] == "b" * 64
    assert task.model_resolution["failure"]["reason_code"] == "metadata_invalid"
    assert "preflight_input_status" not in task.model_resolution["failure"]["evidence"]
    with pytest.raises(TaskSupportError, match="/limit"):
        require_task_support(task)
    assert task.model_resolution["failure"]["evidence"]["raw_sha256"] == "b" * 64


def test_unknown_architecture_remains_unknown_after_static_preflight():
    task = _task(model_config={})
    require_task_support(task)
    assert task.model_resolution["compatibility"]["status"] == "unknown"
    assert task.model_resolution["compatibility"]["evidence_level"] == "static"
    output = explain_resolution(task)
    assert "Static compatibility: unknown" in output
    assert "Runtime Validation: not_started" in output


def test_runtime_correction_rechecks_the_same_task_without_a_stale_failure():
    task = _task(runtime_profile_id="missing-runtime")
    with pytest.raises(TaskSupportError):
        require_task_support(task)
    task.runtime_profile_id = ""
    require_task_support(task)
    assert task.model_resolution["status"] == "candidate"
    assert task.model_resolution["compatibility"]["status"] == "supported"
    assert "failure" not in task.model_resolution


def test_batch_correction_rechecks_the_same_task_without_a_stale_failure():
    task = _task(pipeline_tag="image-to-text", task_family="cv", model_config={})
    with pytest.raises(TaskSupportError, match="--batch-size 1"):
        require_task_support(task, batch_size=2)
    require_task_support(task, batch_size=1)
    assert task.model_resolution["status"] == "candidate"
    assert "failure" not in task.model_resolution


@pytest.mark.parametrize("devices", [("on",), ("cpu", "gpu"), ("cuda:0",)])
def test_cpu_only_backend_reports_device_failure_and_can_retry_on_cpu(devices, tmp_path):
    task = _task(pipeline_tag="tabular-classification", task_family="structured",
                 runtime_backend="onnxruntime", library_name="onnx", model_config={},
                 repository_files=("iris.onnx",), model_spec={
                     "schema_version": 1, "format": "onnxruntime", "task": "tabular-classification",
                     "model_file": "iris.onnx", "feature_dim": 4,
                 })
    require_task_support(task, devices=["cpu"])
    with pytest.raises(TaskSupportError) as caught:
        require_task_support(task, devices=devices)

    failure = caught.value.failure
    assert (failure.stage, failure.reason_code, failure.device) == ("preflight", "runtime_unavailable", "gpu")
    assert failure.evidence["extension_id"] == "onnxruntime-cpu"
    assert failure.evidence["capability"] == "cuda"
    assert "--gpus off" in failure.evidence["next_action"]
    saved = json.loads(write_model_resolution(task, tmp_path).read_text())
    assert saved["status"] == "blocked"
    assert saved["failure"] == failure.to_dict()
    assert saved["requested_devices"] == list(devices)
    assert "compatibility" not in saved
    assert "precision" not in saved

    require_task_support(task, devices=["off"])
    assert "failure" not in task.model_resolution
    assert task.model_resolution["precision"]["off"]["device"] == "cpu"


def test_missing_artifact_evidence_keeps_overall_compatibility_unknown():
    task = _task(repository_files=())
    require_task_support(task)
    compatibility = task.model_resolution["compatibility"]
    assert compatibility["status"] == "unknown"
    assert compatibility["interface_status"] == "supported"
    assert compatibility["artifact_status"] == "unknown"
    assert "Static compatibility: unknown" in explain_resolution(task)


def test_fresh_failure_does_not_retain_previous_success_evidence():
    task = _task()
    require_task_support(task)
    assert task.model_resolution["compatibility"]["status"] == "supported"
    task.runtime_profile_id = "missing-runtime"
    with pytest.raises(TaskSupportError):
        require_task_support(task)
    assert "compatibility" not in task.model_resolution
    assert "precision" not in task.model_resolution


def test_candidate_is_never_relabelled_as_resolved():
    output = explain_resolution(_task())
    assert "Static Resolution: candidate" in output
    assert "Interface Resolution: candidate" in output
    assert "Interface Resolution: resolved" not in output
    assert "Measurement: not_started" in output
