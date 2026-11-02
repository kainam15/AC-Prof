"""Runtime implementations own run-specific bindings without mutating saved identities."""
from copy import deepcopy
from dataclasses import asdict
from unittest.mock import Mock

import pytest

from acprof.host.detect import TaskInfo
from acprof.host.images.runtime import ImageInfo
from acprof.host.run_state import RunState
from acprof.host.runtimes import environment, service
from acprof.host.runtimes.contracts import ProcessInfo, runtime_identity, runtime_kind
from acprof.host.runtimes.docker import DockerRuntime
from acprof.host.runtimes.process import ProcessRuntime


def process_artifact():
    return ProcessInfo(
        "/environment/venv/bin/python", "/environment/code", "/models/pinned",
        "fixed-runtime-identity",
        {"logs_dir": "/old/output/logs", "environment_id": "fixed-environment",
         "gpu_device": {"cuda_visible_devices": "GPU-recorded"},
         "model_store": {"plan_sha256": "fixed-model"}},
    )


def test_prepare_process_owns_gpu_binding_without_mutating_inputs(tmp_path, monkeypatch):
    prepared = process_artifact()
    original = deepcopy(prepared.runtime_environment)
    prepare = Mock(return_value=prepared)
    monkeypatch.setattr(environment, "prepare_process", prepare)
    selected = {"cuda_visible_devices": "GPU-selected"}
    task = object()

    result = ProcessRuntime().prepare(
        task, "/project", tmp_path, gpu_device=selected, timeout_seconds=12,
    )

    prepare.assert_called_once_with(task, "/project", tmp_path, rebuild=False, timeout_seconds=12)
    assert result.runtime_environment["gpu_device"] == selected
    assert result.runtime_environment["gpu_device"] is not selected
    assert prepared.runtime_environment == original
    assert result.environment_id == prepared.environment_id
    assert result.model_path == prepared.model_path


def test_restore_rebinds_logs_and_preserves_frozen_gpu_and_snapshot(tmp_path, monkeypatch):
    snapshot = asdict(process_artifact())
    original = deepcopy(snapshot)
    restore = Mock(side_effect=lambda value: ProcessInfo(**value))
    monkeypatch.setattr(environment, "restore_process", restore)

    restored = ProcessRuntime().restore(snapshot, output_dir=tmp_path / "moved")

    restore.assert_called_once_with(snapshot)
    assert restored.runtime_environment["logs_dir"] == str(tmp_path / "moved" / "logs")
    assert restored.runtime_environment["gpu_device"] == {"cuda_visible_devices": "GPU-recorded"}
    assert restored.environment_id == "fixed-runtime-identity"
    assert snapshot == original
    assert restored.runtime_environment is not snapshot["runtime_environment"]


def test_restore_identity_failure_propagates_before_log_rebinding(tmp_path, monkeypatch):
    snapshot = asdict(process_artifact())
    original = deepcopy(snapshot)
    error = ValueError("recorded Process/model identity changed")
    monkeypatch.setattr(environment, "restore_process", Mock(side_effect=error))

    with pytest.raises(ValueError, match="identity changed") as caught:
        ProcessRuntime().restore(snapshot, output_dir=tmp_path)

    assert caught.value is error
    assert snapshot == original


def test_run_state_supplies_current_directory_to_runtime_restore(tmp_path, monkeypatch):
    from acprof.tasks import interfaces

    state = object.__new__(RunState)
    state.directory = tmp_path / "resumed"
    task = TaskInfo("fixture/model", "fill-mask", "nlp", "transformers_pipeline",
                    "transformers", "a" * 40, "fixture")
    state.data = {"runtime": {
        "task": asdict(task), "image": asdict(process_artifact()), "kind": "process",
        "planned": {"scales": [1.0], "source": "fixture"},
        "compute_plan": "", "execution_plan": "",
    }}
    original = deepcopy(state.data)
    runtime = ProcessRuntime()
    monkeypatch.setattr(service, "execution_runtime", Mock(return_value=runtime))
    monkeypatch.setattr(interfaces, "validate_execution_identity", Mock())
    monkeypatch.setattr(environment, "restore_process", lambda value: ProcessInfo(**value))

    restored_task, restored, planned, compute, execution = state.restore_runtime()

    assert restored_task.model_revision == task.model_revision
    assert restored.runtime_environment["logs_dir"] == str(state.directory / "logs")
    assert restored.runtime_environment["gpu_device"] == {"cuda_visible_devices": "GPU-recorded"}
    assert state.data == original
    assert planned.scales == [1.0]
    assert (compute, execution) == ("", "")


def test_docker_preparation_keeps_interface_probe_and_image_identity(tmp_path, monkeypatch):
    from acprof.host.images import runtime as images
    from acprof.host.preparation import interface_probe

    events = []
    task, image = object(), ImageInfo("sha256:" + "a" * 64)
    monkeypatch.setattr(interface_probe, "probe_interface",
                        lambda *args, **kwargs: events.append("probe"))
    prepare = Mock(side_effect=lambda *args, **kwargs: events.append("image") or image)
    monkeypatch.setattr(images, "prepare_image", prepare)

    result = DockerRuntime().prepare(task, "/project", tmp_path, gpu_device={})

    assert result is image
    assert events == ["probe", "image"]
    prepare.assert_called_once_with(task, "/project", reuse_existing=True)


def test_docker_restore_keeps_immutable_identity_check(tmp_path, monkeypatch):
    from acprof.host.images import runtime as images

    snapshot = asdict(ImageInfo("sha256:" + "a" * 64, runtime_environment={"identity": "fixed"}))
    verify = Mock()
    monkeypatch.setattr(images, "require_image_identity", verify)

    restored = DockerRuntime().restore(snapshot, output_dir=tmp_path)

    verify.assert_called_once_with(snapshot["tag"], snapshot["runtime_environment"])
    assert asdict(restored) == snapshot
    with pytest.raises(ValueError, match="immutable Docker image ID"):
        DockerRuntime().restore({"tag": "mutable:latest"}, output_dir=tmp_path)
    assert verify.call_count == 1


@pytest.mark.parametrize(
    ("artifact", "kind", "identity"),
    [(process_artifact(), "process", "fixed-runtime-identity"),
     (ImageInfo("sha256:" + "a" * 64), "docker", "sha256:" + "a" * 64)],
)
def test_public_artifact_identity_supports_both_runtimes(artifact, kind, identity):
    assert runtime_kind(artifact) == kind
    assert runtime_identity(artifact) == identity
