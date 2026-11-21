"""Reject changed frozen configuration before restoring or repairing experiment state."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict
from unittest.mock import Mock

import pytest

from acprof.artifacts import atomic_write_json, file_sha256
from acprof.host import run_state
from acprof.host.images.runtime import ImageInfo
from acprof.host.measurement import window_recovery
from acprof.models.inference_configuration import configuration_identity
from tests.support.inference_configuration_fixture import frozen_task, write_frozen_run


@pytest.fixture
def frozen_state(tmp_path, monkeypatch, request):
    monkeypatch.setattr(run_state, "MEASUREMENT_LOCK_ROOT", tmp_path)
    monkeypatch.setattr(run_state, "host_identity", lambda _: {"fixture": "fixed"})
    options = {"cpus": "1", "mems": "4", "gpus": "off", "warmup": 0, "repeat": 1}
    return write_frozen_run(tmp_path / "experiment", options, frozen_task(generation=getattr(request, "param", None)))


def tree_bytes(directory):
    return {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("frozen_state", [{"max_length": 10}], indirect=True)
def test_resume_requires_proof_that_the_frozen_total_was_effective(frozen_state, monkeypatch):
    state = frozen_state
    assert run_state.validate_resume(state.directory, state.data, project_dir=".") != "preparation"
    validation_path = state.layout.path("runtime_validation.json")
    validation = json.loads(validation_path.read_text())
    del validation["devices"]["off"]["consumer_receipt"]["calls"][0]["parameters"]["max_new_tokens"]
    atomic_write_json(validation_path, validation)
    meta_path = state.layout.path("static_meta.json")
    meta = json.loads(meta_path.read_text())
    meta["runtime_validation"] = validation
    atomic_write_json(meta_path, meta)
    state.data["artifacts"][str(meta_path.relative_to(state.directory))] = file_sha256(meta_path)
    atomic_write_json(state.path, state.data)
    before = tree_bytes(state.directory)
    repair = Mock()
    monkeypatch.setattr(window_recovery, "repair_result_projections", repair)
    with pytest.raises(run_state.RunStateError, match="max_new_tokens"):
        run_state.validate_resume(state.directory, run_state.load_run_state(state.directory), project_dir=".", repair=True)
    repair.assert_not_called()
    assert tree_bytes(state.directory) == before


@pytest.mark.parametrize("change", ["digest", "values", "model_binding", "plan", "metadata", "receipt"])
def test_changed_configuration_evidence_precedes_projection_repair(frozen_state, monkeypatch, change):
    state = frozen_state
    data = copy.deepcopy(state.data)
    runtime = data["runtime"]
    configuration = runtime["task"]["inference_configuration"]
    if change == "digest":
        runtime["inference_configuration_sha256"] = "b" * 64
    elif change == "values":
        configuration["loading"]["off"]["dtype"] = "FP16"
    elif change == "model_binding":
        configuration["model"]["model_id"] = "fixture/another"
        configuration["identity_sha256"] = configuration_identity(configuration)
        runtime["inference_configuration_sha256"] = configuration["identity_sha256"]
    elif change == "plan":
        path = state.layout.path("input_scale_plan.json")
        plan = json.loads(path.read_text())
        plan["entries"][0]["payload"]["text"] = "changed workload"
        atomic_write_json(path, plan)
    elif change == "metadata":
        path = state.layout.path("static_meta.json")
        metadata = json.loads(path.read_text())
        metadata["inference_configuration"] = frozen_task(tokens=11).inference_configuration
        atomic_write_json(path, metadata)
    else:
        path = state.layout.path("runtime_validation.json")
        validation = json.loads(path.read_text())
        validation["devices"]["off"]["consumer_receipt"]["status"] = "not_run"
        atomic_write_json(path, validation)
    before = tree_bytes(state.directory)
    repair = Mock(side_effect=AssertionError("recovery writes must not run"))
    monkeypatch.setattr(window_recovery, "repair_result_projections", repair)

    with pytest.raises(run_state.RunStateError, match="inference_configuration|冻结推理配置"):
        run_state.validate_resume(state.directory, data, project_dir=".", repair=True)

    repair.assert_not_called()
    assert tree_bytes(state.directory) == before


@pytest.mark.parametrize(("field", "value"), [
    ("device", "cuda"), ("task_id", "another-task"), ("inference_configuration_sha256", "b" * 64),
])
def test_resume_rejects_foreign_loading_evidence_even_when_saved_reports_agree(frozen_state, monkeypatch, field, value):
    state = frozen_state
    validation_path = state.layout.path("runtime_validation.json")
    validation = json.loads(validation_path.read_text())
    report = validation["devices"]["off"]
    report["configuration_validation"][field] = value
    if field == "device":
        report["device"] = value
    atomic_write_json(validation_path, validation)
    metadata_path = state.layout.path("static_meta.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["runtime_validation"] = validation
    atomic_write_json(metadata_path, metadata)
    before = tree_bytes(state.directory)
    repair = Mock(side_effect=AssertionError("recovery writes must not run"))
    monkeypatch.setattr(window_recovery, "repair_result_projections", repair)

    with pytest.raises(run_state.RunStateError, match=field):
        run_state.validate_resume(state.directory, state.data, project_dir=".", repair=True)

    repair.assert_not_called()
    assert tree_bytes(state.directory) == before


def test_valid_resume_reuses_frozen_object_without_auto_resolution(frozen_state, monkeypatch):
    state = frozen_state
    snapshot = copy.deepcopy(state.data["runtime"]["task"])
    restore = Mock(return_value=ImageInfo("sha256:" + "b" * 64))
    monkeypatch.setattr("acprof.host.runtimes.service.execution_runtime", lambda _: type("Runtime", (), {"restore": restore})())
    assert run_state.validate_resume(state.directory, state.data, project_dir=".") != "preparation"

    restored, *_ = state.restore_runtime()

    assert asdict(restored) == {**snapshot, "repository_sources": {}}
    restore.assert_called_once()


def test_direct_restore_rejects_mutated_configuration_before_runtime_access(frozen_state, monkeypatch):
    state = frozen_state
    state.data["runtime"]["task"]["inference_configuration"]["loading"]["off"]["dtype"] = "FP16"
    runtime = Mock(side_effect=AssertionError("runtime must not be accessed"))
    monkeypatch.setattr("acprof.host.runtimes.service.execution_runtime", runtime)

    with pytest.raises(run_state.RunStateError, match="inference_configuration"):
        state.restore_runtime()
    runtime.assert_not_called()
