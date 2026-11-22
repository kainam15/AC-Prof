"""Post-hoc work restores the original task and frozen configuration, never defaults."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from acprof.host.configuration_transport import CONFIGURATION_ENV, inference_environment
from acprof.host.posthoc.context import PosthocError, load_result_context
from acprof.models.inference_configuration import decode_inference_configuration
from tests.measurement.profiling.test_posthoc import TestPosthocProfile as _PosthocFixture
from tests.support.inference_configuration_fixture import frozen_task


@pytest.fixture
def frozen_result(tmp_path):
    _PosthocFixture()._write_fixture(tmp_path, include_gpu=False)
    task = frozen_task()
    plan_path = tmp_path / "input_scale_plan.json"
    plan = json.loads(plan_path.read_text())
    plan.update(model_id=task.model_id, pipeline_tag=task.pipeline_tag,
                task_workload=copy.deepcopy(task.selected_tasks[0]),
                inference_configuration_sha256=task.inference_configuration["identity_sha256"])
    plan_path.write_text(json.dumps(plan))
    metadata_path = tmp_path / "static_meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        schema_version=8, model_name=task.model_id, model_revision=task.model_revision,
        pipeline_tag=task.pipeline_tag, runtime_backend=task.runtime_backend,
        runtime_environment={"profile_id": task.runtime_profile_id, "adapter": task.model_adapter},
        inference_configuration=copy.deepcopy(task.inference_configuration),
        input_scale_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    metadata_path.write_text(json.dumps(metadata))
    return tmp_path, task


def test_posthoc_keeps_frozen_configuration_and_active_task(frozen_result):
    directory, task = frozen_result
    before = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}

    context = load_result_context(directory)

    assert context.task_info.inference_configuration == task.inference_configuration
    assert context.task_info.selected_tasks == task.selected_tasks
    encoded = inference_environment(context.task_info)[CONFIGURATION_ENV]
    assert decode_inference_configuration(encoded) == task.inference_configuration
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("change", ["missing_task", "changed_generation", "foreign_configuration"])
def test_posthoc_rejects_changed_bindings_even_with_matching_plan_byte_hash(frozen_result, change):
    directory, _task = frozen_result
    plan_path = directory / "input_scale_plan.json"
    plan = json.loads(plan_path.read_text())
    if change == "missing_task":
        plan.pop("task_workload")
    elif change == "changed_generation":
        plan["task_workload"]["config"]["generation"]["max_new_tokens"] = 99
    else:
        plan["inference_configuration_sha256"] = "b" * 64
    plan_path.write_text(json.dumps(plan))
    metadata_path = directory / "static_meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["input_scale_plan_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    before = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}

    with pytest.raises(PosthocError, match="frozen inference configuration"):
        load_result_context(directory)

    assert {path: path.read_bytes() for path in before} == before

@pytest.mark.parametrize("variant", ["fp16", "checkpoints/tiny/"])
def test_posthoc_replays_the_original_checkpoint_variant(frozen_result, variant):
    from acprof.models.inference_configuration import configuration_identity

    directory, _ = frozen_result
    meta_path = directory / "static_meta.json"
    plan_path = directory / "input_scale_plan.json"
    meta = json.loads(meta_path.read_text())
    config = meta["inference_configuration"]
    config["model"]["variant"] = variant
    config["identity_sha256"] = configuration_identity(config)
    plan = json.loads(plan_path.read_text())
    plan["inference_configuration_sha256"] = config["identity_sha256"]
    plan_path.write_text(json.dumps(plan))
    meta["input_scale_plan_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    meta["model_resolution"] = {"model_variant": variant}
    meta_path.write_text(json.dumps(meta))
    context = load_result_context(directory)
    assert inference_environment(context.task_info)["ACPROF_MODEL_VARIANT"] == variant
    meta["model_resolution"]["model_variant"] = "other"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(PosthocError, match="model.variant"):
        load_result_context(directory)
