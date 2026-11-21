"""A confirmed external workload must be the one used to build the input plan."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from acprof import workloads
from acprof.application.run import inference_workload, resource_matrix
from acprof.artifacts import read_input_scale_plan
from acprof.experiments.configuration import RunConfig
from acprof.host.images.runtime import ImageInfo
from acprof.host.input_plan import plan_input_scales
from acprof.host.preparation.inference_configuration import (
    InferenceConfigurationReviewRequired,
    prepare_execution_configuration,
)
from tests.support.task_configuration_fixture import routable_task, task_model


@pytest.fixture(params=["cv", "multimodal"])
def vision_files(request, tmp_path):
    if request.param == "cv":
        config = {"model_type": "resnet", "architectures": ["ResNetForImageClassification"]}
        task = task_model("cv", "image-classification", "transformers_pipeline", config)
        task.repository_files = ("config.json", "model.safetensors")
        task.repository_metadata = {"config.json": config}
        scale, digest_field = "0.25", "workload_spec_sha256"
    else:
        task = routable_task("multimodal")
        scale, digest_field = "32", "manifest_sha256"
    source = tmp_path / "inputs" / "vision.json"
    source.parent.mkdir()
    for color in ("red", "blue"):
        Image.new("RGB", (2, 2), color).save(source.with_name(color + ".png"))
    source.write_text('{"schema_version":1,"image_path":"red.png"}', encoding="utf-8")
    output = tmp_path / "results"
    run_config = replace(RunConfig.smoke(task.model_id, runtime="docker"), output_dir=str(output), input_scales=scale)
    return SimpleNamespace(task=task, config=run_config, source=source, output=output, digest_field=digest_field)


def _confirmed_plan(vision, workload_spec):
    args = replace(vision.config, workload_spec=workload_spec).to_namespace()
    axes = resource_matrix(args)
    workflow = SimpleNamespace(interactive=False)
    with pytest.raises(InferenceConfigurationReviewRequired) as review:
        prepare_execution_configuration(vision.task, args, workflow, workload=inference_workload(args, *axes))
    args.confirm_inference = review.value.identity_sha256
    args.inference_config = str(review.value.requests_path)
    frozen = prepare_execution_configuration(vision.task, args, workflow, workload=inference_workload(args, *axes))

    def plan(path=workload_spec):
        return plan_input_scales(frozen, ImageInfo(tag="fixture"), *axes, args.batch_size, args.output_dir,
                                 input_scales=args.input_scales, workload_spec_path=path,
                                 input_scale_policy=args.input_scale_policy)

    return plan, review.value.identity_sha256


def test_confirmed_plan_accepts_equivalent_json_and_records_consumed_bytes(vision_files):
    source = vision_files.source
    plan_inputs, identity = _confirmed_plan(vision_files, str(source))
    source.write_text('{\n  "image_path": "red.png",\n  "schema_version": 1\n}\n', encoding="utf-8")

    planned = plan_inputs()

    plan = read_input_scale_plan(planned.plan_file, expected_sha256=planned.plan_sha256)
    assert plan["inference_configuration_sha256"] == identity
    metadata = plan["workload"]
    assert metadata[vision_files.digest_field] == hashlib.sha256(source.read_bytes()).hexdigest()
    asset_digest = hashlib.sha256(source.with_name("red.png").read_bytes()).hexdigest()
    if vision_files.task.task_family == "cv":
        assert metadata["source_sha256"] == [asset_digest]
    else:
        assert [asset["sha256"] for asset in metadata["assets"]] == [asset_digest]


@pytest.mark.parametrize("change", ["before_generator", "during_generator"])
def test_confirmed_plan_rejects_changed_relative_image_before_publication(vision_files, monkeypatch, change):
    source = vision_files.source
    plan_inputs, _ = _confirmed_plan(vision_files, str(source))
    original = source.read_bytes()
    changed = json.dumps({"schema_version": 1, "image_path": "blue.png"}).encode()
    if change == "before_generator":
        source.write_bytes(changed)
    else:
        factory = workloads.get_generator

        def read_changed_file(*args, **kwargs):
            source.write_bytes(changed)
            try:
                return factory(*args, **kwargs)
            finally:
                source.write_bytes(original)

        monkeypatch.setattr(workloads, "get_generator", read_changed_file)

    with pytest.raises(ValueError, match="workload specification changed"):
        plan_inputs()
    assert not list(vision_files.output.rglob("input_scale_plan.json"))


@pytest.mark.parametrize("confirm_source", [True, False], ids=["omitted", "added"])
def test_plan_cannot_add_or_drop_a_confirmed_external_source(vision_files, confirm_source):
    path = str(vision_files.source)
    plan_inputs, _ = _confirmed_plan(vision_files, path if confirm_source else "")
    previous = plan_inputs()
    previous_content = Path(previous.plan_file).read_bytes()

    with pytest.raises(ValueError, match="workload specification changed"):
        plan_inputs("" if confirm_source else path)
    assert Path(previous.plan_file).read_bytes() == previous_content


def test_confirmed_source_requires_a_digest_from_the_generator(vision_files, monkeypatch):
    plan_inputs, _ = _confirmed_plan(vision_files, str(vision_files.source))
    monkeypatch.setattr(workloads, "get_generator", lambda *args, **kwargs: object())

    with pytest.raises(ValueError, match="generator did not report.*SHA-256"):
        plan_inputs()
    assert not list(vision_files.output.rglob("input_scale_plan.json"))
