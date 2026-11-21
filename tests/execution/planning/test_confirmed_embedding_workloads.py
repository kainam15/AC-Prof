"""Trace the reviewed sentence encoder workload through collection and recovery."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.application.outcome import RunOutcome
from acprof.application.run import inference_workload, resource_matrix
from acprof.application.task_runs import TaskCollection
from acprof.artifacts import atomic_write_json, file_sha256, read_input_scale_plan, read_json_object
from acprof.container.handlers.sentence_encoders import SentenceEncoderHandler
from acprof.container.inference_configuration import (
    attach_configuration_context,
    capture_generation_consumption,
    verify_model_session_configuration,
)
from acprof.container.model_session import ModelLoadConfig
from acprof.experiments.configuration import RunConfig
from acprof.host import run_state
from acprof.host.images.runtime import ImageInfo
from acprof.host.input_plan import PlannedInputScales, families, plan_input_scales
from acprof.host.input_plan.materialize import write_scale_plan_file
from acprof.host.measurement import window_recovery
from acprof.host.preparation.inference_configuration import (
    InferenceConfigurationReviewRequired,
    prepare_execution_configuration,
)
from acprof.host.preparation.task_selection import select_tasks
from tests.support.task_selection_fixtures import sentence_encoder_task
from tests.workloads.nlp.test_nlp_handler import FakeTokenizer


def _confirm(tmp_path, interface, params):
    model = sentence_encoder_task(interface)
    source = tmp_path / "embedding.json"
    source.write_text(json.dumps({"schema_version": 1, "task": interface, "params": params,
                                  "input_budget": {"max_text_characters": 4096}}))
    config = replace(RunConfig.smoke(model.model_id, runtime="docker"), task=interface,
                     input_scales="8", workload_spec=str(source), output_dir=str(tmp_path / "results"))
    args = config.to_namespace()
    workflow = SimpleNamespace(interactive=False)
    task = select_tasks(model, args, workflow)
    axes = resource_matrix(args)
    with pytest.raises(InferenceConfigurationReviewRequired) as review:
        prepare_execution_configuration(task, args, workflow, workload=inference_workload(args, *axes))
    config = replace(config, confirm_inference=review.value.identity_sha256,
                     inference_config=str(review.value.requests_path))
    args = config.to_namespace()
    frozen = prepare_execution_configuration(task, args, workflow, workload=inference_workload(args, *axes))
    assert frozen.inference_configuration["workload"]["parameter_defaults"] == params
    return TaskCollection(config, frozen)


@pytest.mark.parametrize(("interface", "params"), [
    ("feature-extraction", {"prompt_name": "search", "normalize_embeddings": True}),
    ("sentence-similarity", {"prompt": "passage: ", "normalize_embeddings": False}),
    ("feature-extraction", {"prompt": "", "normalize_embeddings": False}),
    ("sentence-similarity", {}),
])
def test_confirmed_parameters_reach_encode_and_survive_collection_restore(tmp_path, monkeypatch, interface, params):
    collection = _confirm(tmp_path, interface, params)
    source = Path(collection.config.workload_spec)
    original = source.read_bytes()
    pipe = SimpleNamespace(tokenizer=FakeTokenizer("[MASK]", model_max_length=64),
                           prompts={"search": "query: "}, default_prompt_name="search",
                           encode=Mock(side_effect=lambda texts, **kwargs: list(range(len(texts)))),
                           similarity=Mock(return_value=[[1., 1.]]))
    handler = SentenceEncoderHandler("sentence")
    context = {"task_type": interface, "pipeline": pipe}
    monkeypatch.setattr(families, "start_probe_session", Mock())
    monkeypatch.setattr(families, "stop_service_session", Mock())

    def probe_payload(_session, payload, _desc):
        processed = handler.preprocess(context, payload)
        return {"payload": payload, "effective_input_scale": processed["_effective_input_scale"],
                "truncated_by_limit": processed["_truncated_by_limit"]}

    monkeypatch.setattr(families, "post_probe_payload", probe_payload)
    used = []

    def execute(child, task):
        args = child.to_namespace()
        axes = resource_matrix(args)
        task = prepare_execution_configuration(task, args, SimpleNamespace(interactive=False),
                                               workload=inference_workload(args, *axes), task_child=True)
        planned = plan_input_scales(task, ImageInfo(tag="fixture"), *axes, args.batch_size,
                                    str(child.result_dir(tmp_path)), input_scales=args.input_scales,
                                    workload_spec_path=args.workload_spec)
        plan = read_input_scale_plan(planned.plan_file, expected_sha256=planned.plan_sha256)
        payload = plan["entries"][0]["payload"]
        handler.predict(context, handler.preprocess(context, payload))
        used.append((args.workload_spec, plan["inference_configuration_sha256"], plan["workload"], payload,
                     pipe.encode.call_args.kwargs))
        return RunOutcome("ok")

    assert collection.run(execute).status == "ok"
    restored = TaskCollection.restore(replace(collection.config, task_run_id=collection.run_id, resume=True))
    assert restored.run(execute).status == "ok"
    assert len(used) == 2
    assert used[0] == used[1]
    path, identity, metadata, payload, kwargs = used[0]
    assert identity == collection.config.confirm_inference
    assert read_json_object(path)["params"] == metadata["params"] == payload["params"] == params
    assert payload["input_budget"]["max_text_characters"] == 4096
    options = {key: kwargs[key] for key in ("prompt", "prompt_name", "normalize_embeddings") if key in kwargs}
    assert options == params
    assert Path(path) != source
    assert source.read_bytes() == original


@pytest.mark.parametrize(("params", "message"), [
    ({"normalize_embeddings": "false"}, "must be a boolean"),
    ({"prompt_name": "search", "prompt": ""}, "either prompt or prompt_name"),
    ({"max_new_tokens": 8}, "unsupported params"),
])
def test_invalid_source_options_stop_before_publishing_a_confirmation(tmp_path, params, message):
    with pytest.raises(ValueError, match=message):
        _confirm(tmp_path, "feature-extraction", params)
    assert not list(tmp_path.rglob(".inference-configurations/*.json"))
    assert not list(tmp_path.rglob("task_workloads/*.json"))


@pytest.fixture
def interrupted_encoder(tmp_path, monkeypatch):
    params = {"prompt": "query: ", "normalize_embeddings": True}
    task = _confirm(tmp_path, "feature-extraction", params).task
    frozen = task.inference_configuration
    config = ModelLoadConfig(task.model_id, task.task_family, task.pipeline_tag, task.runtime_backend, task.model_revision,
                             inference_configuration=frozen, inference_task_id=task.selected_tasks[0]["task_id"])
    session = SimpleNamespace(config=config, device="cpu", context={})
    attach_configuration_context(config, "cpu", session.context)
    with capture_generation_consumption(session) as receipt:
        report = verify_model_session_configuration(session)
    digest = frozen["identity_sha256"]
    validation = {"schema_version": 1, "status": "ok", "inference_configuration_sha256": digest,
                  "devices": {"off": {"status": "ok", "device": "cpu", "inference_configuration_sha256": digest,
                                      "configuration_validation": report, "consumer_receipt": receipt}}}
    monkeypatch.setattr(run_state, "MEASUREMENT_LOCK_ROOT", tmp_path)
    monkeypatch.setattr(run_state, "host_identity", lambda _: {"fixture": "fixed"})
    options = {"cpus": "1", "mems": "4", "gpus": "off", "warmup": 0, "repeat": 1}
    state = run_state.RunState(tmp_path / "experiment", options, resume=False, project_dir=".")
    try:
        path = state.layout.path("input_scale_plan.json")
        entries = [{"input_scale": scale, "payload": {"text": "test", "params": params}} for scale in (8, 16)]
        sha = write_scale_plan_file(str(path), task, entries, workload={"params": params})
        plan = PlannedInputScales([8, 16], "manual", plan_file=str(path), plan_sha256=sha, workload={"params": params})
        atomic_write_json(state.layout.path("runtime_validation.json"), validation)
        atomic_write_json(state.layout.path("static_meta.json"), {
            "schema_version": 8, "inference_configuration": frozen, "runtime_validation": validation,
        })
        state.bind_runtime(task, ImageInfo("sha256:" + "b" * 64), plan, "", "")
    finally:
        state.close("interrupted")
    run_state.validate_resume(state.directory, state.data, project_dir=".")
    return state


@pytest.mark.parametrize("lost_from", ["entire_plan", "second_payload", "metadata"])
def test_resume_rejects_matching_hashes_for_a_plan_that_lost_encoder_options(interrupted_encoder, monkeypatch, lost_from):
    state = interrupted_encoder
    path = state.layout.path("input_scale_plan.json")
    plan = read_input_scale_plan(path)
    if lost_from in {"entire_plan", "metadata"}:
        plan["workload"]["params"] = {}
    if lost_from == "entire_plan":
        for entry in plan["entries"]:
            entry["payload"]["params"] = {}
    if lost_from == "second_payload":
        plan["entries"][1]["payload"]["params"].pop("normalize_embeddings")
    # Older runs could record the wrong options with consistent hashes throughout.
    atomic_write_json(path, plan)
    expected_sha = file_sha256(path)
    snapshot = state.data["runtime"]["planned"]
    snapshot.update(workload=plan["workload"], plan_sha256=expected_sha)
    state.data["artifacts"][snapshot["plan_file"]] = expected_sha
    atomic_write_json(state.path, state.data)
    before = state.path.read_bytes()
    repair = Mock()
    monkeypatch.setattr(window_recovery, "repair_result_projections", repair)

    with pytest.raises(run_state.RunStateError, match="embedding"):
        run_state.validate_resume(state.directory, run_state.load_run_state(state.directory), project_dir=".", repair=True)
    repair.assert_not_called()
    assert state.path.read_bytes() == before
    assert file_sha256(path) == expected_sha
