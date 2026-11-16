"""Static resolution, frozen payloads and independent Mock receipts agree."""
import copy
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.container.inference_configuration import (
    attach_configuration_context,
    bound_load_options,
    capture_generation_consumption,
    validate_inference_payload,
    verify_generation_consumption,
    verify_model_session_configuration,
)
from acprof.container.model_session import ModelLoadConfig, ModelSession
from acprof.failures import RuntimeFailure
from acprof.host.configuration_transport import verify_runtime_configuration
from acprof.host.input_plan.materialize import get_task_generator
from acprof.models.configuration_resolution import resolve_inference_configuration
from acprof.models.inference_configuration import (
    configuration_identity,
    decode_inference_configuration,
    encode_inference_configuration,
    freeze_inference_configuration,
    validate_frozen_configuration,
)
from acprof.tasks.groups import group_inference_tasks
from tests.support.task_configuration_fixture import routable_task


@pytest.mark.parametrize("preset", ["custom", "auto", "benchmark"])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("family", ["audio", "timeseries", "multimodal", "diffusion"])
def test_resolved_parameters_survive_freezing_payload_and_runtime_receipt(monkeypatch, family, grouped, preset):
    original = routable_task(family)
    if grouped:
        original.selected_tasks = group_inference_tasks(original.selected_tasks, backend=original.runtime_backend)
    task, candidate = resolve_inference_configuration(
        original, {"schema_version": 1, "preset": preset}, devices=["off"],
    )
    task.inference_configuration = freeze_inference_configuration(candidate, candidate["identity_sha256"])
    assert decode_inference_configuration(encode_inference_configuration(task.inference_configuration)) == task.inference_configuration
    selected = task.selected_tasks[0]
    values = selected["config"]["inference"]
    assert values
    assert candidate["rule_version"] == "acprof-inference-v2"
    config = ModelLoadConfig(
        task.model_id, family, task.pipeline_tag, task.runtime_backend,
        model_revision=task.model_revision, adapter=task.model_adapter,
        inference_configuration=task.inference_configuration, inference_task_id=selected["task_id"],
    )
    monkeypatch.setenv("ACPROF_RUNTIME_PROFILE", task.runtime_profile_id)
    assert bound_load_options(config, "cpu", None) == candidate["loading"]["off"]
    class EulerDiscreteScheduler:
        pass

    class StableDiffusionXLPipeline:
        scheduler = EulerDiscreteScheduler()

        def __call__(self, **kwargs):
            return kwargs

    model = SimpleNamespace(generate=Mock(), config=SimpleNamespace(model_type="whisper"))
    context = {"pipeline": StableDiffusionXLPipeline()} if family == "diffusion" else (
        {"pipeline": SimpleNamespace(predict=Mock())} if family == "timeseries" else {"model": model}
    )
    attach_configuration_context(config, "cpu", context)
    session = ModelSession(config, Mock(), context, SimpleNamespace(inference_context=nullcontext), "cpu")
    generator = get_task_generator(task, 1)
    payload = generator.generate({"audio": 1, "timeseries": 8, "multimodal": 224, "diffusion": 256}[family])
    validate_inference_payload(session, payload)
    with capture_generation_consumption(session) as receipt:
        if family == "timeseries":
            context["pipeline"].predict(object(), prediction_length=payload["prediction_length"])
        elif family == "diffusion":
            context["pipeline"](**payload["params"])
        else:
            kwargs = dict(payload["params"])
            if family == "audio":
                kwargs["task"] = kwargs.pop("asr_task")
            model.generate(**kwargs)
    report = {
        "status": "ok", "device": "cpu", "inference_configuration_sha256": candidate["identity_sha256"],
        "configuration_validation": verify_model_session_configuration(session),
        "consumer_receipt": verify_generation_consumption(session, receipt),
    }
    verify_runtime_configuration(task, report, expected_device="cpu", require_consumers=True)
    corrupt = copy.deepcopy(report)
    observation = next(iter(corrupt["consumer_receipt"]["calls"][0]["parameters"].values()))
    observation["source"] = "request_echo"
    with pytest.raises(RuntimeFailure, match="source"):
        verify_runtime_configuration(task, corrupt, expected_device="cpu", require_consumers=True)
    if family == "diffusion":
        context["pipeline"].scheduler = object()
        with pytest.raises(RuntimeFailure, match="scheduler"):
            verify_model_session_configuration(session)
    legacy = copy.deepcopy(task.inference_configuration)
    legacy["rule_version"] = "acprof-inference-v1"
    legacy["identity_sha256"] = configuration_identity(legacy)
    with pytest.raises(ValueError, match="rule v2"):
        validate_frozen_configuration(legacy)
