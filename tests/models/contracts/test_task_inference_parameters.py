"""Task-specific controls use registered consumers without importing model libraries."""
import copy

import pytest

from acprof.models.configuration import parameter_declaration
from acprof.tasks.groups import group_inference_tasks
from acprof.tasks.parameters import apply_parameter_review, parameter_review
from tests.support.task_configuration_fixture import task_model


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize(("family", "interface", "backend", "config", "values"), [
    ("audio", "automatic-speech-recognition", "transformers_pipeline",
     {"model_type": "whisper"}, {"language": "zh", "asr_task": "transcribe"}),
    ("timeseries", "time-series-forecasting", "chronos",
     {"model_type": "t5", "chronos_config": {}}, {"prediction_length": 24}),
    ("multimodal", "image-text-to-text", "transformers_model",
     {"model_type": "qwen2_5_vl"}, {"max_new_tokens": 32, "do_sample": False}),
    ("diffusion", "text-to-image", "diffusers",
     {"_class_name": "StableDiffusionXLPipeline"},
     {"num_inference_steps": 12, "guidance_scale": 4.5}),
])
def test_task_controls_are_validated_materialized_and_fingerprinted(family, interface, backend, config, values, grouped):
    model = task_model(family, interface, backend, config)
    if grouped:
        model.selected_tasks = group_inference_tasks(model.selected_tasks, backend=backend)
    before = copy.deepcopy(model.selected_tasks)
    review = parameter_review(model, model.selected_tasks)
    answers = {field["path"]: field["value"] for field in review["questions"] + review["parameter_advanced"]}
    assert {f"task_0.{key}" for key in values} <= answers.keys()
    answers.update({f"task_0.{key}": value for key, value in values.items()})
    selected = apply_parameter_review(model, model.selected_tasks, answers, materialize_defaults=True)[0]
    assert selected["config"]["inference"] == values
    assert selected["fingerprint"] != before[0]["fingerprint"]
    assert model.selected_tasks == before
    if grouped and family == "multimodal":
        assert selected["execution_profile"]["output_length_policy"] == "user_max_new_tokens_32"
        assert selected["semantic_tasks"] == before[0]["semantic_tasks"]


def test_whisper_controls_are_not_offered_to_ctc_or_custom_audio_handlers():
    model = task_model("audio", "automatic-speech-recognition", "transformers_pipeline", {"model_type": "wav2vec2"})
    assert not parameter_declaration(model, model.selected_tasks[0]).parameters


def test_sdxl_scheduler_is_a_loading_control_with_an_observation_method():
    model = task_model("diffusion", "text-to-image", "diffusers", {"_class_name": "StableDiffusionXLPipeline"})
    fields = {spec.name: spec for spec in parameter_declaration(model, model.selected_tasks[0]).parameters}
    assert fields["scheduler"].scope == "loading"
    assert fields["scheduler"].validation_method == "scheduler.class"
    assert "DDIMScheduler" in fields["scheduler"].choices


def test_multimodal_sampling_remains_rejected_by_the_existing_contract():
    model = task_model("multimodal", "image-text-to-text", "transformers_model", {"model_type": "qwen2_5_vl"})
    review = parameter_review(model, model.selected_tasks)
    answers = {field["path"]: field["value"] for field in review["questions"] + review["parameter_advanced"]}
    answers["task_0.do_sample"] = True
    with pytest.raises(ValueError, match="do_sample"):
        apply_parameter_review(model, model.selected_tasks, answers)
