"""Reviewed parameters reach every planner generator before probes and materialization."""
import json
from dataclasses import replace

from acprof.application.run import inference_workload
from acprof.experiments.configuration import RunConfig
from acprof.host.input_plan.materialize import get_task_generator
from acprof.models.configuration_resolution import resolve_inference_configuration
from acprof.models.inference_configuration import freeze_inference_configuration
from tests.support.task_configuration_fixture import routable_task, task_model


def test_forecast_horizon_is_in_generated_payload_and_metadata():
    model = task_model("timeseries", "time-series-forecasting", "chronos", {"model_type": "t5"})
    model.selected_tasks[0]["config"]["inference"] = {"prediction_length": 24}
    generator = get_task_generator(model, 1)
    assert generator.generate(8)["prediction_length"] == 24
    assert generator.plan_metadata()["inference_parameters"]["prediction_length"] == 24


def test_whisper_language_is_in_payload_and_metadata_without_changing_audio_source():
    model = task_model("audio", "automatic-speech-recognition", "transformers_pipeline", {"model_type": "whisper"})
    model.selected_tasks[0]["config"]["inference"] = {"language": "zh", "asr_task": "transcribe"}
    generator = get_task_generator(model, 1)
    assert generator.generate(1)["params"]["language"] == "zh"
    assert generator.plan_metadata()["inference"]["language"] == "zh"


def test_multimodal_controls_reach_the_generated_request():
    model = task_model("multimodal", "image-text-to-text", "transformers_model", {"model_type": "qwen2_5_vl"})
    model.selected_tasks[0]["config"]["inference"] = {"max_new_tokens": 32, "do_sample": False}
    generator = get_task_generator(model, 1)
    assert generator.generate(224)["params"] == {"max_new_tokens": 32, "do_sample": False}


def test_diffusion_controls_reach_the_generated_request():
    model = task_model("diffusion", "text-to-image", "diffusers", {"_class_name": "StableDiffusionXLPipeline"})
    model.selected_tasks[0]["config"]["inference"] = {"num_inference_steps": 12, "guidance_scale": 4.5}
    generator = get_task_generator(model, 1)
    assert generator.generate(256)["params"]["num_inference_steps"] == 12
    assert generator.generate(256)["params"]["guidance_scale"] == 4.5


def test_custom_workload_parameters_remain_visible_and_reach_frozen_payload(tmp_path):
    task = routable_task("diffusion")
    path = tmp_path / "workload.json"
    path.write_text(json.dumps({"schema_version": 1, "params": {"num_inference_steps": 9, "guidance_scale": 3.5}}))
    options = replace(RunConfig.smoke(task.model_id), workload_spec=str(path)).to_namespace()
    workload = inference_workload(options, [1], [1], ["off"])
    task, candidate = resolve_inference_configuration(task, {"schema_version": 1}, devices=["off"], workload=workload)
    task.inference_configuration = freeze_inference_configuration(candidate, candidate["identity_sha256"])
    source = next(item["source"]["suggested_source"] for item in candidate["decisions"]
                  if item["path"].endswith(".num_inference_steps"))
    assert source == {"kind": "workload_specification", "sha256": workload["specification_sha256"],
                      "parameter": "num_inference_steps"}
    payload = get_task_generator(task, 1, workload_spec_path=str(path)).generate(256)
    assert payload["params"]["num_inference_steps"] == 9
    assert payload["params"]["guidance_scale"] == 3.5
