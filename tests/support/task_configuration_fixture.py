"""Synthetic metadata only; these fixtures contain no checkpoint weights."""
from acprof.host.detect import TaskInfo
from acprof.tasks.workloads import parse_task


def task_model(family, interface, backend, config):
    task = TaskInfo(
        "fixture/task-controls", interface, family, backend,
        "diffusers" if family == "diffusion" else "transformers",
        "a" * 40, "fixture", model_config=config,
    )
    task.selected_tasks = [parse_task(interface, {}, encoder_decoder=False, source="fixture")]
    return task


def routable_task(family):
    interface, backend, config, profile = {
        "audio": ("automatic-speech-recognition", "transformers_pipeline",
                  {"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}, "audio-cpu"),
        "timeseries": ("time-series-forecasting", "chronos",
                       {"model_type": "t5", "architectures": ["T5ForConditionalGeneration"],
                        "chronos_config": {"prediction_length": 64}}, "timeseries-cpu"),
        "multimodal": ("image-text-to-text", "transformers_model",
                       {"model_type": "qwen2_5_vl", "architectures": ["Qwen2_5_VLForConditionalGeneration"]},
                       "multimodal-transformers4576-cpu"),
        "diffusion": ("text-to-image", "diffusers", {
            "_class_name": "StableDiffusionXLPipeline", "unet": ["diffusers", "UNet2DConditionModel"],
            "scheduler": ["diffusers", "EulerDiscreteScheduler"],
        }, "diffusion-cpu"),
    }[family]
    task = task_model(family, interface, backend, config)
    task.runtime_profile_id = profile
    if family == "diffusion":
        task.repository_files = ("model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
                                 "scheduler/scheduler_config.json")
        task.repository_metadata = {"model_index.json": config}
    else:
        task.repository_files = ("config.json", "model.safetensors")
        task.repository_metadata = {"config.json": config}
    return task
