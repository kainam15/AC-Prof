"""Offline model metadata for task selection and inference profiles."""
import json
from pathlib import Path

from acprof.host.detect import TaskInfo
from acprof.models.candidates import apply_model_candidate, discover_model_candidates
from acprof.tasks.groups import group_inference_tasks


def t5_task():
    root = Path(__file__).parents[1] / "fixtures" / "t5_small_metadata"
    source = json.loads((root / "source.json").read_text())
    task = TaskInfo(source["model_id"], "text2text-generation", "nlp", "transformers_model",
                    "transformers", source["revision"], "hub_api",
                    model_config=json.loads((root / "config.json").read_text()),
                    repository_metadata={"generation_config.json": json.loads(
                        (root / "generation_config.json").read_text())},
                    hub_metadata={"pipeline_tag": "translation"})
    task.task_discovery = discover_model_candidates(task)["task_discovery"]
    return task


def t5_profiles(**kwargs):
    task = t5_task()
    return group_inference_tasks(task.task_discovery["tasks"], backend=task.runtime_backend, **kwargs)


def sentence_encoder_task(interface):
    config = {"architectures": ["MPNetForMaskedLM"], "model_type": "mpnet"}
    modules = [{"idx": 0, "path": "", "type": "sentence_transformers.models.Transformer"},
               {"idx": 1, "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"}]
    task = TaskInfo("fixture/sentence-encoder", interface, "nlp", "sentence_transformers",
                    "sentence-transformers", "a" * 40, "fixture", model_config=config,
                    repository_files=("config.json", "modules.json", "1_Pooling/config.json", "model.safetensors"),
                    repository_metadata={"config.json": config, "modules.json": modules,
                                         "1_Pooling/config.json": {"pooling_mode_mean_tokens": True}},
                    hub_metadata={"pipeline_tag": interface})
    task.model_resolution = apply_model_candidate(task, discover_model_candidates(task, override_tag=interface))
    return task
