"""Pinned author metadata; no network, weights or inference in UI tests."""
import json
from pathlib import Path

from acprof.host.detect import TaskInfo
from acprof.models.candidates import discover_model_candidates
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
