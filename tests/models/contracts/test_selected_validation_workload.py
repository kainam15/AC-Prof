"""Isolated runtime validation uses the same selected NLP request as a task collection."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.application.task_runs import TaskCollection
from acprof.artifacts import atomic_write_json
from acprof.experiments.configuration import RunConfig
from acprof.host.input_plan.materialize import get_task_generator
from acprof.host.preparation.model_inspection import validate_model_runtime
from acprof.host.preparation.task_inspection import inspect_selected_tasks
from acprof.host.preparation.task_selection import TaskSelectionRequired, activate_task
from acprof.host.runtime_validation import validation_input
from acprof.models.evidence import content_digest
from acprof.models.inference_configuration import configuration_identity
from acprof.tasks.workloads import parse_task
from tests.support.inference_configuration_fixture import frozen_task
from tests.support.task_selection_fixtures import t5_task


@pytest.fixture
def runtime_boundary():
    def read_request(*, task_info, planned, **kwargs):
        return validation_input(task_info, planned)[0]

    with patch("acprof.host.preflight.require_collection_host") as host, patch(
        "acprof.host.preflight.require_native_docker"
    ), patch("acprof.host.images.runtime.prepare_image") as image, patch(
        "acprof.host.runtime_validation.validate_runtime", side_effect=read_request
    ):
        yield host, image


def assert_collection_request(task, root, source=None):
    config = replace(RunConfig.smoke(task.model_id), output_dir=str(root / "collection"),
                     workload_spec=str(source) if source else "")
    child = TaskCollection(config, task).child_config(task.selected_tasks[0])
    formal = get_task_generator(task, 1, workload_spec_path=child.workload_spec)
    kwargs = {"workload_spec_path": source} if source else {}
    entry = validate_model_runtime(task, root / "probe", **kwargs)
    plan = json.loads((root / "probe/runtime_validation_input.json").read_bytes())
    assert entry["payload"] == formal.generate(entry["input_scale"])
    assert plan["scope"] == "runtime_validation_only"
    assert plan["task_workload"] == task.selected_tasks[0]
    assert plan["workload"]["workload_spec_sha256"] == hashlib.sha256(Path(child.workload_spec).read_bytes()).hexdigest()
    return entry["payload"], plan


@pytest.mark.parametrize(("profile", "prefix", "ngram"), [
    ("translation_en_to_fr", "translate English to French: ", 0),
    ("summarization", "summarize: ", 3),
])
def test_t5_selected_request_matches_collection(tmp_path, runtime_boundary, profile, prefix, ngram):
    task, error = inspect_selected_tasks(t5_task(), SimpleNamespace(tasks=profile, backend="", task=""))
    assert error is None
    payload, _ = assert_collection_request(task, tmp_path)
    assert payload["text"].startswith(prefix)
    assert payload["params"]["num_beams"] == 4
    assert payload["params"]["no_repeat_ngram_size"] == ngram
    assert payload["params"]["max_new_tokens"] == 64
    assert payload["task_workload"]["fingerprint"] == task.selected_tasks[0]["fingerprint"]


def test_conversational_selection_matches_collection(tmp_path, runtime_boundary):
    task = frozen_task()
    task.inference_configuration = {}
    selected = parse_task("conversational", {"input_text": "A chat message", "max_new_tokens": 5},
                          encoder_decoder=False, source="fixture")
    payload, _ = assert_collection_request(activate_task(task, selected, backend="transformers_model"), tmp_path)
    assert payload["text"].startswith("A chat message")
    assert payload["params"]["max_new_tokens"] == 5
    assert payload["task_workload"]["interface"] == "conversational"


def test_frozen_request_matches_original_spec_and_binds_effective_plan(tmp_path, runtime_boundary):
    task = frozen_task()
    spec = {"schema_version": 1, "params": {"max_new_tokens": 7}, "input_budget": {"max_text_characters": 12000}}
    frozen = task.inference_configuration
    frozen["workload"]["specification_sha256"] = content_digest(spec)
    frozen["identity_sha256"] = configuration_identity(frozen)
    path = tmp_path / "original.json"
    atomic_write_json(path, spec)
    payload, plan = assert_collection_request(task, tmp_path, source=path)
    assert payload["input_budget"]["max_text_characters"] == 12000
    assert plan["inference_configuration_sha256"] == frozen["identity_sha256"]
    assert plan["workload"]["workload_spec_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
    runtime_boundary[1].reset_mock()
    atomic_write_json(path, {**spec, "params": {}})
    with pytest.raises(ValueError, match="workload specification changed"):
        validate_model_runtime(task, tmp_path / "changed", workload_spec_path=path)
    with pytest.raises(ValueError, match="workload specification"):
        validate_model_runtime(task, tmp_path / "absent")
    runtime_boundary[1].assert_not_called()


def test_multiple_selected_profiles_fail_before_preparation(tmp_path, runtime_boundary):
    task = t5_task()
    task.selected_tasks = task.task_discovery["tasks"][:2]
    with pytest.raises(TaskSelectionRequired, match="choose one inference profile"):
        validate_model_runtime(task, tmp_path / "probe")
    assert not (tmp_path / "probe").exists()
    for prepare in runtime_boundary:
        prepare.assert_not_called()
