"""The native single-user chat route needs a causal checkpoint and a tokenizer default."""
import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.failures import RuntimeFailure
from acprof.host.detect import TaskInfo, detect_task
from acprof.host.preparation.task_selection import select_tasks
from acprof.models.resolution import resolve_model_interface
from acprof.tasks.interfaces import execution_path, validate_execution_identity


def native_task(template=None, *, config=None, files=()):
    config = config or {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}
    task = TaskInfo("example/model", "conversational", "nlp", "transformers_pipeline", "transformers", "a" * 40, "hub")
    task.model_config = config
    task.repository_metadata = {"config.json": config, "tokenizer_config.json": {"chat_template": template}}
    task.repository_files = ("config.json", "model.safetensors", "tokenizer_config.json", *files)
    return task


@pytest.mark.parametrize("template", ["{{ messages }}", {"default": "{{ messages }}"}, [
    {"name": "default", "template": "{{ messages }}"}, {"name": "tool_use", "template": "tool"},
]])
def test_native_chat_preserves_acprof_task_and_records_actual_transformers_pipeline(template):
    task = native_task(template)
    result = resolve_model_interface(task)
    assert result["pipeline_task"] == "text-generation"
    assert result["task"] == "conversational"
    assert result["chat_template_source"] == "tokenizer_config.json"
    path = execution_path(task, task.pipeline_tag, backend=task.runtime_backend)
    assert path["input_protocol"] == "single_user_chat_v1"
    assert "input_protocol" not in execution_path(task, "text-generation", backend=task.runtime_backend)


@pytest.mark.parametrize(("template", "files", "config"), [
    (None, (), None), ("  ", (), None), ({"tool_use": "{{ tools }}"}, (), None),
    ({"default": 6}, (), None), ([{"name": "default"}], (), None),
    ("valid embedded", ("chat_templates/tool_use.jinja",), None),
    ("{{ messages }}", (), {"model_type": "bert", "architectures": ["BertForMaskedLM"]}),
    ("{{ messages }}", (), {"model_type": "gpt2"}),
])
def test_missing_template_or_causal_head_does_not_pass_native_chat(template, files, config):
    task = native_task(template, config=config, files=files)
    with pytest.raises(RuntimeFailure) as error:
        resolve_model_interface(task)
    assert error.value.failure.reason_code in {"model_contract_required", "metadata_invalid", "runtime_task_unsupported"}
    assert "chat" in error.value.failure.detail.lower()


@pytest.mark.parametrize("name", ["chat_template.jinja", "chat_templates/default.jinja"])
def test_pinned_external_default_overrides_embedded_and_binds_bytes(tmp_path, name):
    from acprof.hub.repository import MetadataFile

    task = native_task("embedded", files=(name,))
    task.hub_metadata = {"tags": ["conversational"]}
    raw = b"{{ messages[0].content }}"
    def hydrate(content):
        with patch("acprof.host.detect._detect_from_hub", return_value=copy.deepcopy(task)), patch(
                "acprof.host.detect._download_metadata", return_value=MetadataFile(tmp_path / name, content)) as download:
            detected = detect_task(task.model_id, override_tag="conversational", revision=task.model_revision)
        assert download.call_args.args[:3] == (task.model_id, name, task.model_revision)
        return detected
    selected = hydrate(raw)
    resolution = resolve_model_interface(selected)
    assert resolution["chat_template_source"] == name
    expected = {"file": name, "sha256": hashlib.sha256(raw).hexdigest()}
    path = execution_path(selected, "conversational", backend="transformers_pipeline")
    assert path["chat_template"] == expected
    changed = hydrate(raw + b"!")
    assert execution_path(changed, "conversational", backend="transformers_pipeline") != path
    assert execution_path(changed, "text-generation", backend="transformers_pipeline") == execution_path(
        selected, "text-generation", backend="transformers_pipeline")
    with pytest.raises(RuntimeFailure, match="not UTF-8"):
        resolve_model_interface(hydrate(b"\xff"))
    with pytest.raises(RuntimeFailure, match="chat template"):
        resolve_model_interface(hydrate(b" \n"))


def test_old_native_chat_execution_identity_is_not_reused():
    task = native_task("{{ messages }}")
    with patch("acprof.host.detect._detect_from_hub", return_value=task):
        task = detect_task(task.model_id, override_tag="conversational", revision=task.model_revision)
    selected = select_tasks(task, SimpleNamespace(tasks="conversational", task="", backend=""), single_profile=True)
    validate_execution_identity(selected)
    # Rebuild a self-consistent old profile under the old path so only execution identity changed.
    from acprof.tasks.groups import reidentify_inference_profile
    choice, = selected.selected_tasks
    choice["execution_path"].pop("input_protocol", None)
    selected.selected_tasks = [reidentify_inference_profile(choice)]
    with pytest.raises(ValueError, match="identity changed"):
        validate_execution_identity(selected)


def test_runtime_resolution_preserves_explicit_custom_name_and_other_backends(tmp_path):
    from acprof.models.spec import resolve_pipeline_task

    def resolve(backend):
        return resolve_pipeline_task(str(tmp_path), "conversational", backend=backend, adapter="family-default")
    assert resolve("transformers_pipeline") == ("text-generation", True)
    assert resolve("transformers_model") == ("conversational", False)
    (tmp_path / "acprof_model.json").write_text(json.dumps({"schema_version": 1, "format": "transformers-pipeline",
        "task": "conversational", "pipeline_task": "text-generation"}))
    (tmp_path / "config.json").write_text(json.dumps({"custom_pipelines": {"text-generation": {"impl": "custom.Chat"}}}))
    (tmp_path / "custom.py").write_text("class Chat: pass\n")
    assert resolve("transformers_pipeline") == ("text-generation", False)


@pytest.mark.parametrize("custom", [
    {"custom_pipelines": {"text-generation": {"impl": "custom.Chat"}}},
    {"custom_pipelines": {"conversational": {"impl": "custom.Chat"}}},
    {"auto_map": {"AutoModelForCausalLM": "custom.Model"}},
])
def test_undeclared_custom_route_cannot_receive_native_chat_evidence(tmp_path, custom):
    from acprof.models.spec import resolve_pipeline_task

    task = native_task("{{ messages }}", config={"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", **custom},
                       files=("custom.py",))
    if "conversational" in custom.get("custom_pipelines", {}):
        resolution = resolve_model_interface(task)
        assert resolution["interface_kind"] == "custom_pipeline"
        assert resolution["pipeline_task"] == "conversational"
        assert "chat_template_source" not in resolution
    else:
        with pytest.raises(RuntimeFailure, match="not registered"):
            resolve_model_interface(task)
    assert "input_protocol" not in execution_path(task, "conversational", backend="transformers_pipeline")
    (tmp_path / "config.json").write_text(json.dumps(task.model_config))
    assert resolve_pipeline_task(str(tmp_path), "conversational", backend="transformers_pipeline",
                                 adapter="family-default") == ("conversational", False)


def test_legacy_chat_without_a_recorded_protocol_requires_a_new_run():
    task = native_task("{{ messages }}")
    with pytest.raises(ValueError, match="identity changed"):
        validate_execution_identity(task, restoring=True)
    task.model_resolution = resolve_model_interface(task)
    validate_execution_identity(task, restoring=True)
    task.pipeline_tag = "text-generation"
    task.model_resolution = {}
    validate_execution_identity(task, restoring=True)


def test_pending_legacy_chat_collection_keeps_history_and_cannot_dispatch(tmp_path):
    from dataclasses import replace

    from acprof.application.task_runs import TaskCollection
    from acprof.artifacts import atomic_write_json
    from acprof.experiments.configuration import RunConfig
    from acprof.tasks.workloads import parse_task

    task = native_task("{{ messages }}")
    task.selected_tasks = [parse_task("conversational", {}, encoder_decoder=False, source="legacy")]
    config = replace(RunConfig.smoke(task.model_id), output_dir=str(tmp_path))
    collection = TaskCollection(config, task)
    atomic_write_json(collection.path, collection.manifest)
    before = collection.path.read_bytes()
    resume = replace(config, resume=True, task_run_id=collection.run_id)
    _, saved = TaskCollection.read_manifest(resume)
    assert saved["tasks"][0]["status"] == "pending"
    with pytest.raises(ValueError, match="native chat execution identity changed"):
        TaskCollection.restore(resume)
    assert collection.path.read_bytes() == before


def test_external_read_failure_is_scoped_to_chat_and_preserves_the_provider_diagnostic():
    from acprof.hub.repository import MetadataSizeError

    name = "chat_template.jinja"
    task = native_task("embedded", files=(name,))
    task.hub_metadata = {"tags": ["conversational"]}
    with patch("acprof.host.detect._detect_from_hub", return_value=task), patch(
            "acprof.host.detect._download_metadata", side_effect=MetadataSizeError(name, 200, size=500)):
        detected = detect_task(task.model_id, override_tag="conversational", revision=task.model_revision)
    with pytest.raises(RuntimeFailure) as error:
        resolve_model_interface(detected)
    assert error.value.failure.reason_code == "metadata_unavailable"
    assert error.value.failure.evidence["lookup_reason"] == "metadata_size_limit"
    detected.pipeline_tag = "text-generation"
    assert resolve_model_interface(detected)["pipeline_task"] == "text-generation"
