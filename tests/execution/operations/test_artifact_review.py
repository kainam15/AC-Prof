"""Inspect and interactive collection must agree on which gaps a user can repair."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from acprof.cli.main import main
from acprof.events.preparation import encode_reply, parse_event
from acprof.experiments.arguments import build_parser
from acprof.host.preparation.collection_workflow import PreparationInterrupted, PreparationWorkflow
from acprof.hub.repository import MetadataFile


def cached_snapshot(monkeypatch, model, revision, *, task, library, files, config=None):
    info = SimpleNamespace(sha=revision, pipeline_tag=task, library_name=library, config=config,
                           siblings=[SimpleNamespace(rfilename=name) for name in files])

    def model_info(repo_id, ref, *, source):
        assert (repo_id, ref, source) == (model, revision, "huggingface")
        return info

    def metadata(repo_id, name, ref, **kwargs):
        assert (repo_id, name, ref) == (model, "config.json", revision)
        assert config is not None
        return MetadataFile(Path(name), json.dumps(config).encode())

    monkeypatch.setenv("ACPROF_MODEL_SOURCE", "huggingface")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("acprof.host.model_store.store.cached_model_info", model_info)
    monkeypatch.setattr("acprof.host.model_store.store.cached_metadata", metadata)
    return build_parser().parse_args(["--model", model, "--revision", revision, "--gpus", "off"])


def requests(stdout):
    return [event["request"] for line in stdout.splitlines()
            if (event := parse_event(line)) and "request" in event]


def test_known_coreml_layout_has_one_reason_in_inspect_and_collection(monkeypatch, tmp_path, capsys):
    model = "argmaxinc/whisperkit-coreml"
    revision = "0f63a7800b00dd0226abd051b906c246e1907482"
    args = cached_snapshot(
        monkeypatch, model, revision, task="automatic-speech-recognition", library="whisperkit",
        files=["config.json", "openai_whisper-tiny/AudioEncoder.mlmodelc/model.mil",
               "openai_whisper-tiny/TextDecoder.mlmodelc/weights/weight.bin"],
        config={"name": "whisperkit-coreml", "version": "0.4", "device_support": []},
    )
    assert main(["inspect", model, "--revision", revision, "--explain", "--output-dir", str(tmp_path)]) == 2
    printed = capsys.readouterr().out
    report = json.loads((tmp_path / "model_resolution.json").read_text())

    monkeypatch.setattr("sys.stdin", io.StringIO(encode_reply(1, "cancel")))
    with pytest.raises(PreparationInterrupted):
        PreparationWorkflow(interactive=True).resolve(args)
    request, = requests(capsys.readouterr().out)

    assert report["status"] == "blocked"
    assert report["model_revision"] == revision
    failure = report["failure"]
    assert failure["reason_code"] == "artifact_layout_unsupported"
    assert "Reason: artifact_layout_unsupported" in printed
    diagnostics = json.dumps(report["configuration_options"]["diagnostics"], ensure_ascii=False)
    for misleading in ("请设置 --backend", "--model-spec", "推理接口信息不足或相互矛盾"):
        assert misleading not in printed
        assert misleading not in diagnostics
    assert request["kind"] == "error"
    assert request["failure"]["reason_code"] == failure["reason_code"]
    for key, value in {"artifact_format": "CoreML", "declared_task": "automatic-speech-recognition",
                       "library_name": "whisperkit"}.items():
        assert request["failure"]["evidence"][key] == failure["evidence"][key] == value
    assert not request.get("questions")


def test_mixed_repository_proceeds_with_its_native_checkpoint(monkeypatch, capsys):
    args = cached_snapshot(
        monkeypatch, "fixture/mixed-whisper", "a" * 40, task="automatic-speech-recognition",
        library="whisperkit", files=["config.json", "model.safetensors", "export/Encoder.mlmodelc/model.mil"],
        config={"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]},
    )
    resolved = PreparationWorkflow(interactive=True).resolve(args)
    assert resolved.runtime_backend == "transformers_pipeline"
    assert resolved.model_resolution["artifact_layout"]["weights"][0]["files"] == ["model.safetensors"]
    assert not requests(capsys.readouterr().out)


def test_missing_listing_can_be_reviewed_and_explicit_backend_stays_unverified(monkeypatch, capsys):
    args = cached_snapshot(
        monkeypatch, "fixture/unknown-layout", "a" * 40, task="automatic-speech-recognition",
        library="custom-library", files=[],
        config={"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]},
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(encode_reply(1, "cancel")))
    with pytest.raises(PreparationInterrupted):
        PreparationWorkflow(interactive=True).resolve(args)
    review, = requests(capsys.readouterr().out)
    assert review["kind"] == "review"
    assert "尚未确认兼容性" in review["summary"]
    args.backend = "transformers_pipeline"
    resolved = PreparationWorkflow(interactive=True).resolve(args)
    assert resolved.runtime_backend == args.backend
    assert resolved.model_resolution["artifact_layout"]["reason_code"] == "artifact_layout_unknown"


def test_local_model_spec_can_complete_interactive_collection_review(monkeypatch, tmp_path, capsys):
    args = cached_snapshot(monkeypatch, "fixture/undeclared-policy", "a" * 40, task=None,
                           library=None, files=["policy.pt"])
    spec = tmp_path / "model_spec.json"
    spec.write_text(json.dumps({"schema_version": 1, "task": "robotics", "format": "torchscript",
                                "model_file": "policy.pt", "feature_dim": 7}))
    replies = encode_reply(1, "answer", answers={"model_spec": str(spec)}) + encode_reply(2, "confirm")
    monkeypatch.setattr("sys.stdin", io.StringIO(replies))
    resolved = PreparationWorkflow(interactive=True).resolve(args)
    review, confirmation = requests(capsys.readouterr().out)
    assert review["kind"] == "review"
    assert [field["path"] for field in review["questions"]] == ["model_spec"]
    assert confirmation["resolved"] is True
    assert resolved.model_spec["model_file"] == "policy.pt"
    assert (resolved.pipeline_tag, resolved.runtime_backend) == ("robotics", "torchscript")


def test_interactive_checkpoint_choice_reaches_the_pinned_loader_and_download_plan(monkeypatch, capsys):
    from acprof.container.handlers import transformers_pipeline_load_kwargs
    from acprof.container.inference_configuration import bound_load_options
    from acprof.container.model_session import ModelLoadConfig
    from acprof.host.configuration_transport import inference_environment
    from acprof.models.spec import model_download_request

    revision = "a" * 40
    names = ["config.json", "model.safetensors", "model.fp16.safetensors"]
    args = cached_snapshot(monkeypatch, "fixture/whisper", revision, task="automatic-speech-recognition",
                           library="transformers", files=names,
                           config={"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]})
    monkeypatch.setattr("sys.stdin", io.StringIO(encode_reply(1, "answer", answers={"model_variant": "fp16"})))
    task = PreparationWorkflow(interactive=True).resolve(args)
    request, = requests(capsys.readouterr().out)
    assert request["questions"][0]["options"] == ["(default)", "fp16"]
    assert request["fields"]["Revision"] == revision
    assert args.model_variant == task.model_variant == model_download_request(task)["model_variant"] == "fp16"
    assert task.model_resolution["artifact_layout"]["weights"][0]["files"] == ["model.fp16.safetensors"]
    for key, value in inference_environment(task).items():
        monkeypatch.setenv(key, value)
    config = ModelLoadConfig.from_environment()
    assert transformers_pipeline_load_kwargs(bound_load_options(config, "cpu", None)) == {"model_kwargs": {"variant": "fp16"}}


def test_directory_choice_uses_its_own_pinned_architecture_and_assets(monkeypatch, tmp_path, capsys):
    from acprof.container.inference_configuration import bound_load_options
    from acprof.container.model_session import ModelLoadConfig
    from acprof.models.download_plan import plan_download
    from acprof.models.files import checkpoint_directories

    revision = "b" * 40
    folders = ("checkpoints/tiny/", "checkpoints/base/")
    files = {prefix + name: {"size": size} for prefix in folders for name, size in
             (("config.json", 10), ("model.safetensors", 98 if prefix.endswith("tiny/") else 180), ("tokenizer.json", 5))}
    args = cached_snapshot(monkeypatch, "fixture/variants", revision, task="text-generation", library="custom",
                           files=files, config={"model_type": "gpt2"})
    tiny = {"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}
    def metadata(repo, name, ref, **kwargs):
        assert (repo, name, ref) == (args.model, "checkpoints/tiny/config.json", revision)
        return MetadataFile(Path(name), json.dumps(tiny).encode())
    monkeypatch.setattr("acprof.host.model_store.store.cached_metadata", metadata)
    monkeypatch.setattr("sys.stdin", io.StringIO(encode_reply(1, "answer", answers={"model_variant": folders[0]})))
    task = PreparationWorkflow(interactive=True).resolve(args)
    request, = requests(capsys.readouterr().out)
    assert request["questions"][0]["options"] == sorted(folders)
    assert task.pipeline_tag == "automatic-speech-recognition"
    assert task.model_config["model_type"] == "whisper"
    assert task.metadata_sources["config.json"]["repository_path"] == "checkpoints/tiny/config.json"
    plan = plan_download(model_id=args.model, revision=revision, family=task.task_family, backend=task.runtime_backend,
                         files=files, read_json={"checkpoints/tiny/config.json": tiny}.__getitem__,
                         native_model_types={"whisper"}, model_variant=task.model_variant)
    assert {item["path"] for item in plan["files"]} == {name for name in files if name.startswith(folders[0])}
    assert plan["selected_bytes"] == 113
    assert plan["weights"][0]["files"] == ["checkpoints/tiny/model.safetensors"]
    (tmp_path / folders[0]).mkdir(parents=True)
    config = ModelLoadConfig(args.model, "audio", task.pipeline_tag, task.runtime_backend,
                             model_path=str(tmp_path), model_variant=plan["model_variant"])
    assert Path(config.resolve_source()) == tmp_path / folders[0]
    assert bound_load_options(config, "cpu", None) is None
    assert checkpoint_directories(set(files) | {"model_index.json"}) == []
    assert checkpoint_directories(set(files) | {"modules.json"}) == []
