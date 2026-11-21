"""Largest-scale probes use the effective selected workload through input planning."""
import json
import re
from pathlib import Path
from unittest.mock import Mock

import pytest
from huggingface_hub import constants
from huggingface_hub.errors import OfflineModeIsEnabled

from acprof.artifacts import read_input_scale_plan
from acprof.cli import probe
from acprof.host import largest_scale_probe
from acprof.host.docker_runtime import RunningContainer
from acprof.host.images.runtime import ImageInfo
from acprof.host.input_plan import families
from acprof.host.largest_scale_probe import PROBE_SUMMARY_NAME, read_probe_summary


@pytest.fixture
def t5_probe(tmp_path, monkeypatch):
    fixture = Path(__file__).parents[2] / "fixtures/t5_small_metadata"
    source = json.loads((fixture / "source.json").read_text())
    hub = tmp_path / "hub"
    snapshot = hub / "models--google-t5--t5-small/snapshots" / source["revision"]
    snapshot.mkdir(parents=True)
    for name in source["files"]:
        (snapshot / name).write_bytes((fixture / name).read_bytes())
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(hub))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", Mock(side_effect=OfflineModeIsEnabled("offline")))
    monkeypatch.setenv("ACPROF_MODEL_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("ACPROF_MODEL_SOURCE", "huggingface")
    for name in ("bootstrap_project_env", "require_collection_host", "require_native_docker", "require_cgroup_prerequisites"):
        monkeypatch.setattr(probe, name, Mock())
    image = Mock(return_value=ImageInfo(tag="fixture-probe"))
    monkeypatch.setattr(probe, "prepare_image", image)
    monkeypatch.setattr(families, "start_probe_session", Mock())
    monkeypatch.setattr(families, "stop_service_session", Mock())
    monkeypatch.setattr(families, "post_probe_payload", lambda _session, payload, _desc: {
        "payload": payload, "truncated_by_limit": False, "effective_input_scale": 32,
    })

    session = RunningContainer("fixture", "http://127.0.0.1:8002", 8002, 0.1)
    monkeypatch.setattr(largest_scale_probe, "start_container_session", Mock(return_value=session))
    monkeypatch.setattr(largest_scale_probe, "stop_container_session", Mock())
    response = Mock(status_code=200)
    response.json.return_value = {"effective_input_scale": 32}
    post = Mock(return_value=response)
    monkeypatch.setattr(largest_scale_probe.requests, "post", post)
    output = tmp_path / "result"
    args = ["--model", source["model_id"], "--revision", source["revision"], "--output-dir", str(output),
            "--gpus", "off", "--input-scales", "32"]
    return args, output, image, post


@pytest.mark.parametrize(("requested", "prefix", "ngram", "members"), [
    ("translation_en_to_fr", "translate English to French: ", 0, ["translation_en_to_fr"]),
    ("summarization", "summarize: ", 3, ["summarization"]),
    ("translation_en_to_fr,translation_en_to_de", "translate English to German: ", 0,
     ["translation_en_to_de", "translation_en_to_fr"]),
])
def test_pinned_profile_reaches_the_largest_request(t5_probe, tmp_path, requested, prefix, ngram, members):
    args, output, image, post = t5_probe
    spec = tmp_path / "workload.json"
    spec.write_text(json.dumps({"schema_version": 1, "params": {"max_new_tokens": 23}}))

    assert probe.main([*args, "--tasks", requested, "--workload-spec", str(spec)]) == 0

    image.assert_called_once()
    post.assert_called_once()
    assert post.call_args.args == ("http://127.0.0.1:8002/predict",)
    selected, = image.call_args.args[0].selected_tasks
    assert selected["execution_profile"]["semantic_task_ids"] == members
    summary, = output.rglob(PROBE_SUMMARY_NAME)
    plan = read_input_scale_plan(read_probe_summary(summary)["artifacts"]["input_scale_plan"])
    assert plan["task_workload"] == selected
    payload = post.call_args.kwargs["json"]
    assert plan["entries"][-1]["payload"] == payload
    assert payload["text"].startswith(prefix)
    assert payload["params"]["num_beams"] == 4
    assert payload["params"]["no_repeat_ngram_size"] == ngram
    assert payload["params"]["max_new_tokens"] == 23
    assert payload["task_workload"]["fingerprint"] == selected["fingerprint"]


@pytest.mark.parametrize("choice", [None, "all", "unknown-task", "translation_en_to_de,summarization"])
def test_unresolved_probe_choice_names_available_profiles_before_output(t5_probe, capsys, choice):
    args, output, image, post = t5_probe
    assert probe.main([*args, *(["--tasks", choice] if choice else [])]) == 2
    error = capsys.readouterr().err
    assert "one inference profile" in error
    assert "--tasks text2text-generation-" in error
    image.assert_not_called()
    post.assert_not_called()
    assert not output.exists()


def test_advertised_profile_id_can_be_used_verbatim(t5_probe, capsys):
    args, output, _image, post = t5_probe
    assert probe.main(args) == 2
    choices = re.findall(r"--tasks (text2text-generation-[a-f0-9]+)", capsys.readouterr().err)
    assert len(set(choices)) == 2

    assert probe.main([*args, "--tasks", choices[0]]) == 0

    post.assert_called_once()
    summary, = output.rglob(PROBE_SUMMARY_NAME)
    plan = read_input_scale_plan(read_probe_summary(summary)["artifacts"]["input_scale_plan"])
    assert plan["task_workload"]["task_id"] == choices[0]
    assert post.call_args.kwargs["json"] == plan["entries"][-1]["payload"]


@pytest.mark.parametrize("choice", [[], ["--tasks", "all"]])
def test_single_declared_workload_is_automatic(t5_probe, tmp_path, choice):
    args, _output, _image, post = t5_probe
    spec = tmp_path / "declaration.json"
    spec.write_text(json.dumps({"schema_version": 1, "task_declaration": {"task_id": "translation_en_to_ro"}}))
    assert probe.main([*args, "--workload-spec", str(spec), *choice]) == 0
    post.assert_called_once()
    assert post.call_args.kwargs["json"]["text"].startswith("translate English to Romanian: ")


@pytest.mark.parametrize(("specification", "reason"), [
    ({"task": "fill-mask"}, "conflicts with task interface"),
    ({"task_workload": {"task_id": "partial"}}, "invalid frozen task workload"),
])
def test_invalid_workload_stops_before_preparation(t5_probe, tmp_path, capsys, specification, reason):
    args, output, image, post = t5_probe
    spec = tmp_path / "invalid.json"
    spec.write_text(json.dumps({"schema_version": 1, **specification}))
    assert probe.main([*args, "--tasks", "translation_en_to_fr", "--workload-spec", str(spec)]) == 2
    assert reason in capsys.readouterr().err
    image.assert_not_called()
    post.assert_not_called()
    assert not output.exists()


def test_image_workload_keeps_relative_assets_with_automatic_selection(t5_probe, tmp_path):
    from PIL import Image

    _args, output, _image, post = t5_probe
    revision = "a" * 40
    snapshot = Path(constants.HF_HUB_CACHE) / "models--fixture--vision/snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type":"resnet","architectures":["ResNetForImageClassification"]}')
    spec_dir = tmp_path / "example"
    spec_dir.mkdir()
    asset = spec_dir / "sample.png"
    Image.new("RGB", (2, 2), "red").save(asset)
    spec = spec_dir / "vision.json"
    spec.write_text(json.dumps({"schema_version": 1, "image_path": asset.name}))
    post.return_value.json.return_value = {"effective_input_scale": 0.25}

    assert probe.main(["--model", "fixture/vision", "--revision", revision, "--gpus", "off",
                       "--input-scales", "0.25", "--workload-spec", str(spec), "--output-dir", str(output)]) == 0

    summary, = output.rglob(PROBE_SUMMARY_NAME)
    plan = read_input_scale_plan(read_probe_summary(summary)["artifacts"]["input_scale_plan"])
    assert plan["task_workload"]["interface"] == "image-classification"
    assert plan["workload"]["source_paths"] == [str(asset)]
    assert not list(output.rglob("task_workload.json"))
    assert post.call_args.kwargs["json"] == plan["entries"][0]["payload"]
