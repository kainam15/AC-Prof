"""Largest-scale probes use the effective selected workload through input planning."""
import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from huggingface_hub import constants
from huggingface_hub.errors import OfflineModeIsEnabled

from acprof.artifacts import read_input_scale_plan
from acprof.cli import probe
from acprof.host import largest_scale_probe
from acprof.host.docker_runtime import RunningContainer
from acprof.host.images.runtime import ImageInfo
from acprof.host.input_plan import families, probe as input_probe
from acprof.host.largest_scale_probe import PROBE_SUMMARY_NAME, read_probe_summary


@pytest.fixture
def gpu_inventory(monkeypatch):
    from acprof.host import gpu_device

    result = SimpleNamespace(returncode=0, stderr="", stdout="GPU-second, 2, 0000:02:00.0, Second GPU, 8192, N/A")
    query = Mock(return_value=result)
    monkeypatch.setattr(gpu_device, "run_command", query)
    monkeypatch.setattr("acprof.host.images.runtime._host_cuda_version", lambda: (12, 8))
    return query


@pytest.fixture
def t5_probe(tmp_path, monkeypatch, gpu_inventory):
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
    monkeypatch.setattr(input_probe, "start_service_session", Mock())
    monkeypatch.setattr(families, "stop_service_session", Mock())
    monkeypatch.setattr(families, "post_probe_payload", lambda _session, payload, _desc: {
        "payload": payload, "truncated_by_limit": False, "effective_input_scale": 32,
    })

    session = RunningContainer("fixture", "http://127.0.0.1:8002", 8002, 0.1)
    def start_session(**kwargs):
        from acprof.host.gpu_device import selected_gpu_device

        return replace(session, gpu_device=selected_gpu_device() if kwargs["gpu"] == "on" else {})

    monkeypatch.setattr(largest_scale_probe, "start_container_session", Mock(side_effect=start_session))
    monkeypatch.setattr(largest_scale_probe, "stop_container_session", Mock())
    response = Mock(status_code=200)
    response.json.return_value = {"effective_input_scale": 32}
    post = Mock(return_value=response)
    monkeypatch.setattr(largest_scale_probe.requests, "post", post)
    output = tmp_path / "result"
    args = ["--model", source["model_id"], "--revision", source["revision"], "--output-dir", str(output),
            "--input-scales", "32"]
    return args, output, image, post


@pytest.mark.parametrize(("gpu_args", "expected"), [
    ([], "off"), (["--gpus", "on,off"], "off"), (["--gpus", "on"], "on"),
])
def test_probe_uses_one_device_from_preparation_to_prediction(t5_probe, monkeypatch, gpu_args, expected):
    args, output, image, post = t5_probe
    image.return_value.runtime_environment = {"build_fingerprint": "fixture"}
    validate = Mock()
    monkeypatch.setattr("acprof.host.runtime_validation.validate_runtime", validate)

    assert probe.main([*args, "--tasks", "summarization", "--cpus", "1,4",
                       "--mems", "8,2", *gpu_args]) == 0

    summary_path, = output.rglob(PROBE_SUMMARY_NAME)
    summary = read_probe_summary(summary_path)
    planning = input_probe.start_service_session.call_args.kwargs
    predicting = largest_scale_probe.start_container_session.call_args.kwargs
    assert (planning["gpu"], validate.call_args.kwargs["gpu_list"], predicting["gpu"]) == (
        expected, [expected], expected,
    )
    assert image.call_args.args[0].model_resolution["requested_devices"] == [expected]
    assert (planning["cpu"], planning["mem"]) == (4, 8)
    assert (predicting["cpu"], predicting["mem"]) == (1, 2)
    assert summary["resource"]["gpu_mode"] == expected
    assert summary["memory_probe"]["candidate_order_gb"] == [2, 8]
    post.assert_called_once()


def test_gpu_only_probe_reports_device_failure_before_prediction(t5_probe, monkeypatch, capsys):
    args, output, image, post = t5_probe
    image.return_value.runtime_environment = {"build_fingerprint": "fixture"}
    validate = Mock(side_effect=RuntimeError("fixture GPU unavailable"))
    monkeypatch.setattr("acprof.host.runtime_validation.validate_runtime", validate)

    assert probe.main([*args, "--tasks", "summarization", "--gpus", "on"]) == 2

    assert "fixture GPU unavailable" in capsys.readouterr().err
    assert validate.call_args.kwargs["gpu_list"] == ["on"]
    post.assert_not_called()
    assert not list(output.rglob(PROBE_SUMMARY_NAME))


def test_kokoro_probe_rejects_gpu_before_build_and_can_retry_on_cpu(t5_probe, monkeypatch, capsys):
    from tests.models.contracts.test_kokoro import task

    model = task()
    _, output, image, post = t5_probe
    monkeypatch.setattr("acprof.host.detect.detect_task", lambda **kwargs: model)
    args = ["--model", model.model_id, "--revision", model.model_revision,
            "--input-scales", "32", "--output-dir", str(output)]
    assert probe.main([*args, "--gpus", "on"]) == 2

    error = capsys.readouterr().err
    failure = json.loads(error.split("ACPROF_FAILURE=", 1)[1])
    assert failure["reason_code"] == "runtime_unavailable"
    assert failure["runtime_profile"] == "kokoro-cpu"
    assert "--gpus off" in failure["detail"]
    image.assert_not_called()
    post.assert_not_called()
    assert not output.exists()

    assert probe.main([*args, "--gpus", "off,on"]) == 0
    summary_path, = output.rglob(PROBE_SUMMARY_NAME)
    assert read_probe_summary(summary_path)["resource"]["gpu_mode"] == "off"
    post.assert_called_once()


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


@pytest.fixture
def reviewable_t5_probe(t5_probe, monkeypatch):
    snapshot, = Path(constants.HF_HUB_CACHE).glob("models--google-t5--t5-small/snapshots/*")
    inventory = SimpleNamespace(sha=snapshot.name, pipeline_tag="translation", library_name="transformers",
                                siblings=[SimpleNamespace(rfilename=name) for name in (
                                    "config.json", "generation_config.json", "model.safetensors",
                                )])
    monkeypatch.setattr("acprof.host.model_store.store.cached_model_info", Mock(return_value=inventory))
    return t5_probe


def test_probe_reviews_changed_requests_and_executes_the_confirmed_payload(reviewable_t5_probe, tmp_path, monkeypatch, capsys):
    args, output, image, post = reviewable_t5_probe
    validate = Mock()
    monkeypatch.setattr("acprof.host.runtime_validation.validate_runtime", validate)
    request_path = tmp_path / "requests.json"
    request_path.write_text('{"schema_version": 1}')
    command = [*args, "--tasks", "summarization", "--input-scales", "64", "--timeout-seconds", "7",
               "--cpus", "1,2,4", "--mems", "8,2", "--gpus", "off,on", "--inference-config", str(request_path)]
    reviews = output / ".inference-configurations"

    def review(*extra):
        assert probe.main([*command, *extra]) == 0
        output_text = capsys.readouterr().out
        digest = re.search(r"Configuration SHA-256: ([a-f0-9]{64})", output_text)[1]
        assert f"--confirm-inference {digest}" in output_text
        image.assert_not_called()
        validate.assert_not_called()
        post.assert_not_called()
        assert not list(output.rglob("probes"))
        return digest, json.loads((reviews / f"{digest}.requests.json").read_text())

    initial, requests = review()
    task_fields, = requests["tasks"].values()
    for name, value in (("num_beams", 2), ("batch_size", 2), ("input_scales", "32")):
        task_fields[name] = {"mode": "explicit", "value": value}
    request_path.write_text(json.dumps(requests))
    confirmed, _ = review("--confirm-inference", initial)
    assert confirmed != initial
    changed_timeout, _ = review("--timeout-seconds", "9", "--confirm-inference", confirmed)
    assert changed_timeout != confirmed

    assert probe.main([*command, "--confirm-inference", confirmed]) == 0
    path, = output.rglob(PROBE_SUMMARY_NAME)
    summary = read_probe_summary(path)
    frozen = json.loads((path.parent / "inference_configuration.json").read_text())
    assert summary["inference_configuration_sha256"] == frozen["identity_sha256"] == confirmed
    assert frozen["state"] == "frozen"
    assert frozen["workload"]["resources"] == {"runtime": "docker", "cpus": [1, 4], "mems": [2, 8], "gpus": ["off"]}
    assert frozen == image.call_args.args[0].inference_configuration
    assert validate.call_args.kwargs["task_info"].inference_configuration == frozen
    plan = read_input_scale_plan(summary["artifacts"]["input_scale_plan"])
    assert plan["inference_configuration_sha256"] == confirmed
    assert (summary["input"]["batch_size"], summary["input"]["planned_scale"]) == (2, 32)
    payload = post.call_args.kwargs["json"]
    assert payload == plan["entries"][-1]["payload"]
    assert payload["batch_size"] == 2
    assert payload["params"]["num_beams"] == 2
    frozen["workload"]["probe"]["timeout_seconds"] = 9
    (path.parent / "inference_configuration.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="does not match its contents"):
        read_probe_summary(path)


def test_invalid_probe_inference_request_stops_before_image_or_results(t5_probe, tmp_path, capsys):
    args, output, image, post = t5_probe
    path = tmp_path / "invalid.json"
    path.write_text('{"schema_version":')
    assert probe.main([*args, "--tasks", "summarization", "--inference-config", str(path)]) == 2
    assert "inference_config" in capsys.readouterr().err
    image.assert_not_called()
    post.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("change", ["payload", "configuration", "missing_plan_sha256"])
def test_reviewed_probe_rechecks_the_actual_plan_before_cold_start(reviewable_t5_probe, tmp_path, monkeypatch, capsys, change):
    import hashlib

    args, output, _image, post = reviewable_t5_probe
    request = tmp_path / "requests.json"
    request.write_text('{"schema_version":1}')
    command = [*args, "--tasks", "summarization", "--inference-config", str(request)]
    assert probe.main(command) == 0
    normalized, = output.glob(".inference-configurations/*.requests.json")
    digest = normalized.name.split(".")[0]

    def change_plan(*, planned, **kwargs):
        path = Path(planned.plan_file)
        if change == "missing_plan_sha256":
            planned.plan_sha256 = ""
        else:
            plan = read_input_scale_plan(path)
            if change == "configuration":
                plan.pop("inference_configuration_sha256")
            else:
                plan["entries"][-1]["payload"]["text"] = "different input"
            path.write_text(json.dumps(plan))
            if change == "configuration":
                planned.plan_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr("acprof.host.runtime_validation.validate_runtime", change_plan)
    assert probe.main([*command, "--confirm-inference", digest]) == 2
    error = capsys.readouterr().err
    assert {"payload": "hash does not match", "configuration": "inference_configuration_mismatch",
            "missing_plan_sha256": "requires the SHA-256"}[change] in error
    largest_scale_probe.start_container_session.assert_not_called()
    post.assert_not_called()
    assert not list(output.rglob(PROBE_SUMMARY_NAME))


@pytest.mark.parametrize("edit", ["none", "before_planning", "during_planning"])
def test_reviewed_image_probe_keeps_relative_assets_and_checks_external_spec(t5_probe, tmp_path, monkeypatch, capsys, edit):
    from PIL import Image

    _, output, image, post = t5_probe
    revision = "b" * 40
    snapshot = Path(constants.HF_HUB_CACHE) / "models--fixture--vision/snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type":"resnet","architectures":["ResNetForImageClassification"]}')
    inventory = SimpleNamespace(sha=revision, pipeline_tag="image-classification", library_name="transformers",
                                siblings=[SimpleNamespace(rfilename=name) for name in ("config.json", "model.safetensors")])
    monkeypatch.setattr("acprof.host.model_store.store.cached_model_info", Mock(return_value=inventory))
    validate = Mock()
    monkeypatch.setattr("acprof.host.runtime_validation.validate_runtime", validate)
    spec_dir = tmp_path / "inputs"
    spec_dir.mkdir()
    asset = spec_dir / "example.png"
    Image.new("RGB", (2, 2), "red").save(asset)
    spec = spec_dir / "vision.json"
    document = {"schema_version": 1, "image_path": asset.name}
    spec.write_text(json.dumps(document))
    request = tmp_path / "requests.json"
    request.write_text('{"schema_version":1}')
    command = ["--model", "fixture/vision", "--revision", revision, "--gpus", "off", "--input-scales", "0.25",
               "--workload-spec", str(spec), "--inference-config", str(request), "--output-dir", str(output)]
    assert probe.main(command) == 0
    normalized, = output.glob(".inference-configurations/*.requests.json")
    digest = normalized.name.split(".")[0]
    original_plan = probe.plan_input_scales

    if edit == "before_planning":
        def edit_source(*args, **kwargs):
            spec.write_text(json.dumps({**document, "workload_id": "different"}))
            return image.return_value

        image.side_effect = edit_source
    elif edit == "during_planning":
        def temporary_edit(*args, **kwargs):
            spec.write_text(json.dumps({**document, "workload_id": "during-read"}))
            try:
                return original_plan(*args, **kwargs)
            finally:
                spec.write_text(json.dumps(document))

        monkeypatch.setattr(probe, "plan_input_scales", temporary_edit)
    post.return_value.json.return_value = {"effective_input_scale": 0.25}
    result = probe.main([*command, "--confirm-inference", digest])
    if edit == "none":
        assert result == 0
        path, = output.rglob(PROBE_SUMMARY_NAME)
        summary = read_probe_summary(path)
        plan = read_input_scale_plan(summary["artifacts"]["input_scale_plan"])
        assert plan["workload"]["source_paths"] == [str(asset)]
        assert post.call_args.kwargs["json"] == plan["entries"][0]["payload"]
        assert summary["inference_configuration_sha256"] == digest
    else:
        assert result == 2
        assert "workload specification changed" in capsys.readouterr().err
        validate.assert_not_called()
        largest_scale_probe.start_container_session.assert_not_called()
        post.assert_not_called()
        assert not list(output.rglob(PROBE_SUMMARY_NAME))


@pytest.mark.parametrize("selector", ["2", "GPU-second"])
def test_gpu_probe_pins_the_selected_physical_card_in_every_stage(t5_probe, gpu_inventory, monkeypatch, selector):
    from acprof.host import gpu_device

    args, output, image, post = t5_probe
    image.return_value.runtime_environment = {"build_fingerprint": "fixture"}
    monkeypatch.setenv("ACPROF_GPU_DEVICE", "0")
    observed = {}

    def observe(stage, mock):
        original = mock.side_effect
        def call(*args, **kwargs):
            observed[stage] = gpu_device.gpu_docker_args()[1]
            return original(*args, **kwargs) if original else mock.return_value
        mock.side_effect = call

    validate = Mock()
    monkeypatch.setattr("acprof.host.runtime_validation.validate_runtime", validate)
    for stage, mock in (("image", image), ("plan", input_probe.start_service_session),
                        ("validate", validate), ("predict", largest_scale_probe.start_container_session)):
        observe(stage, mock)
    assert probe.main([*args, "--tasks", "summarization", "--gpus", "on", "--gpu-device", selector]) == 0
    gpu_inventory.assert_called_once()
    assert f"--id={selector}" in gpu_inventory.call_args.args[0]
    assert observed == {stage: "device=GPU-second" for stage in ("image", "plan", "validate", "predict")}
    assert gpu_device.selected_gpu_device() == {}
    summary, = output.rglob(PROBE_SUMMARY_NAME)
    device = read_probe_summary(summary)["resource"]["gpu_device"]
    assert (device["uuid"], device["index"]) == ("GPU-second", 2)
    post.assert_called_once()


def test_cpu_probe_skips_gpu_selector_and_gpu_failure_stops_before_build(t5_probe, gpu_inventory, capsys):
    args, output, image, post = t5_probe
    gpu_inventory.return_value = SimpleNamespace(returncode=1, stderr="GPU unavailable")
    command = [*args, "--tasks", "summarization", "--gpu-device", "2"]
    assert probe.main([*command, "--gpus", "on"]) == 2
    assert "Cannot resolve GPU 2" in capsys.readouterr().err
    gpu_inventory.assert_called_once()
    image.assert_not_called()
    post.assert_not_called()
    assert not output.exists()
    gpu_inventory.reset_mock()
    assert probe.main([*command, "--gpus", "on,off"]) == 0
    gpu_inventory.assert_not_called()
    path, = output.rglob(PROBE_SUMMARY_NAME)
    assert "gpu_device" not in read_probe_summary(path)["resource"]


def test_reviewed_gpu_probe_requires_new_confirmation_if_index_maps_to_another_card(
    reviewable_t5_probe, gpu_inventory, tmp_path, monkeypatch,
):
    args, output, image, post = reviewable_t5_probe
    request = tmp_path / "requests.json"
    request.write_text('{"schema_version":1}')
    command = [*args, "--tasks", "summarization", "--gpus", "on", "--gpu-device", "2", "--inference-config", str(request)]
    assert probe.main(command) == 0
    review_root = output / ".inference-configurations"
    previous, = review_root.glob("*.requests.json")
    old_digest = previous.name.split(".")[0]
    gpu_inventory.return_value.stdout = "GPU-replacement, 2, 0000:05:00.0, Replacement GPU, 8192, N/A"
    assert probe.main([*command, "--confirm-inference", old_digest]) == 0
    image.assert_not_called()
    post.assert_not_called()
    updated, = set(review_root.glob("*.requests.json")) - {previous}
    new_digest = updated.name.split(".")[0]
    new = json.loads((review_root / f"{new_digest}.json").read_text())
    assert new["workload"]["resources"]["gpu_device_uuid"] == "GPU-replacement"
    monkeypatch.setattr("acprof.host.runtime_validation.validate_runtime", Mock())
    assert probe.main([*command, "--confirm-inference", new_digest]) == 0
    path, = output.rglob(PROBE_SUMMARY_NAME)
    summary = read_probe_summary(path)
    assert summary["resource"]["gpu_device"]["uuid"] == "GPU-replacement"
    assert summary["inference_configuration_sha256"] == new_digest
    summary["resource"]["gpu_device"]["uuid"] = "GPU-second"
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="GPU UUID mismatch"):
        read_probe_summary(path)


def test_gpu_probe_rejects_a_session_for_a_different_card_before_predict(t5_probe, monkeypatch, capsys):
    args, output, _image, post = t5_probe
    start = largest_scale_probe.start_container_session
    start.side_effect = None
    start.return_value = RunningContainer("fixture", "http://127.0.0.1:8002", 8002, 0.1,
                                          gpu_device={"uuid": "GPU-other", "index": 5})
    inspect = Mock(return_value={"OOMKilled": True})
    monkeypatch.setattr(largest_scale_probe.container_state, "inspect_container_state", inspect)
    assert probe.main([*args, "--tasks", "summarization", "--gpus", "on", "--gpu-device", "2", "--mems", "2,8"]) == 2
    assert "GPU UUID mismatch" in capsys.readouterr().err
    post.assert_not_called()
    start.assert_called_once()
    largest_scale_probe.stop_container_session.assert_called_once()
    inspect.assert_not_called()
    path, = output.rglob(PROBE_SUMMARY_NAME)
    assert json.loads(path.read_text())["status"] == "error"
