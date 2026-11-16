"""No checkpoint, Torch install, network or synthesis is required by these fixtures."""
import importlib
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from acprof.container.handlers import HandlerRegistry
from tests.models.contracts.test_kokoro import CONFIG, FILES, REVISION


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", Mock(side_effect=AssertionError("network forbidden")))
    monkeypatch.setattr(socket.socket, "connect", Mock(side_effect=AssertionError("network forbidden")))


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    # Resolve the real registry first so missing registration is an assertion failure.
    handler = HandlerRegistry.get("audio", "kokoro", adapter="kokoro")
    module = importlib.import_module(type(handler).__module__)
    for name in FILES:
        target = tmp_path / name
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(CONFIG) if name == "config.json" else "fixture")
    model = SimpleNamespace(vocab=CONFIG["vocab"], context_length=512)
    model.to = Mock(return_value=model)
    model.eval = Mock(return_value=model)
    model_factory = Mock(return_value=model)
    voice = np.ones((510, 1, 256), dtype=np.float32)
    calls = []

    class Pipeline:
        def __init__(self, **kwargs):
            self.lang_code = kwargs["lang_code"]
            self.voices = {}
            self.g2p = Mock(side_effect=lambda text: (text, [SimpleNamespace(text=text, phonemes=text)]))

        def en_tokenize(self, tokens):
            yield tokens[0].text, tokens[0].phonemes, tokens

        def load_voice(self, path):
            assert str(tmp_path) in path
            return voice

        def generate_from_tokens(self, phonemes, **kwargs):
            assert kwargs["voice"] is voice
            calls.append((phonemes, kwargs["speed"]))
            yield SimpleNamespace(audio=np.ones(12, dtype=np.float32), phonemes=phonemes)
            calls.append("consumed")

    monkeypatch.setattr(module, "_check_g2p_dependencies", lambda: None)
    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KModel=model_factory, KPipeline=Pipeline))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="float32"))
    ctx = handler.load(str(tmp_path), "text-to-speech", "kokoro", "cpu", REVISION, adapter="kokoro")
    return handler, ctx, model_factory, calls, tmp_path


def test_four_phases_local_loading_complete_segments_and_audio_contract(runtime):
    handler, ctx, factory, calls, root = runtime
    assert factory.call_args.kwargs["model"] == str(root / "kokoro-v1_0.pth")
    assert factory.call_args.kwargs["config"] == str(root / "config.json")
    raw = {"text": "ab\na", "params": {"voice": "af_heart", "language": "a", "speed": 1.5}}
    processed = handler.preprocess(ctx, raw)
    assert calls == []
    assert processed["_effective_input_scale"] == 3
    assert processed["_workload"]["input"]["text"]["tokens"] is None
    output = handler.predict(ctx, processed)
    assert calls == [("ab", 1.5), "consumed", ("a", 1.5), "consumed"]
    response = handler.postprocess(ctx, output)
    assert response["audio_sample_rate"] == 24000
    assert response["audio_num_samples"] == 24
    assert response["audio_duration_s"] == 0.001
    validation = handler.validate_output(ctx, raw, processed, output, response)
    assert validation["protocol"]["status"] == "verified"
    assert validation["task"]["status"] == "verified"
    assert validation["workload_contract"]["input"]["text"]["tokens"] is None


@pytest.mark.parametrize(("raw", "match"), [
    ({"text": ""}, "non-empty"), ({"phonemes": "a" * 511}, "510"),
    ({"text": "a" * 511}, "510"), ({"phonemes": "?"}, "vocabulary"),
    ({"text": "a", "params": {"voice": "missing"}}, "voice"),
    ({"text": "a", "params": {"language": "z"}}, "language"),
    ({"text": "a", "params": {"speed": float("nan")}}, "speed"),
    ({"text": "a", "phonemes": "a"}, "exactly one"),
])
def test_invalid_inputs_fail_before_synthesis(runtime, raw, match):
    handler, ctx, _, calls, _ = runtime
    with pytest.raises(ValueError, match=match):
        handler.preprocess(ctx, raw)
    assert calls == []


def test_phonemes_skip_g2p_and_segment_errors_are_not_partial_success(runtime):
    handler, ctx, _, calls, _ = runtime
    processed = handler.preprocess(ctx, {"phonemes": ["ab", "a"]})
    ctx["pipelines"]["a"].g2p.assert_not_called()

    def failing(*args, **kwargs):
        yield SimpleNamespace(audio=np.ones(12, dtype=np.float32))
        raise RuntimeError("second segment failed")

    ctx["pipelines"]["a"].generate_from_tokens = failing
    with pytest.raises(ValueError, match="second segment failed"):
        handler.predict(ctx, processed)


@pytest.mark.parametrize("audio", [np.array([]), np.array([np.nan]), np.array([np.inf]),
                                    np.ones((2, 3)), np.array([1], dtype=np.int16)])
def test_independent_validator_rejects_invalid_audio(runtime, audio):
    handler, ctx, *_ = runtime
    processed = handler.preprocess(ctx, {"phonemes": "a"})
    output = handler.predict(ctx, processed)
    response = handler.postprocess(ctx, output)
    output["segments"][0].audio = audio
    with pytest.raises(ValueError, match="audio"):
        handler.validate_output(ctx, {}, processed, output, response)


@pytest.mark.parametrize("missing", FILES[:3])
def test_missing_snapshot_files_precede_model_loading(runtime, missing):
    handler, _, factory, _, root = runtime
    factory.reset_mock()
    (root / missing).unlink()
    with pytest.raises(ValueError, match="missing|incomplete"):
        handler.load(str(root), "text-to-speech", "kokoro", "cpu", REVISION, adapter="kokoro")
    factory.assert_not_called()


@pytest.mark.parametrize("speed", [1.5, 2])
def test_frozen_controls_are_observed_at_synthesis_and_mismatch_is_rejected(runtime, speed):
    from acprof.container.inference_configuration import (
        capture_generation_consumption,
        verify_generation_consumption,
    )
    handler, ctx, *_ = runtime
    params = {"language": "a", "voice": "af_heart", "speed": speed}
    ctx["_frozen_inference"] = params
    config = SimpleNamespace(task_family="audio", backend="kokoro", adapter="kokoro", task_type="text-to-speech",
                             inference_task_id="speech", inference_configuration={"identity_sha256": "fixture"})
    session = SimpleNamespace(config=config, handler=handler, context=ctx, device="cpu")
    prepared = handler.preprocess(ctx, {"phonemes": "ab", "params": params})
    with capture_generation_consumption(session) as receipt:
        handler.predict(ctx, prepared)
    assert verify_generation_consumption(session, receipt)["status"] == "verified"
    prepared["speed"] = 3.0
    with pytest.raises(ValueError, match="frozen"):
        with capture_generation_consumption(session):
            handler.predict(ctx, prepared)


def test_workload_keeps_word_construction_distinct_from_phoneme_scale():
    from acprof.workloads import get_generator
    generator = get_generator("audio", "hexgrad/Kokoro-82M", "text-to-speech", 1, model_adapter="kokoro",
                               inference_parameters={"voice": "bf_emma", "language": "b", "speed": 1.25})
    payload = generator.generate_for_word_count(6)
    assert len(payload["text"].split()) == 6
    assert payload["params"] == {"voice": "bf_emma", "language": "b", "speed": 1.25}
    assert generator.plan_metadata()["input_scale_type"] == "phoneme_ids"
    assert generator.scale_label(12) == "phon12"
    assert generator.effective_input_scale(6, payload) is None


def test_loading_accepts_model_store_blob_links(runtime):
    handler, _, _, _, root = runtime
    blob = root.parent / (root.name + "-blob")
    (root / "kokoro-v1_0.pth").rename(blob)
    (root / "kokoro-v1_0.pth").symlink_to(blob)
    assert handler.load(str(root), "text-to-speech", "kokoro", "cpu", REVISION, adapter="kokoro")["voices"]


def test_g2p_dependency_check_blocks_spacy_auto_download(monkeypatch):
    from acprof.container.handlers.kokoro import _check_g2p_dependencies
    download = Mock(side_effect=AssertionError("must not download"))
    monkeypatch.setitem(sys.modules, "spacy", SimpleNamespace(util=SimpleNamespace(is_package=lambda _: False), cli=SimpleNamespace(download=download)))
    monkeypatch.setitem(sys.modules, "misaki", SimpleNamespace(espeak=SimpleNamespace(EspeakFallback=Mock())))
    with pytest.raises(ValueError, match="en_core_web_sm"):
        _check_g2p_dependencies()
    download.assert_not_called()


def test_g2p_dependency_check_rejects_missing_espeak_library(monkeypatch):
    from acprof.container.handlers.kokoro import _check_g2p_dependencies
    monkeypatch.setitem(sys.modules, "spacy", SimpleNamespace(util=SimpleNamespace(is_package=lambda _: True)))
    monkeypatch.setitem(sys.modules, "misaki", SimpleNamespace(espeak=SimpleNamespace(EspeakFallback=Mock(side_effect=RuntimeError("library unavailable")))))
    with pytest.raises(ValueError, match="espeak-ng"):
        _check_g2p_dependencies()


@pytest.mark.parametrize(("device", "options"), [("cuda", {}), ("cpu", {"dtype": "FP16"}), ("cpu", {"trust_remote_code": True})])
def test_unsupported_device_and_loading_options_fail_before_loading(runtime, device, options):
    handler, _, factory, _, root = runtime
    factory.reset_mock()
    with pytest.raises(ValueError, match="FP32"):
        handler.load(str(root), "text-to-speech", "kokoro", device, REVISION, options, adapter="kokoro")
    factory.assert_not_called()


@pytest.mark.parametrize("rate", [0, 22050, 24000.0, True])
def test_validator_rejects_invalid_sample_rate(runtime, rate):
    handler, ctx, *_ = runtime
    prepared = handler.preprocess(ctx, {"phonemes": "a"})
    output = handler.predict(ctx, prepared)
    response = handler.postprocess(ctx, output)
    output["sample_rate"] = rate
    with pytest.raises(ValueError, match="sample rate"):
        handler.validate_output(ctx, {}, prepared, output, response)


@pytest.mark.parametrize("manual", [True, False])
def test_existing_scale_planners_use_observed_phonemes(runtime, monkeypatch, tmp_path, manual):
    from acprof.container.handlers import InputLimitError
    from acprof.host.images.runtime import ImageInfo
    from acprof.host.input_plan import families
    from tests.models.contracts.test_kokoro import task

    handler, ctx, _, calls, _ = runtime
    ctx["pipelines"]["a"].g2p.side_effect = lambda text: ("", [SimpleNamespace(text=text, phonemes=" ".join(["ab"] * len(text.split())))])
    monkeypatch.setattr(families, "start_probe_session", lambda *a: SimpleNamespace(base_url="fixture"))
    monkeypatch.setattr(families, "stop_service_session", Mock())
    monkeypatch.setattr(families, "request_nlp_scale_meta", lambda *a: handler.get_scale_metadata(ctx, {}))

    def probe(_session, payload, desc):
        try:
            prepared = handler.preprocess(ctx, payload)
            return {"effective_input_scale": prepared["_effective_input_scale"], "truncated_by_limit": False,
                    "reason": prepared["_probe_reason"], "payload": payload}
        except InputLimitError as exc:
            return {**exc.probe_metadata, "payload": payload}

    monkeypatch.setattr(families, "post_probe_payload", probe)
    model = task(model_adapter="kokoro", runtime_profile_id="kokoro-cpu")
    plan = (families.plan_manual_nlp_scales if manual else families.plan_nlp_auto_scales)(
        task_info=model, image_info=ImageInfo("fixture"), cpu_list=[1], mem_list=[2], gpu_list=["off"],
        batch_size=1, output_dir=str(tmp_path / "plan"), **({"scales": [1, 2]} if manual else {}))
    assert calls == []
    assert plan.workload["input_scale_type"] == "phoneme_ids"
    entries = json.loads(Path(plan.plan_file).read_text())["entries"]
    assert all(entry["input_scale"] == 3 * len(entry["payload"]["text"].split()) - 1 for entry in entries)
    assert all(entry["scale_label"].startswith("phon") for entry in entries)
    if manual:
        assert plan.scales == [2.0, 5.0]
    else:
        assert len(plan.scales) == 6


def test_multiple_valid_segments_can_exceed_the_auto_planning_cap(runtime):
    handler, ctx, _, calls, _ = runtime
    processed = handler.preprocess(ctx, {"phonemes": ["a" * 500, "b" * 500]})
    assert processed["_effective_input_scale"] == 1000
    assert processed["_workload"]["input"]["text"]["model_input_ids_including_boundaries"] == 1004
    response = handler.postprocess(ctx, handler.predict(ctx, processed))
    assert response["segment_count"] == 2
    assert calls.count("consumed") == 2


def test_missing_kokoro_dependency_is_classified_by_registry(runtime, monkeypatch):
    from acprof.container.handlers import HandlerDependencyMissingError, load_handler
    handler, _, _, _, root = runtime
    monkeypatch.setitem(sys.modules, "kokoro", None)
    with pytest.raises(HandlerDependencyMissingError, match="kokoro"):
        load_handler(handler, str(root), "text-to-speech", "kokoro", "cpu", REVISION, adapter="kokoro")


def test_independent_validation_consumes_host_frozen_configuration(runtime, monkeypatch):
    from contextlib import nullcontext

    from acprof.container.runtime_validate import validate
    from acprof.host.preparation.task_selection import select_tasks
    from acprof.models.candidates import apply_model_candidate, discover_model_candidates
    from acprof.models.configuration_resolution import (
        build_configuration_draft,
        resolve_inference_configuration,
    )
    from acprof.models.inference_configuration import (
        encode_inference_configuration,
        freeze_inference_configuration,
    )
    from tests.models.contracts.test_kokoro import task

    _, context, _, calls, root = runtime
    context["model"].parameters = lambda: [SimpleNamespace(dtype="torch.float32", is_floating_point=lambda: True)]
    model = task(model_adapter="kokoro", runtime_profile_id="kokoro-cpu")
    model.model_resolution = apply_model_candidate(model, discover_model_candidates(model))
    model = select_tasks(model, SimpleNamespace(task="", tasks="", backend="", model_spec="", workload_spec=""),
                         SimpleNamespace(interactive=False))
    draft = build_configuration_draft(model, devices=["off"], workload={"batch_size": 1})
    _, candidate = resolve_inference_configuration(model, draft["requests"], devices=["off"], workload={"batch_size": 1})
    frozen = freeze_inference_configuration(candidate, candidate["identity_sha256"])
    for key, value in {
        "MODEL_ID": model.model_id, "MODEL_REVISION": REVISION, "MODEL_LOCAL_PATH": str(root),
        "TASK_FAMILY": "audio", "TASK_TYPE": "text-to-speech", "RUNTIME_BACKEND": "kokoro", "USE_GPU": "0",
        "ACPROF_MODEL_ADAPTER": "kokoro", "ACPROF_INFERENCE_CONFIGURATION": encode_inference_configuration(frozen),
        "ACPROF_INFERENCE_TASK_ID": model.selected_tasks[0]["task_id"],
    }.items():
        monkeypatch.setenv(key, value)
    execution = SimpleNamespace(inference_context=nullcontext, metadata=lambda: {})
    monkeypatch.setattr("acprof.container.model_session.prepare_execution", lambda config: (execution, "cpu"))
    result = validate({"text": "ab\na", "input_scale": 3,
                       "params": {"language": "a", "voice": "af_heart", "speed": 1.0}})
    assert result["status"] == "ok"
    assert result["consumer_receipt"]["status"] == "verified"
    assert result["validation"]["protocol"]["status"] == "verified"
    assert result["validation"]["task"]["status"] == "verified"
    assert result["workload_contract"]["input"]["text"]["tokens"] is None
    assert result["effective_input_scale"] == 3
    assert result["response"]["audio_num_samples"] == 24
    assert calls.count("consumed") == 2
