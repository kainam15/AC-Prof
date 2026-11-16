"""Static Kokoro routing and artifact plans never import inference libraries."""
from dataclasses import replace

import pytest

from acprof.extensions import CATALOG, UnsupportedExtensionError, load_catalog
from acprof.host.detect import TaskInfo
from acprof.models.backend_candidates import describe_backend_candidates
from acprof.models.download_plan import plan_download

MODEL = "hexgrad/Kokoro-82M"
REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
CONFIG = {"istftnet": {"upsample_rates": [10, 6]}, "plbert": {"max_position_embeddings": 512},
          "vocab": {"a": 43, "b": 44, " ": 16}, "n_token": 178, "style_dim": 128,
          "hidden_dim": 512, "n_layer": 3, "n_mels": 80, "max_dur": 50,
          "dropout": 0.2, "text_encoder_kernel_size": 5}
FILES = ("config.json", "kokoro-v1_0.pth", "voices/af_heart.pt", "voices/bf_emma.pt")


def task(**kwargs):
    values = dict(model_id=MODEL, model_revision=REVISION, pipeline_tag="text-to-speech",
                  task_family="audio", library_name="kokoro", runtime_backend="kokoro",
                  model_config={}, repository_metadata={"config.json": CONFIG},
                  repository_files=FILES, detection_method="hub_api")
    return TaskInfo(**(values | kwargs))


def test_exact_identity_routes_without_transformers_and_infers_missing_task():
    for library in ("kokoro", "transformers", "unknown"):
        resolved = CATALOG.resolve(model_id=MODEL, library=library, config=CONFIG, files=FILES)
        assert (resolved.task, resolved.backend, resolved.declaration.adapter) == ("text-to-speech", "kokoro", "kokoro")


def test_registered_io_describes_both_text_and_native_phonemes():
    resolved = CATALOG.resolve(model_id=MODEL, config=CONFIG)
    schema = resolved.io_format["input"]["json_schema"]
    assert schema["required"] == []
    assert schema["oneOf"] == [{"required": ["text"]}, {"required": ["phonemes"]}]
    assert "phonemes" in schema["properties"]


def test_explicit_route_and_identity_conflicts_remain_errors():
    assert CATALOG.resolve("text-to-speech", backend="kokoro", model_id=MODEL).declaration.adapter == "kokoro"
    with pytest.raises(UnsupportedExtensionError):
        CATALOG.select_extension(task(runtime_backend="transformers_pipeline"))
    with pytest.raises(UnsupportedExtensionError):
        CATALOG.select_extension(task(model_id="someone/Kokoro-82M", model_adapter="kokoro"))
    catalog = load_catalog()
    rule = next(rule for rule in catalog.backend_rules if rule.backend == "kokoro")
    catalog.backend_rules.append(replace(rule, backend="transformers_pipeline"))
    with pytest.raises(UnsupportedExtensionError, match="ambiguous backend"):
        catalog.resolve("text-to-speech", model_id=MODEL, library="kokoro")


def test_offers_remain_static_and_reject_unregistered_architectures():
    offers = describe_backend_candidates(task())
    selected = [item for item in offers if item["selectable"]]
    assert [item["backend"] for item in selected] == ["kokoro"]
    assert selected[0]["runtime_validation"] == {"status": "not_run"}
    assert selected[0]["compatibility"]["checkpoint_verified"] is False
    for model_id in ("owner/my-kokoro", "hexgrad/Kokoro-82M-copy", "owner/tts"):
        offers = describe_backend_candidates(task(model_id=model_id, model_config={"model_type": "unknown_voice"}))
        assert not any(item["selectable"] for item in offers)


@pytest.mark.parametrize("missing", FILES[:3])
def test_missing_required_artifact_is_rejected_before_runtime(missing):
    offers = describe_backend_candidates(task(repository_files=tuple(f for f in FILES if f != missing)))
    assert not any(item["selectable"] for item in offers)


@pytest.mark.parametrize(("revision", "config"), [
    ("0" * 40, CONFIG),
    (REVISION, CONFIG | {"model_type": "unregistered_voice"}),
    (REVISION, CONFIG | {"n_token": 999}),
])
def test_exact_identity_does_not_override_an_unreviewed_snapshot(revision, config):
    offers = describe_backend_candidates(task(model_revision=revision, repository_metadata={"config.json": config}))
    assert not any(item["selectable"] for item in offers)


def test_download_plan_selects_native_weights_config_and_voices():
    files = {name: {"size": 10} for name in (*FILES, "README.md", "model.onnx", "other.pth")}
    plan = plan_download(model_id=MODEL, revision=REVISION, family="audio", backend="kokoro",
                         adapter="kokoro", files=files, read_json=lambda _: CONFIG)
    assert plan["effective_policy"] == "selected"
    assert {item["path"] for item in plan["files"]} == {*FILES, "README.md"}
    assert plan["weights"][0]["files"] == ["kokoro-v1_0.pth"]


def test_candidate_discovery_uses_exact_identity_when_hub_task_is_absent():
    from acprof.models.candidates import discover_model_candidates
    model = task(pipeline_tag="unknown", task_family="unknown", runtime_backend="unknown",
                 hub_metadata={"pipeline_tag": None}, detection_method="config_infer")
    decision = discover_model_candidates(model)
    assert decision["status"] == "candidate"
    assert [(candidate["task"], candidate["backend"]) for candidate in decision["candidates"]] == [("text-to-speech", "kokoro")]


def test_hub_detection_preserves_native_config_for_static_layout_checks(monkeypatch):
    from types import SimpleNamespace

    from acprof.host import detect
    from acprof.models.candidates import discover_model_candidates

    hub = SimpleNamespace(pipeline_tag="text-to-speech", library_name="kokoro", sha=REVISION,
                          config=CONFIG, card_data={}, tags=[])
    monkeypatch.setattr("acprof.host.model_store.store.cached_model_info", lambda *a, **kw: hub)
    monkeypatch.setattr(detect, "_repository_metadata", lambda *a: {
        "repository_files": FILES, "repository_metadata": {"config.json": CONFIG},
        "metadata_errors": (), "metadata_diagnostics": (), "metadata_sources": {},
    })
    detected = detect._detect_from_hub(MODEL)
    assert detected.runtime_backend == "kokoro"
    assert detected.model_config == {}  # Shared architecture metadata is intentionally filtered.
    assert discover_model_candidates(detected)["status"] == "candidate"
    assert [item["backend"] for item in describe_backend_candidates(detected) if item["selectable"]] == ["kokoro"]


def test_confirmation_freezes_kokoro_controls_without_claiming_runtime_success():
    from types import SimpleNamespace

    from acprof.host.preparation.task_selection import select_tasks
    from acprof.models.candidates import apply_model_candidate, discover_model_candidates
    from acprof.models.configuration_resolution import (
        build_configuration_draft,
        resolve_inference_configuration,
    )
    from acprof.models.inference_configuration import freeze_inference_configuration

    model = task(model_adapter="kokoro", runtime_profile_id="kokoro-cpu")
    model.model_resolution = apply_model_candidate(model, discover_model_candidates(model))
    model = select_tasks(model, SimpleNamespace(task="", tasks="", backend="", model_spec="", workload_spec=""),
                         SimpleNamespace(interactive=False))
    draft = build_configuration_draft(model, devices=["off"], workload={"batch_size": 1})
    updated, candidate = resolve_inference_configuration(model, draft["requests"], devices=["off"], workload={"batch_size": 1})
    frozen = freeze_inference_configuration(candidate, candidate["identity_sha256"])
    assert frozen["validation"] == {"static": "validated", "runtime": "not_run"}
    assert updated.selected_tasks[0]["config"]["inference"] == {"language": "a", "voice": "af_heart", "speed": 1.0}
