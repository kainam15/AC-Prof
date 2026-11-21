"""Static artifact inspection rejects incompatible layouts without loading weights."""
from __future__ import annotations

import pytest

from acprof.failures import RuntimeFailure
from acprof.host.detect import TaskInfo
from acprof.models.resolution import resolve_model_interface


def _task(*, files=("config.json", "model.safetensors"), metadata=None, **overrides):
    config = {"model_type": "bert", "architectures": ["BertForMaskedLM"]}
    values = dict(
        model_id="fixture/layout", pipeline_tag="fill-mask", task_family="nlp",
        runtime_backend="transformers_pipeline", library_name="transformers",
        model_revision="a" * 40, detection_method="hub_api", model_config=config,
        repository_files=files, repository_metadata=metadata or {"config.json": config},
    )
    values.update(overrides)
    return TaskInfo(**values)


@pytest.mark.parametrize("foreign", [
    ("model.gguf",),
    ("encoder.mlpackage/Manifest.json", "encoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin"),
    ("encoder.mlmodel",),
    ("checkpoint.safetensors",),
])
def test_native_config_does_not_make_foreign_weights_transformers(foreign):
    task = _task(files=("config.json", *foreign))
    with pytest.raises(RuntimeFailure) as caught:
        resolve_model_interface(task)
    assert caught.value.failure.reason_code == "artifact_layout_unsupported"
    assert caught.value.failure.evidence["files"] == sorted(task.repository_files)


def test_mixed_native_and_gguf_keeps_native_checkpoint():
    result = resolve_model_interface(_task(files=("config.json", "model.safetensors", "model.gguf")))
    assert result["artifact_format"] == "transformers"
    assert result["artifact_layout"]["status"] == "identified"
    assert result["artifact_layout"]["weights"][0]["files"] == ["model.safetensors"]


def test_explicit_transformers_loader_ignores_unselected_sentence_transformer_chain():
    task = _task(
        files=("config.json", "model.safetensors", "modules.json", "encoder/config.json", "encoder/model.gguf"),
        metadata={
            "config.json": {"model_type": "bert", "architectures": ["BertForMaskedLM"]},
            "modules.json": [{"path": "encoder", "type": "sentence_transformers.models.Transformer"}],
            "encoder/config.json": {"model_type": "bert"},
        },
    )
    result = resolve_model_interface(task)
    assert result["backend"] == "transformers_pipeline"
    assert result["artifact_layout"]["status"] == "identified"
    assert result["artifact_layout"]["weights"][0]["files"] == ["model.safetensors"]


def test_transformers_root_loader_cannot_use_only_sentence_transformer_subdirectory():
    task = _task(
        files=("modules.json", "encoder/config.json", "encoder/model.safetensors"),
        metadata={
            "modules.json": [{"path": "encoder", "type": "sentence_transformers.models.Transformer"}],
            "encoder/config.json": {"model_type": "bert"},
        },
    )
    with pytest.raises(RuntimeFailure) as caught:
        resolve_model_interface(task)
    assert caught.value.failure.reason_code == "artifact_layout_unsupported"
    assert "loader root" in caught.value.failure.detail


def test_selected_checkpoint_index_with_missing_shard_is_blocked():
    task = _task(
        files=("config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors"),
        metadata={
            "config.json": {"model_type": "bert", "architectures": ["BertForMaskedLM"]},
            "model.safetensors.index.json": {"weight_map": {
                "first": "model-00001-of-00002.safetensors",
                "second": "model-00002-of-00002.safetensors",
            }},
        },
    )
    with pytest.raises(RuntimeFailure) as caught:
        resolve_model_interface(task)
    assert caught.value.failure.reason_code == "artifact_incomplete"
    assert "model-00002-of-00002.safetensors" in caught.value.failure.detail


def test_unknown_repository_listing_is_explicitly_unverified():
    result = resolve_model_interface(_task(files=()))
    assert result["artifact_layout"]["status"] == "unknown"
    assert result["artifact_layout"]["reason_code"] == "artifact_layout_unknown"
    assert result["artifact_format"] == "unknown"


def test_sentence_transformer_resolves_declared_component_checkpoint():
    task = _task(
        files=("modules.json", "0_Transformer/config.json", "0_Transformer/model.safetensors",
               "0_Transformer/model.gguf", "1_Pooling/config.json"),
        metadata={
            "modules.json": [
                {"idx": 0, "name": "0", "path": "0_Transformer", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            ],
            "0_Transformer/config.json": {"model_type": "bert"},
            "1_Pooling/config.json": {"pooling_mode_mean_tokens": True},
        },
        pipeline_tag="feature-extraction", runtime_backend="sentence_transformers",
        library_name="sentence-transformers",
    )
    result = resolve_model_interface(task)
    assert result["artifact_layout"]["weights"][0]["files"] == ["0_Transformer/model.safetensors"]


@pytest.mark.parametrize("modern", [False, True], ids=["legacy", "v5.6"])
@pytest.mark.parametrize("missing", [None, "projection/config.json", "projection/model.safetensors"])
def test_sentence_encoder_identifies_each_dense_checkpoint_or_reports_the_missing_file(modern, missing):
    from acprof.models.artifact_layout import checkpoint_metadata_indexes
    from acprof.tasks.interfaces import execution_path

    refs = (["sentence_transformers.base.modules.transformer.Transformer",
             "sentence_transformers.sentence_transformer.modules.pooling.Pooling",
             "sentence_transformers.base.modules.dense.Dense",
             "sentence_transformers.sentence_transformer.modules.normalize.Normalize"] if modern else
            [f"sentence_transformers.models.{name}" for name in ("Transformer", "Pooling", "Dense", "Normalize")])
    paths = ["backbone", "pool", "projection", "normalize"]
    metadata = {
        "modules.json": [{"idx": i, "name": str(i), "type": ref, "path": path}
                         for i, (ref, path) in enumerate(zip(refs, paths, strict=True))],
        "backbone/config.json": {"model_type": "bert"},
        "backbone/model.safetensors.index.json": {"weight_map": {"embeddings": "part-1.safetensors"}},
        "pool/config.json": {"word_embedding_dimension": 8},
        "projection/config.json": {"in_features": 8, "out_features": 4},
    }
    files = set(metadata) | {"backbone/part-1.safetensors", "projection/model.safetensors",
                             "projection/model.safetensors.index.json", "elsewhere/model.safetensors"}
    if missing:
        files.remove(missing)
        metadata.pop(missing, None)
    task = _task(files=tuple(files), metadata=metadata, runtime_backend="sentence_transformers",
                 pipeline_tag="feature-extraction", library_name="sentence-transformers")
    assert checkpoint_metadata_indexes(files, metadata) == ["backbone/model.safetensors.index.json"]
    assert [c["type"] for c in execution_path(task, "feature-extraction")["components"]] == refs
    if missing:
        with pytest.raises(RuntimeFailure) as caught:
            resolve_model_interface(task)
        assert caught.value.failure.reason_code == "artifact_incomplete"
        assert missing in caught.value.failure.detail
    else:
        layout = resolve_model_interface(task)["artifact_layout"]
        assert layout["status"] == "identified"
        assert [ck["files"] for ck in layout["weights"]] == [
            ["backbone/model.safetensors.index.json", "backbone/part-1.safetensors"],
            ["projection/model.safetensors"],
        ]


def test_single_file_diffusion_does_not_claim_pipeline_layout():
    task = _task(
        files=("config.json", "sd_xl_base_1.0.safetensors"), model_config={},
        pipeline_tag="text-to-image", task_family="diffusion", runtime_backend="diffusers",
        library_name="diffusers",
    )
    with pytest.raises(RuntimeFailure) as caught:
        resolve_model_interface(task)
    assert caught.value.failure.reason_code == "artifact_layout_unsupported"


def test_diffusers_declared_component_cannot_be_satisfied_by_unrelated_root_weights():
    task = _task(
        files=("model_index.json", "model.safetensors"), model_config={},
        metadata={"model_index.json": {"_class_name": "StableDiffusionPipeline", "unet": ["diffusers", "UNet2DConditionModel"]}},
        pipeline_tag="text-to-image", task_family="diffusion", runtime_backend="diffusers",
        library_name="diffusers",
    )
    with pytest.raises(RuntimeFailure) as caught:
        resolve_model_interface(task)
    assert caught.value.failure.reason_code == "artifact_incomplete"
    assert "unet" in caught.value.failure.detail


@pytest.mark.parametrize("has_config", [False, True], ids=["missing-config", "complete-optional-component"])
def test_diffusers_native_checkpoint_requires_component_config(has_config):
    files = ("model_index.json", "unet/diffusion_pytorch_model.safetensors")
    if has_config:
        files += ("unet/config.json",)
    task = _task(
        files=files, model_config={},
        metadata={"model_index.json": {
            "_class_name": "DDPMPipeline", "unet": ["diffusers", "UNet2DModel"],
            "safety_checker": [None, None],
        }},
        pipeline_tag="text-to-image", task_family="diffusion", runtime_backend="diffusers",
        library_name="diffusers",
    )
    if not has_config:
        with pytest.raises(RuntimeFailure) as caught:
            resolve_model_interface(task)
        assert caught.value.failure.reason_code == "artifact_incomplete"
        assert "unet/config.json" in caught.value.failure.detail
    else:
        result = resolve_model_interface(task)
        assert result["artifact_layout"]["status"] == "identified"
        assert result["artifact_layout"]["weights"][0]["files"] == ["unet/diffusion_pytorch_model.safetensors"]


@pytest.mark.parametrize("backend", ["sentence_transformers", "diffusers"])
@pytest.mark.parametrize("artifact", ["model.gguf", "model.mlmodel"])
def test_native_component_declaration_does_not_support_foreign_weights(backend, artifact):
    if backend == "sentence_transformers":
        metadata = {
            "modules.json": [{"path": "encoder", "type": "sentence_transformers.models.Transformer"}],
            "encoder/config.json": {"model_type": "bert"},
        }
        task = _task(
            files=("modules.json", "encoder/config.json", f"encoder/{artifact}"), metadata=metadata,
            pipeline_tag="feature-extraction", runtime_backend=backend, library_name="sentence-transformers",
        )
    else:
        task = _task(
            files=("model_index.json", "unet/config.json", f"unet/{artifact}"),
            metadata={"model_index.json": {
                "_class_name": "StableDiffusionPipeline", "unet": ["diffusers", "UNet2DConditionModel"],
            }},
            model_config={}, pipeline_tag="text-to-image", task_family="diffusion",
            runtime_backend=backend, library_name="diffusers",
        )
    with pytest.raises(RuntimeFailure) as caught:
        resolve_model_interface(task)
    assert caught.value.failure.reason_code == "artifact_layout_unsupported"
    assert "GGUF" in caught.value.failure.detail or "CoreML" in caught.value.failure.detail
