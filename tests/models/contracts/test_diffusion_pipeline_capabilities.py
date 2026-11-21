"""Task recognition never substitutes for reviewed artifact selection."""
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.host.detect import TaskInfo, detect_task
from acprof.models.artifact_layout import inspect_artifact_layout
from acprof.models.download_plan import plan_download


@pytest.fixture
def cached_snapshot(tmp_path, monkeypatch):
    from huggingface_hub import constants
    from huggingface_hub.errors import OfflineModeIsEnabled

    hub_cache = tmp_path / "hub"
    local_repo = hub_cache / "models--fixture--offline-diffusion"
    snapshot = local_repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (local_repo / "refs").mkdir()
    (local_repo / "refs/main").write_text(snapshot.name)
    monkeypatch.setenv("ACPROF_MODEL_SOURCE", "huggingface")
    monkeypatch.setenv("ACPROF_MODEL_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(hub_cache))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", Mock(side_effect=OfflineModeIsEnabled("offline")))
    return snapshot


@pytest.mark.parametrize(("revision", "pipeline", "task"), [
    ("a" * 40, "StableDiffusionPipeline", "text-to-image"),
    ("main", "DDPMPipeline", "unconditional-image-generation"),
])
def test_offline_index_identifies_the_pipeline_and_its_source(cached_snapshot, revision, pipeline, task):
    from acprof.host.preparation.task_support import require_task_support

    index = {"_class_name": pipeline, "unet": ["diffusers", "UNet2DModel"]}
    raw = json.dumps(index, indent=2).encode()
    (cached_snapshot / "model_index.json").write_bytes(raw)
    if pipeline == "DDPMPipeline":
        (cached_snapshot / "config.json").write_text('{"_class_name":"UNet2DModel"}')

    result = detect_task("fixture/offline-diffusion", revision=revision)
    require_task_support(result)

    assert (result.pipeline_tag, result.task_family, result.runtime_backend) == (task, "diffusion", "diffusers")
    assert result.model_revision == cached_snapshot.name
    assert result.hub_metadata == {}
    assert result.repository_metadata == {"model_index.json": index}
    assert [item["interface"] for item in result.task_discovery["tasks"]] == [task]
    resolution = result.model_resolution
    assert resolution["semantics"]["status"] == "inferred"
    source = resolution["provenance"]["sources"]["model_index.json"]
    assert source["sha256"] == hashlib.sha256(raw).hexdigest()
    assert source["revision"] == cached_snapshot.name
    assert source["encoding"] == "raw"
    assert any((item["source_id"], item["field"], item.get("value")) == ("model_index.json", "/_class_name", pipeline)
               for item in resolution["provenance"]["observations"])
    assert resolution["compatibility"]["status"] == "unknown"
    assert resolution["compatibility"]["checkpoint_verified"] is False
    assert resolution["runtime_validation"]["status"] == "not_run"


@pytest.mark.parametrize("pipeline", ["UnregisteredPipeline", "StableDiffusionInpaintPipeline"])
def test_cached_pipeline_without_a_task_hint_keeps_its_index(cached_snapshot, pipeline):
    (cached_snapshot / "model_index.json").write_text(json.dumps({"_class_name": pipeline}))
    result = detect_task("fixture/offline-diffusion", revision=cached_snapshot.name)
    assert result.repository_metadata["model_index.json"]["_class_name"] == pipeline
    assert result.model_resolution["status"] == "needs_configuration"
    assert not result.model_resolution["selection"]["task"]
    assert not result.task_discovery["tasks"]
    assert any(pipeline in detail and "--task" in detail for detail in result.model_resolution["missing"])


def test_unrelated_index_leaves_transformer_config_discovery_available(cached_snapshot):
    (cached_snapshot / "model_index.json").write_text('{"evaluation_results":[]}')
    (cached_snapshot / "config.json").write_text('{"model_type":"bert","architectures":["BertForMaskedLM"]}')
    result = detect_task("fixture/offline-diffusion")
    assert (result.pipeline_tag, result.library_name, result.model_revision) == ("fill-mask", "transformers", cached_snapshot.name)


def test_cached_pipeline_cannot_supply_a_different_requested_revision(cached_snapshot):
    from acprof.host.preparation.model_errors import ModelLookupError

    (cached_snapshot / "model_index.json").write_text('{"_class_name":"StableDiffusionXLPipeline"}')
    with pytest.raises(ModelLookupError) as caught:
        detect_task("fixture/offline-diffusion", revision="b" * 40)
    assert caught.value.revision == "b" * 40
    assert caught.value.reason_code == "offline"


def test_offline_index_respects_repository_access(cached_snapshot, monkeypatch):
    import httpx
    from huggingface_hub.errors import GatedRepoError

    from acprof.host.preparation.model_errors import ModelLookupError

    (cached_snapshot / "model_index.json").write_text('{"_class_name":"DDPMPipeline"}')
    response = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/api/models/fixture/offline-diffusion"))
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", Mock(side_effect=GatedRepoError("access denied", response=response)))
    with pytest.raises(ModelLookupError) as caught:
        detect_task("fixture/offline-diffusion", revision=cached_snapshot.name, override_tag="text-to-image")
    assert caught.value.reason_code == "access_denied"


@pytest.mark.parametrize("has_store_index", [True, False])
def test_modelscope_index_fallback_uses_only_its_own_store(cached_snapshot, tmp_path, monkeypatch, has_store_index):
    from acprof.host.preparation.model_errors import ModelLookupError
    from acprof.models.download_plan import PLAN_FILENAME, seal_plan

    (cached_snapshot / "model_index.json").write_text('{"_class_name":"StableDiffusionPipeline"}')
    if has_store_index:
        entry = tmp_path / "store" / "entries" / ("c" * 64)
        snapshot = entry / "hf/models--fixture--offline-diffusion/snapshots" / cached_snapshot.name
        snapshot.mkdir(parents=True)
        raw = b'{"_class_name":"DDPMPipeline"}'
        (snapshot / "model_index.json").write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        plan = plan_download(model_id="fixture/offline-diffusion", revision=cached_snapshot.name,
                             family="diffusion", backend="diffusers", policy="full",
                             files={"model_index.json": {"size": len(raw), "lfs_sha256": digest}}, read_json=lambda _: {})
        plan.update(source="modelscope", verification="sha256")
        plan["files"][0]["sha256"] = digest
        (entry / PLAN_FILENAME).write_text(json.dumps(seal_plan(plan)))

    monkeypatch.setenv("ACPROF_MODEL_SOURCE", "modelscope")
    monkeypatch.setattr("acprof.hub.repository.modelscope_info", Mock(side_effect=ConnectionError("offline")))
    if has_store_index:
        result = detect_task("fixture/offline-diffusion", revision=cached_snapshot.name)
        assert (result.model_source, result.pipeline_tag) == ("modelscope", "unconditional-image-generation")
    else:
        with pytest.raises(ModelLookupError) as caught:
            detect_task("fixture/offline-diffusion", revision=cached_snapshot.name)
        assert caught.value.reason_code == "network_error"


@pytest.mark.parametrize(("raw", "pointer"), [
    (b'{"_class_name":"StableDiffusionPipeline","unet":[{}, "UNet"]}', "/unet/0"),
    (b'[]', ""),
])
def test_invalid_cached_index_blocks_before_a_manual_task_or_component_config(cached_snapshot, raw, pointer):
    (cached_snapshot / "model_index.json").write_bytes(raw)
    (cached_snapshot / "config.json").write_text('{"architectures":["BertForMaskedLM"]}')
    result = detect_task("fixture/offline-diffusion", override_tag="text-to-image", revision=cached_snapshot.name)
    failure = result.model_resolution["failure"]
    assert failure["reason_code"] == "metadata_invalid"
    diagnostic = failure["evidence"]["diagnostics"][0]
    assert (diagnostic["file"], diagnostic["json_pointer"]) == ("model_index.json", pointer)
    assert diagnostic["raw_sha256"] == hashlib.sha256(raw).hexdigest()


def test_oversized_cached_index_cannot_authorize_an_explicit_task(cached_snapshot):
    from acprof.host.detect import METADATA_MAX_BYTES
    from acprof.models.files import ModelFilesError

    (cached_snapshot / "model_index.json").write_bytes(b" " * (METADATA_MAX_BYTES + 1))
    with pytest.raises(ModelFilesError, match="exceeds"):
        detect_task("fixture/offline-diffusion", revision=cached_snapshot.name, override_tag="text-to-image")


@pytest.mark.parametrize(('pipeline', 'task'), [
    ('DDPMPipeline', 'unconditional-image-generation'),
    ('StableDiffusionXLPipeline', 'text-to-image'),
    ('PNDMPipeline', 'unconditional-image-generation'),
    ('ShapEPipeline', 'text-to-3d'),
    ('CogVideoXPipeline', 'text-to-video'),
    ('WanPipeline', 'text-to-video'),
])
def test_native_pipeline_task_hints_remain_available_without_claiming_execution(tmp_path, monkeypatch, pipeline, task):
    metadata = tmp_path / 'model_index.json'
    metadata.write_text(json.dumps({'_class_name': pipeline}))
    hub = SimpleNamespace(pipeline_tag=None, library_name='diffusers', sha='a' * 40,
                          siblings=[SimpleNamespace(rfilename='model_index.json')],
                          safetensors=None, card_data={}, config={}, tags=['diffusers'])
    monkeypatch.setattr('huggingface_hub.HfApi.model_info', Mock(return_value=hub))
    download = Mock(side_effect=[SimpleNamespace(file_size=metadata.stat().st_size, commit_hash='a' * 40), str(metadata)])
    monkeypatch.setattr('huggingface_hub.hf_hub_download', download)
    monkeypatch.setattr('acprof.host.model_store.store.cached_model_info', lambda *_args, **_kwargs: None)
    monkeypatch.setattr('acprof.host.model_store.store.cached_metadata', lambda *_args, **_kwargs: None)
    result = detect_task('fixture/pipeline-capabilities', revision='a' * 40)
    assert result.hub_metadata['task_hint'] == task
    assert result.hub_metadata['task_hint_source'] == 'model_index'
    assert result.model_resolution['runtime_validation']['status'] == 'not_run'
    assert [(call.kwargs['filename'], call.kwargs.get('dry_run', False)) for call in download.call_args_list] == [
        ('model_index.json', True), ('model_index.json', False),
    ]


@pytest.mark.parametrize(('pipeline', 'reason'), [
    ('PNDMPipeline', 'unverified_diffusers_pipeline_layout'),
    ('ShapEPipeline', 'unverified_diffusers_pipeline_layout'),
    ('CogVideoXPipeline', 'unverified_diffusers_pipeline_layout'),
    ('WanPipeline', 'unverified_diffusers_pipeline_layout'),
    ('CustomPipeline', 'unregistered_diffusers_pipeline'),
])
def test_unverified_pipeline_retains_all_assets_and_unknown_layout(pipeline, reason):
    files = ('model_index.json', 'unet/config.json', 'unet/diffusion_pytorch_model.safetensors', 'extra.ckpt')
    metadata = {'model_index.json': {'_class_name': pipeline, 'unet': ['diffusers', 'UNet2DModel']}}
    plan = plan_download(model_id='fixture/pipeline', revision='a' * 40, family='diffusion', backend='diffusers',
                         files={name: {'size': 1} for name in files}, read_json=metadata.__getitem__)
    assert plan['effective_policy'] == 'full'
    assert plan['weights'] == []
    assert {entry['path'] for entry in plan['files']} == set(files)
    assert plan['reason'] == reason
    task = TaskInfo('fixture/pipeline', 'text-to-image', 'diffusion', 'diffusers', 'diffusers', 'a' * 40,
                    'test', repository_files=files, repository_metadata=metadata)
    layout = inspect_artifact_layout(task)
    assert layout['status'] == 'unknown'
    assert reason in layout['detail']
