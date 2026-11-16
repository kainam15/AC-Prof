"""Task recognition never substitutes for reviewed artifact selection."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.host.detect import TaskInfo, detect_task
from acprof.models.artifact_layout import inspect_artifact_layout
from acprof.models.download_plan import plan_download


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
    monkeypatch.setattr('huggingface_hub.hf_hub_download', lambda **kwargs:
                        SimpleNamespace(file_size=metadata.stat().st_size, commit_hash='a' * 40)
                        if kwargs.get('dry_run') else str(metadata))
    monkeypatch.setattr('acprof.host.model_store.store.cached_model_info', lambda *_args, **_kwargs: None)
    monkeypatch.setattr('acprof.host.model_store.store.cached_metadata', lambda *_args, **_kwargs: None)
    result = detect_task('fixture/pipeline-capabilities', revision='a' * 40)
    assert result.hub_metadata['task_hint'] == task
    assert result.hub_metadata['task_hint_source'] == 'model_index'
    assert result.model_resolution['runtime_validation']['status'] == 'not_run'


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
