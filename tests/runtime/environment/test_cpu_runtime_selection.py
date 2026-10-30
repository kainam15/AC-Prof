"""CPU-only requests select the existing CPU lock before image preparation."""

from unittest.mock import patch

import pytest

from acprof.host.detect import TaskInfo
from acprof.host.images.runtime import configure_runtime_profile
from acprof.host.preparation.task_support import require_task_support


def _task(devices):
    task = TaskInfo(model_id='google-t5/t5-small', pipeline_tag='summarization',
                    task_family='nlp', runtime_backend='transformers_model',
                    library_name='transformers', model_revision='a' * 40,
                    detection_method='test', model_config={'model_type': 't5'})
    require_task_support(task, devices=devices)
    return task


@pytest.mark.parametrize('devices', [('cpu',), ('off',), ('cpu', 'off')])
def test_cpu_only_requests_use_cpu_lock_without_a_driver_query(devices):
    task = _task(devices)
    with patch.dict('os.environ', {}, clear=True), patch(
        'acprof.host.images.runtime._host_cuda_version',
        side_effect=AssertionError('CPU-only preparation must not inspect the GPU driver'),
    ):
        profile = configure_runtime_profile(task)
        assert profile.profile_id == 'nlp-cpu'
        assert profile.environment.platform.torch_version.endswith('+cpu')
        # Image preparation invokes the same selection again.
        assert configure_runtime_profile(task) == profile
    assert task.model_resolution['requested_devices'] == list(devices)
    assert task.model_resolution['runtime_profile'] == 'nlp-cpu'


@pytest.mark.parametrize('devices', [('gpu',), ('on',), ('off', 'on')])
@pytest.mark.parametrize(('driver', 'expected'), [((12, 4), 'nlp-cu124'), ((12, 8), 'nlp-cu128')])
def test_gpu_and_mixed_requests_keep_driver_compatible_locks(devices, driver, expected):
    task = _task(devices)
    with patch.dict('os.environ', {}, clear=True), patch(
        'acprof.host.images.runtime._host_cuda_version', return_value=driver,
    ):
        assert configure_runtime_profile(task).profile_id == expected


def test_explicit_torch_platform_still_overrides_cpu_default():
    task = _task(('off',))
    with patch.dict('os.environ', {'ACPROF_NLP_TORCH_INDEX_URL': 'https://download.pytorch.org/whl/cu124'},
                    clear=True), patch('acprof.host.images.runtime._host_cuda_version') as driver:
        assert configure_runtime_profile(task).profile_id == 'nlp-cu124'
    driver.assert_not_called()
