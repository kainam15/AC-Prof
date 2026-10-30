import pytest

from acprof.gpu_mode import normalize_gpu_mode
from acprof.host.largest_scale_probe import select_minimum_resources
from acprof.host.measurement.config import ClientConfig
from acprof.host.profiling.compute_plan import NCU_ERROR_FIELD, find_compute_profile_entry
from acprof.host.profiling.execution import _normalize_gpu_modes
from acprof.host.profiling.execution_plan import ExecutionProfileResolver
from acprof.plotting.data import is_gpu_off, is_gpu_on

ON = ['on', ' ON ', '1', '1.0', 'true', 'YES', ' gpu ', True, 1, 1.0]
OFF = ['off', ' OFF ', '0', '0.0', 'false', 'NO', ' cpu ', False, 0, 0.0]
INVALID = ['', 'auto', 'enabled', 'disabled', 'none', None, 2, -1, float('nan')]


@pytest.mark.parametrize(('value', 'expected'), [(v, 'on') for v in ON] + [(v, 'off') for v in OFF])
def test_all_input_consumers_share_gpu_mode_aliases(value, expected):
    assert normalize_gpu_mode(value) == expected
    assert ClientConfig.from_env({'GPU_MODE': str(value)}).gpu_mode == expected
    assert _normalize_gpu_modes([value, expected]) == [expected]
    assert select_minimum_resources([1], [2], [value]) == (1, 2, expected)
    assert is_gpu_on(value) == (expected == 'on')
    assert is_gpu_off(value) == (expected == 'off')
    result = find_compute_profile_entry({'_load_error': 'test-source'}, value, 1)
    assert result[NCU_ERROR_FIELD] == ('test-source' if expected == 'on' else '')


@pytest.mark.parametrize('value', INVALID)
@pytest.mark.parametrize('consumer', [
    normalize_gpu_mode, lambda value: ClientConfig.from_env({'GPU_MODE': str(value)}),
    lambda value: _normalize_gpu_modes([value]), lambda value: select_minimum_resources([1], [2], [value]),
    is_gpu_on, is_gpu_off, lambda value: find_compute_profile_entry({}, value, 1),
    lambda value: ExecutionProfileResolver({'_disabled': True}).resolve(1, 2, value, 1),
])
def test_invalid_modes_raise_before_becoming_cpu(value, consumer):
    with pytest.raises(ValueError, match='gpu_mode'):
        consumer(value)


def test_aliases_canonicalize_configuration_and_frozen_matrix_before_identity(tmp_path):
    from types import SimpleNamespace

    from acprof.application.resources import resource_matrix
    from acprof.experiments.arguments import build_parser
    from acprof.experiments.configuration import RunConfig
    from acprof.host.matrix_plan import matrix_identity

    raw = build_parser().parse_args(['--model', 'org/model', '--gpus', ' CPU , TRUE ', '--cpus', '1', '--mems', '2'])
    config = RunConfig.from_namespace(raw).validate(project_dir=tmp_path)
    assert config.gpus == 'off,on'
    assert resource_matrix(raw)[2] == ['off', 'on']
    task = SimpleNamespace(model_id='org/model', model_revision='a' * 40)
    image = SimpleNamespace(tag='sha256:test')
    values = dict(order='declared', seed=0, prune=False)
    canonical = matrix_identity(task, image, [1], [2], ['off', 'on'], [32], **values)
    assert matrix_identity(task, image, [1], [2], [' CPU ', True], [32], **values) == canonical
    with pytest.raises(ValueError, match='unique'):
        matrix_identity(task, image, [1], [2], ['on', True], [32], **values)


def test_profile_case_alias_matches_canonical_query():
    from acprof.host.profiling.execution_plan import MASSIF_HEAP_PEAK_FIELD

    plan = {'profiles': [{'cpu_cores': 1, 'mem_cap_gb': 2, 'gpu_mode': ' CPU ',
                         'tools': {'massif': {'entries': [{'input_scale': 32, MASSIF_HEAP_PEAK_FIELD: 123}]}}}]}
    assert ExecutionProfileResolver(plan).resolve(1, 2, 'off', 32)[MASSIF_HEAP_PEAK_FIELD] == 123
