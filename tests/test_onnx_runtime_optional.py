"""真实 ORT 容器接口回归；宿主机不安装推理框架。"""
import importlib.util
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.runtime


@pytest.mark.skipif(not (all(importlib.util.find_spec(name) for name in ('onnx', 'onnxruntime', 'numpy'))), reason='requires the no-Torch ONNX Runtime CPU container')
def test_environment_has_no_torch_or_transformers():
    assert (importlib.util.find_spec('torch')) is None
    assert (importlib.util.find_spec('transformers')) is None

@pytest.mark.skipif(not (all(importlib.util.find_spec(name) for name in ('onnx', 'onnxruntime', 'numpy'))), reason='requires the no-Torch ONNX Runtime CPU container')
def test_dynamic_linear_model_runs_full_handler_and_validation_contract():
    from examples.onnxruntime.smoke import create_linear_fixture, exercise
    with tempfile.TemporaryDirectory() as temporary:
        result = exercise(create_linear_fixture(Path(temporary)))
    assert (result['known_reference_passed'])
    assert not (result['torch_installed'])
    assert (result['response']['output_shape']) == ([4, 1])

@pytest.mark.skipif(not (all(importlib.util.find_spec(name) for name in ('onnx', 'onnxruntime', 'numpy'))), reason='requires the no-Torch ONNX Runtime CPU container')
@pytest.mark.parametrize('count', (1, 3, 7))
def test_dynamic_rows_have_known_results_at_distinct_shapes(count):
    import numpy as np

    from acprof.container.handlers import HandlerRegistry
    from examples.onnxruntime.smoke import create_linear_fixture
    with tempfile.TemporaryDirectory() as temporary:
        handler = HandlerRegistry.get('structured', 'onnxruntime')
        root = create_linear_fixture(Path(temporary))
        context = handler.load(str(root), 'tabular-regression', 'onnxruntime', 'cpu')
        payload = {'input_scale': count, 'batch_size': 1,
                   'features': [[float(row)] * 8 for row in range(count)]}
        prepared = handler.preprocess(context, payload)
        raw = handler.predict(context, prepared)
        np.testing.assert_allclose(raw, [[8.0 * row + 0.25] for row in range(count)],
                                   rtol=0, atol=1e-6)
        response = handler.postprocess(context, raw)
        assert (response['output_shape']) == ([count, 1])
        assert (handler.validate_output(context, payload, prepared, raw,
                                                 response)['protocol']['status']) == ('verified')

@pytest.mark.skipif(not (all(importlib.util.find_spec(name) for name in ('onnx', 'onnxruntime', 'numpy'))), reason='requires the no-Torch ONNX Runtime CPU container')
def test_fixed_batch_cannot_silently_change_workload():
    from acprof.container.handlers import HandlerRegistry
    from examples.onnxruntime.smoke import create_linear_fixture
    with tempfile.TemporaryDirectory() as temporary:
        root = create_linear_fixture(Path(temporary), fixed_rows=1)
        handler = HandlerRegistry.get('structured', 'onnxruntime')
        context = handler.load(str(root), 'tabular-regression', 'onnxruntime', 'cpu')
        with pytest.raises(ValueError, match='fixed input shape'):
            handler.preprocess(context, {'input_scale': 2, 'batch_size': 1,
                                         'features': [[0.0] * 8, [1.0] * 8]})
