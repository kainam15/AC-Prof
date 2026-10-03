import importlib
import importlib.util
import tempfile
import types
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest


class TestONNXHandler:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
    def handler(self, root, *, input_shape=('N', 4)):
        assert (importlib.util.find_spec('acprof.container.handlers.onnxruntime')) is not None, 'ONNX Runtime handler is missing'
        fake = types.SimpleNamespace(
            __version__='test', SessionOptions=lambda: types.SimpleNamespace(),
            ExecutionMode=types.SimpleNamespace(ORT_PARALLEL='ORT_PARALLEL', ORT_SEQUENTIAL='ORT_SEQUENTIAL'),
            InferenceSession=lambda *args, **kwargs: types.SimpleNamespace(
                get_inputs=lambda: [types.SimpleNamespace(name='x', shape=list(input_shape), type='tensor(float)')],
                get_outputs=lambda: [types.SimpleNamespace(name='y', shape=['N', 1], type='tensor(float)')],
                get_providers=lambda: ['CPUExecutionProvider'],
                get_session_options=lambda: kwargs['sess_options'],
                run=lambda names, inputs: [inputs['x'].sum(axis=1, keepdims=True)],
            ),
        )
        with patch.dict('sys.modules', {'onnxruntime': fake}):
            module = importlib.import_module('acprof.container.handlers.onnxruntime')
            # This unit fake has no actual ONNX graph; real-container tests audit model bytes.
            artifact_check = patch.object(module, 'validate_artifact')
            artifact_check.start()
            self._request.addfinalizer(partial(artifact_check.stop))
            handler = module.ONNXRuntimeHandler()
            with patch.object(module, 'ort', fake):
                context = handler.load(str(root), 'tabular-regression', 'onnxruntime', 'cpu')
        return handler, context

    def test_full_protocol_and_known_numeric_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model.onnx').write_bytes(b'fixture')
            handler, ctx = self.handler(root)
            payload = {'input_scale': 2, 'batch_size': 1,
                       'features': [[1., 2., 3., 4.], [5., 6., 7., 8.]]}
            prepared = handler.preprocess(ctx, payload)
            output = handler.predict(ctx, prepared)
            assert (output.tolist()) == ([[10.], [26.]])
            response = handler.postprocess(ctx, output)
            assert (response['output_shape']) == ([2, 1])
            validation = handler.validate_output(ctx, payload, prepared, output, response)
            assert (validation['protocol']['status']) == ('verified')
            assert (validation['workload_contract']['input']['tensors']) == ({'x': {'dtype': 'float32', 'shape': [2, 4]}})

    def test_fixed_model_batch_is_enforced_without_splitting_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model.onnx').write_bytes(b'fixture')
            handler, ctx = self.handler(root, input_shape=(1, 4))
            with pytest.raises(ValueError, match='fixed|shape'):
                handler.preprocess(ctx, {'input_scale': 2, 'features': [[1.] * 4] * 2})

    @pytest.mark.parametrize('filenames', ([], ['a.onnx', 'b.onnx']))
    def test_missing_or_ambiguous_artifact_is_explicit(self, filenames):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in filenames:
                (root / name).write_bytes(b'fixture')
            with pytest.raises(ValueError, match='one.*onnx|model_file'):
                self.handler(root)

    def test_nonfinite_input_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model.onnx').write_bytes(b'fixture')
            handler, ctx = self.handler(root)
            with pytest.raises(ValueError, match='finite'):
                handler.preprocess(ctx, {'input_scale': 1, 'features': [[float('nan')] * 4]})
