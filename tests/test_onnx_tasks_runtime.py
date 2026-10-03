"""Offline real ORT task adapters; synthetic graphs prove interfaces, not model quality."""
import base64
import importlib.util
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from examples.onnxruntime.fixtures import create_image_fixture, create_text_fixture

pytestmark = pytest.mark.runtime


@pytest.mark.skipif(not (all(importlib.util.find_spec(name) for name in
                        ('onnx', 'onnxruntime', 'numpy', 'PIL', 'tokenizers'))), reason='requires the no-Torch ONNX Runtime CPU container with task dependencies')
class TestONNXTaskRuntime:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        assert (importlib.util.find_spec('torch')) is None, 'ONNX tests require Torch to be absent'
        assert (importlib.util.find_spec('transformers')) is None
        self.temporary = tmp_path
        self.root = Path(str(self.temporary))

    @staticmethod
    def image_payload(side=4):
        from PIL import Image
        image = Image.new('RGB', (side, side), (255, 0, 0))
        stream = io.BytesIO()
        image.save(stream, format='PNG')
        return {'image_base64': base64.b64encode(stream.getvalue()).decode(),
                'input_scale': side / 224, 'batch_size': 1}

    def load(self, family, root, task):
        from acprof.container.handlers import HandlerRegistry
        handler = HandlerRegistry.get(family, 'onnxruntime')
        return handler, handler.load(str(root), task, 'onnxruntime', 'cpu')

    def test_image_numeric_reference_and_actual_geometry(self):
        import numpy as np
        handler, context = self.load('cv', create_image_fixture(self.root), 'image-classification')
        for side in (4, 8):
            payload = self.image_payload(side)
            prepared = handler.preprocess(context, payload)
            output = handler.predict(context, prepared)
            np.testing.assert_allclose(output['scores'], [[1 / 3, -1 / 3]], atol=1e-6)
            response = handler.postprocess(context, output)
            assert (response['output_type']) == ('classification')
            report = handler.validate_output(context, payload, prepared, output, response)
            assert (response['predicted_classes']) == ([0])
            images = report['workload_contract']['input']['images']
            assert (images['processed_shape']) == ([1, 3, side, side])
            assert (images['processed_resolution']) == ([side, side])
            assert (report['workload_contract']['input']['tensors']) == ({'images': {'dtype': 'float32', 'shape': [1, 3, side, side]}})
            assert (report['protocol']['status']) == ('verified')
        assert (context['runtime_parameters']['effective']['providers']) == (['CPUExecutionProvider'])
        assert (context['runtime_parameters']['inputs']) == ([{'name': 'images', 'type': 'tensor(float)', 'shape': ['batch', 3, 'height', 'width']}])
        assert (len(context['artifact_sha256'])) == (64)

    def test_fixed_image_shape_rejects_implicit_resize_but_honors_explicit_processing(self):
        handler, context = self.load('cv', create_image_fixture(self.root / 'fixed', fixed_size=4),
                                     'image-classification')
        with pytest.raises(ValueError, match='shape'):
            handler.preprocess(context, self.image_payload(8))
        handler, context = self.load('cv', create_image_fixture(self.root / 'resize', fixed_size=4, resize=4),
                                     'image-classification')
        payload = self.image_payload(8)
        prepared = handler.preprocess(context, payload)
        output = handler.predict(context, prepared)
        report = handler.validate_output(context, payload, prepared, output, handler.postprocess(context, output))
        images = report['workload_contract']['input']['images']
        assert (images['original_resolution']) == ([8, 8])
        assert (images['processed_resolution']) == ([4, 4])

    def test_text_multiple_named_integer_inputs_outputs_and_actual_tokens(self):
        import numpy as np
        handler, context = self.load('nlp', create_text_fixture(self.root), 'text-classification')
        payload = {'text': 'hello world', 'batch_size': 1, 'input_scale': 2}
        prepared = handler.preprocess(context, payload)
        assert (set(prepared['inputs'])) == ({'input_ids', 'attention_mask', 'token_type_ids'})
        assert (all(value.dtype == np.int64 for value in prepared['inputs'].values()))
        output = handler.predict(context, prepared)
        np.testing.assert_array_equal(output['logits'], [[14., -14.]])
        np.testing.assert_array_equal(output['token_count'], [[4]])
        response = handler.postprocess(context, output)
        assert (response['output_type']) == ('label')
        report = handler.validate_output(context, payload, prepared, output, response)
        text = report['workload_contract']['input']['text']
        assert (text['tokens']) == (4)
        assert (text['padding']) == ('none')
        assert (text['truncation']) == ('reject')
        assert (prepared['_effective_input_scale']) == (2)
        assert (text['actual_tokens_per_sample']) == ([4])
        assert (report['workload_contract']['input']['tensors']['input_ids']) == ({'dtype': 'int64', 'shape': [1, 4]})
        assert ({item['name'] for item in context['runtime_parameters']['outputs']}) == ({'logits', 'token_count'})

    @pytest.mark.parametrize('payload,reason', [({'text': 'hello world hello', 'batch_size': 1}, 'max_length'), ({'text': 'hello', 'batch_size': 2}, 'batch'), ({'text': ['hello', 'hello world'], 'batch_size': 2}, 'padding'), ({'text': 'hello world', 'input_scale': 5}, 'input_scale')])
    def test_text_rejects_truncation_padding_and_sample_replication(self, payload, reason):
        handler, context = self.load('nlp', create_text_fixture(self.root, max_length=4), 'text-classification')
        with pytest.raises(ValueError, match=reason):
            handler.preprocess(context, payload)
        handler, context = self.load('nlp', create_text_fixture(self.root / 'fixed', fixed_length=5),
                                     'text-classification')
        with pytest.raises(ValueError, match='shape'):
            handler.preprocess(context, {'text': 'hello world', 'batch_size': 1})

    def test_text_rejects_invalid_config_and_wrong_named_input_dtype(self):
        import numpy as np

        from acprof.container.onnx_session import validate_inputs
        root = create_text_fixture(self.root)
        handler, context = self.load('nlp', root, 'text-classification')
        prepared = handler.preprocess(context, {'text': 'hello', 'batch_size': 1})
        bad = dict(prepared['inputs'], input_ids=prepared['inputs']['input_ids'].astype(np.float32))
        with pytest.raises(ValueError, match='dtype'):
            validate_inputs(context, bad)
        with pytest.raises(ValueError, match='named inputs'):
            validate_inputs(context, {'input_ids': prepared['inputs']['input_ids']})
        manifest = json.loads((root / 'acprof_model.json').read_text())
        manifest['tokenizer']['padding'] = 'longest'
        (root / 'acprof_model.json').write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match='padding'):
            handler.load(str(root), 'text-classification', 'onnxruntime', 'cpu')

    def test_equal_length_text_batch_preserves_sample_order_and_dynamic_sequence(self):
        import numpy as np
        handler, context = self.load('nlp', create_text_fixture(self.root), 'text-classification')
        for texts, expected in [(['hello', 'world'], [[9., -9.], [10., -10.]]),
                                (['hello world', 'world world'], [[14., -14.], [15., -15.]])]:
            prepared = handler.preprocess(context, {'text': texts, 'batch_size': 2})
            output = handler.predict(context, prepared)
            np.testing.assert_array_equal(output['logits'], expected)
            assert (prepared['_workload']['input']['text']['actual_tokens_per_sample']) == ([len(texts[0].split()) + 2] * 2)

    def test_single_score_does_not_invent_labels_or_probabilities(self):
        import numpy as np
        handler, context = self.load('nlp', create_text_fixture(self.root), 'text-classification')
        response = handler.postprocess(context, {'logits': np.asarray([[-2.5]], dtype=np.float32)})
        assert (response['output_type']) == ('label')
        assert (response['score_interpretation']) == ('single_score')
        assert ('predicted_classes') not in (response)
        assert ('probabilities') not in (response)

    def test_artifact_hash_is_unverified_at_load_and_checked_outside_execution(self):
        root = create_text_fixture(self.root)
        manifest = json.loads((root / 'acprof_model.json').read_text())
        manifest['artifact_sha256'] = '0' * 64
        (root / 'acprof_model.json').write_text(json.dumps(manifest))
        handler, context = self.load('nlp', root, 'text-classification')
        assert (context['artifact']['verification']) == ('declared')
        payload = {'text': 'hello', 'batch_size': 1}
        prepared = handler.preprocess(context, payload)
        output = handler.predict(context, prepared)
        assert (context['artifact']['verification']) == ('declared')
        with pytest.raises(ValueError, match='SHA256'):
            handler.validate_output(context, payload, prepared, output, handler.postprocess(context, output))

    @pytest.mark.parametrize('inter_op_case', range(2), ids=['(1, ort.ExecutionMode.ORT_SEQUENTIAL)', '(2, ort.ExecutionMode.ORT_PARALLEL)'])
    def test_inter_op_threads_enable_and_report_actual_parallel_execution_mode(self, inter_op_case):
        import onnxruntime as ort

        root = create_text_fixture(self.root)
        (inter_op, expected) = tuple([(1, ort.ExecutionMode.ORT_SEQUENTIAL), (2, ort.ExecutionMode.ORT_PARALLEL)])[inter_op_case]
        with patch.dict(os.environ, {
            'ACPROF_ONNX_INTRA_OP_THREADS': '2', 'ACPROF_ONNX_INTER_OP_THREADS': str(inter_op),
        }):
            handler, context = self.load('nlp', root, 'text-classification')
            actual = context['model'].get_session_options()
            assert (actual.execution_mode) == (expected)
            effective = context['runtime_parameters']['effective']
            assert (effective['intra_op_threads']) == (actual.intra_op_num_threads)
            assert (effective['inter_op_threads']) == (actual.inter_op_num_threads)
            assert (effective['execution_mode']) == (str(expected).rsplit('.', 1)[-1])
            prepared = handler.preprocess(context, {'text': 'hello'})
            assert (handler.predict(context, prepared)['logits'].tolist()) == ([[9., -9.]])

    def test_output_signature_and_nonfinite_output_fail_validation(self):
        import numpy as np
        handler, context = self.load('nlp', create_text_fixture(self.root), 'text-classification')
        payload = {'text': 'hello', 'batch_size': 1}
        prepared = handler.preprocess(context, payload)
        output = handler.predict(context, prepared)
        response = handler.postprocess(context, output)
        with pytest.raises(ValueError, match='named outputs'):
            handler.validate_output(context, payload, prepared, {'logits': output['logits']}, response)
        malformed = dict(output, logits=np.asarray([[float('nan'), 1]], dtype=np.float32))
        with pytest.raises(ValueError, match='finite'):
            handler.validate_output(context, payload, prepared, malformed, response)

    def test_external_weights_cannot_be_verified_from_graph_hash_alone(self):
        import numpy as np
        import onnx
        from onnx import numpy_helper

        root = create_image_fixture(self.root)
        model = onnx.load(root / 'model.onnx')
        model.graph.initializer.append(numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), name='gain'))
        model.graph.node[-1].output[0] = 'unscaled_scores'
        model.graph.node.append(onnx.helper.make_node('Mul', ['unscaled_scores', 'gain'], ['scores']))
        onnx.save_model(model, root / 'model.onnx', save_as_external_data=True,
                        all_tensors_to_one_file=True, location='weights.bin', size_threshold=0)
        handler, context = self.load('cv', root, 'image-classification')
        payload = self.image_payload()
        prepared = handler.preprocess(context, payload)
        output = handler.predict(context, prepared)
        with pytest.raises(ValueError, match='external tensor data'):
            handler.validate_output(context, payload, prepared, output, handler.postprocess(context, output))

    @pytest.mark.parametrize('family_case', range(2), ids=["('cv', 'image-classification', create_image_fixture(root / 'image'), 0.125)", "('nlp', 'text-classification', create_text_fixture(root / 'text'), 2)"])
    def test_existing_family_generators_feed_new_backends_without_backend_branches(self, family_case):
        from acprof.workloads import get_generator
        (family, task, root, scale) = tuple([('cv', 'image-classification', create_image_fixture(self.root / 'image'), 0.125), ('nlp', 'text-classification', create_text_fixture(self.root / 'text'), 2)])[family_case]
        payload = get_generator(family, 'fixture', task, 1).generate(scale)
        handler, context = self.load(family, root, task)
        prepared = handler.preprocess(context, payload)
        output = handler.predict(context, prepared)
        response = handler.postprocess(context, output)
        report = handler.validate_output(context, payload, prepared, output, response)
        assert (report['protocol']['status']) == ('verified')
