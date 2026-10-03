import importlib
import importlib.util
import json

import numpy as np
import pytest


class TestValidationContract:
    def validator(self):
        assert (importlib.util.find_spec('acprof.container.validation')) is not None, 'the shared output validation contract is missing'
        return importlib.import_module('acprof.container.validation').validate_output

    def fixture(self):
        return ({'task_type': 'tabular-regression', 'feature_dim': 2},
                {'input_scale': 2, 'batch_size': 1, 'features': [[1., 2.], [3., 4.]]},
                {'_effective_input_scale': 2., 'args': (np.ones((2, 2)),)},
                np.array([[3.], [7.]]),
                {'task': 'tabular-regression', 'output_type': 'regression',
                 'output_shape': [2, 1], 'n_results': 2})

    def test_valid_shape_finite_workload_and_serialization(self):
        result = self.validator()(*self.fixture())
        assert (result['protocol']['status']) == ('verified')
        assert (result['task']['status']) == ('verified')
        contract = result['workload_contract']
        assert (contract['input']['rows']) == (2)
        assert (contract['output']['shape']) == ([2, 1])
        assert (contract['scenario']) == ({'type': 'serial'})
        json.dumps(result, allow_nan=False)

    def test_wrong_output_shape_rejected(self):
        args = list(self.fixture())
        args[3] = np.ones((1, 1))
        args[4] = {**args[4], 'output_shape': [1, 1], 'n_results': 1}
        with pytest.raises(ValueError, match='shape|rows'):
            self.validator()(*args)

    @pytest.mark.parametrize('bad_case', range(3), ids=["float('nan')", "float('inf')", "-float('inf')"])
    def test_nan_and_inf_raw_output_rejected_even_when_summary_is_finite(self, bad_case):
        bad = tuple((float('nan'), float('inf'), -float('inf')))[bad_case]
        args = list(self.fixture())
        args[3][0, 0] = bad
        with pytest.raises(ValueError, match='finite'):
            self.validator()(*args)

    def test_effective_workload_mismatch_rejected(self):
        args = list(self.fixture())
        args[2]['_effective_input_scale'] = 1.
        with pytest.raises(ValueError, match='scale'):
            self.validator()(*args)

    def test_missing_output_field_rejected(self):
        args = list(self.fixture())
        del args[4]['output_shape']
        with pytest.raises(ValueError, match='output_shape'):
            self.validator()(*args)

    def test_requested_generation_limit_is_never_used_as_actual_count(self):
        result = self.validator()(
            {'task_type': 'text-generation'},
            {'params': {'max_new_tokens': 256}, 'batch_size': 1},
            {'_effective_input_scale': 5, 'text': 'hello'},
            [{'generated_text': 'hello world'}],
            {'task': 'text-generation', 'output_type': 'text', 'n_results': 1},
        )
        generation = result['workload_contract']['generation']
        assert (generation['max_output_tokens']) == (256)
        assert (generation['actual_output_tokens']) is None
        assert (generation['actual_output_tokens_status']) == ('unavailable')
        assert (generation['stop_reason']) is None

    def test_json_nonfinite_and_missing_task_rejected(self):
        for response in ({'task': 'tabular-regression', 'score': float('inf')}, {}):
            args = list(self.fixture())
            args[4] = response
            with pytest.raises(ValueError):
                self.validator()(*args)

    @pytest.mark.parametrize('position,replacement,expected', ((3, None, 'raw output'), (2, {}, 'actual.*scale')))
    def test_missing_raw_output_or_effective_scale_cannot_verify(self, position, replacement, expected):
        args = list(self.fixture())
        args[position] = replacement
        with pytest.raises(ValueError, match=expected):
            self.validator()(*args)

    @pytest.mark.parametrize('missing', ('output_type', 'n_results'))
    @pytest.mark.parametrize('task', ('text-classification', 'image-classification'))
    def test_nlp_and_cv_require_actual_output_summary_fields(self, missing, task):
        base = {'task': task, 'output_type': 'classification', 'n_results': 1}
        response = {key: value for key, value in base.items() if key != missing}
        with pytest.raises(ValueError, match=missing):
            self.validator()({'task_type': task}, {'input_scale': 1},
                             {'_effective_input_scale': 1}, [{'label': 'a', 'score': 0.9}], response)

    @pytest.mark.parametrize('task,output_type', (('text-generation', 'text'), ('automatic-speech-recognition', 'transcription')))
    def test_generation_requires_text_or_observed_tokens(self, task, output_type):
        response = {'task': task, 'output_type': output_type, 'n_results': 1, 'text': ''}
        with pytest.raises(ValueError, match='empty|evidence'):
            self.validator()({'task_type': task}, {'input_scale': 1},
                             {'_effective_input_scale': 1}, {'text': ''}, response)

    def test_missing_shape_is_explicit_and_is_not_claimed_as_checked(self):
        report = self.validator()({'task_type': 'text-generation'}, {'input_scale': 1},
                                 {'_effective_input_scale': 1}, [{'generated_text': 'hello'}],
                                 {'task': 'text-generation', 'output_type': 'text', 'n_results': 1})
        assert ('shape') not in (report['protocol']['checks'])
        assert (report['protocol']['aspects']['shape']['status']) == ('unavailable')

    @pytest.mark.parametrize('status', ('unsupported', 'unavailable', 'error'))
    def test_runtime_probe_refuses_custom_validator_unverified_status(self, status):
        import os
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import patch

        from acprof.container.runtime_validate import validate

        handler = SimpleNamespace(
            preprocess=lambda *args: {'_effective_input_scale': 1},
            predict=lambda *args: [[1.]], postprocess=lambda *args: {'task': 'tabular-regression'},
            validate_output=lambda *args: {'protocol': {'status': 'verified'},
                                          'task': {'status': status}, 'workload_contract': {}},
        )
        runtime = SimpleNamespace(inference_context=nullcontext, metadata=lambda: {})
        with patch.dict(os.environ, {
            'TASK_FAMILY': 'structured', 'RUNTIME_BACKEND': 'onnxruntime',
            'TASK_TYPE': 'tabular-regression', 'MODEL_ID': 'test/model', 'MODEL_REVISION': 'fixed',
        }), patch('acprof.container.handlers.HandlerRegistry.get', return_value=handler), patch(
            'acprof.container.handlers.load_handler', return_value={},
        ), patch('acprof.container.handlers.resolve_model_source', return_value='test/model'), patch(
            'acprof.container.execution.configured_execution', return_value=(runtime, 'cpu'),
        ), pytest.raises(ValueError, match='validation.*verified'):
            validate({'input_scale': 1})
