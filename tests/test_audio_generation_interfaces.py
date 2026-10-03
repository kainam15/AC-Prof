"""Audio generation uses native interfaces, independent of checkpoint names."""
import contextlib
import sys
import types
import wave
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from acprof.container.handlers.multimodal import MultimodalHandler
from acprof.host.detect import TaskInfo
from acprof.host.task_support import TaskSupportError, require_task_support
from acprof.runtime_profiles import select_runtime_profile
from tests.test_multimodal_handler import audio_payload


def task(model_type, **kwargs):
    values = dict(model_id='unseen/audio-checkpoint', pipeline_tag='audio-text-to-text',
                  task_family='multimodal', runtime_backend='transformers_model',
                  library_name='transformers', model_revision='a' * 40,
                  detection_method='hub_api', model_config={'model_type': model_type})
    values.update(kwargs)
    return TaskInfo(**values)


@pytest.mark.parametrize('architecture,library', (('voxtral', 'vllm'), ('qwen2_audio', 'transformers'), ('granite_speech', 'transformers')))
def test_native_audio_architectures_share_preflight_without_id_rules(architecture, library):
    info = task(architecture, library_name=library)
    require_task_support(info)
    assert (select_runtime_profile(info).profile_id) == ('multimodal-transformers4576')
    assert (info.model_adapter) == ('family-default')
    assert (info.model_resolution['status']) == ('candidate')

def test_new_auto_registration_selects_new_environment():
    info = task('audioflamingo3')
    require_task_support(info)
    assert (select_runtime_profile(info).profile_id) == ('multimodal-transformers560-cu128')

def test_text_generation_subconfig_uses_registered_auto_not_full_speech_output():
    from acprof import model_resolution

    config = {'model_type': 'qwen2_5_omni', 'thinker_config': {
        'model_type': 'qwen2_5_omni_thinker', 'audio_config': {'model_type': 'qwen2_5_omni_audio_encoder'},
    }}
    info = task('qwen2_5_omni', model_config=config)
    require_task_support(info)
    assert (select_runtime_profile(info).profile_id) == ('multimodal-transformers560-cu128')
    assert (model_resolution.audio_text_loader('5.6.0', config)) == (('AutoModelForImageTextToText', 'thinker_config'))

@pytest.mark.parametrize('config', ({'model_type': 'not_registered'}, {'model_type': 'custom', 'auto_map': {'AutoConfig': 'custom.Config'}}))
def test_unknown_or_custom_architectures_still_fail_before_build(config):
    with pytest.raises(TaskSupportError):
        require_task_support(task(config['model_type'], model_config=config))

@pytest.mark.parametrize('config', ({'model_type': 'not_registered', 'head': {'model_type': 'qwen2_5_omni_thinker'}}, {'model_type': 'qwen2_5_omni', 'text_config': {'model_type': 'qwen2'}}, {'model_type': 'qwen2_5_omni', 'head_a': {'model_type': 'qwen2_5_omni_thinker'}, 'head_b': {'model_type': 'qwen3_omni_moe_thinker'}}))
def test_submodels_cannot_hide_an_unknown_parent_or_discard_audio(config):
    with pytest.raises(TaskSupportError):
        require_task_support(task(config['model_type'], model_config=config))


class TestAudioGenerationHandler:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.handler = MultimodalHandler()
        self.torch = types.SimpleNamespace(float16='fp16', float32='fp32', inference_mode=contextlib.nullcontext)
        self.processor = Mock()
        self.processor.chat_template = None
        self.processor.feature_extractor = types.SimpleNamespace(sampling_rate=16000, n_samples=480000)
        self.processor.batch_decode.return_value = ['sound']
        self.processor.tokenizer.encode.return_value = [42]
        self.model = Mock(device='cpu')
        self.model.config = types.SimpleNamespace(is_encoder_decoder=False)
        self.model.generate.return_value = np.array([[1, 2, 42]])
        self.ctx = dict(model=self.model, processor=self.processor, task_type='audio-text-to-text',
                        mode='generate', model_type='voxtral', device='cpu')
        self.request = {'samples': [{'text': 'Describe the sound.', 'audio_base64': audio_payload(), 'sampling_rate': 16000}],
                        'params': {'max_new_tokens': 2}}

    @pytest.mark.parametrize('architecture', ('voxtral', 'qwen2_audio', 'granite_speech'))
    def test_load_uses_the_same_auto_registry_as_preflight(self, architecture):
        config = types.SimpleNamespace(model_type=architecture, to_dict=lambda: {'model_type': architecture})
        transformers = types.SimpleNamespace(__version__='4.57.6', AutoConfig=Mock(), AutoProcessor=Mock(),
                                               AutoModelForSeq2SeqLM=Mock())
        transformers.AutoConfig.from_pretrained.return_value = config
        transformers.AutoProcessor.from_pretrained.return_value = self.processor
        transformers.AutoModelForSeq2SeqLM.from_pretrained.return_value = self.model
        with patch.dict(sys.modules, torch=self.torch, transformers=transformers):
            ctx = self.handler.load('unseen/audio-checkpoint', 'audio-text-to-text', 'transformers_model',
                                    'cpu', 'a' * 40, {'attention_implementation': 'eager'})
        kwargs = transformers.AutoModelForSeq2SeqLM.from_pretrained.call_args.kwargs
        assert (kwargs['revision']) == ('a' * 40)
        assert (kwargs['attn_implementation']) == ('eager')
        assert not (kwargs['trust_remote_code'])
        assert (ctx['model']) is (self.model)

    def test_native_chat_processor_receives_audio_and_text_without_jinja_requirement(self):
        paths = []

        def native(messages, **kwargs):
            assert (set(kwargs)) == ({'tokenize', 'return_dict', 'return_tensors'})
            assert (len(messages)) == (1)
            content = messages[0]['content']
            audio = next(item for item in content if item['type'] == 'audio')
            assert (next(item['text'] for item in content if item['type'] == 'text')) == ('Describe the sound.')
            paths.append(Path(audio['path']))
            with wave.open(str(paths[-1]), 'rb') as wav:
                assert ((wav.getframerate(), wav.getnframes())) == ((16000, 160))
            assert (kwargs['tokenize'])
            assert (kwargs['return_dict'])
            assert (kwargs['return_tensors']) == ('pt')
            return {'input_ids': np.array([[1, 2]]), 'input_features': np.ones((1, 8, 10))}

        self.processor.apply_chat_template.side_effect = native
        processed = self.handler.preprocess(self.ctx, self.request)
        assert not (paths[0].exists())
        self.processor.assert_not_called()
        self.model.generate.assert_not_called()
        with patch.dict(sys.modules, torch=self.torch):
            result = self.handler.postprocess(self.ctx, self.handler.predict(self.ctx, processed))
        assert (result['texts']) == (['sound'])
        assert (result['output_token_count']) == (1)
        assert (processed['_workload']['input']['audio']['audio_seconds']) == (0.01)

    @pytest.mark.parametrize('audio_case', range(3), ids=['{}', "{'input_features': None}", "{'input_features': np.empty((0, 8))}"])
    def test_native_processor_cannot_drop_audio_or_return_empty_features(self, audio_case):
        audio = tuple(({}, {'input_features': None}, {'input_features': np.empty((0, 8))}))[audio_case]
        with pytest.raises(ValueError, match='audio'):
            self.processor.apply_chat_template.return_value = {'input_ids': np.array([[1, 2]]), **audio}
            self.handler.preprocess(self.ctx, self.request)

    def test_native_processor_failure_removes_temporary_audio(self):
        paths = []

        def native(messages, **kwargs):
            paths.append(Path(messages[0]['content'][0]['path']))
            raise ValueError('invalid audio template')

        self.processor.apply_chat_template.side_effect = native
        with pytest.raises(ValueError, match='invalid audio template'):
            self.handler.preprocess(self.ctx, self.request)
        assert not (paths[0].exists())

    def test_processor_kwargs_are_routed_through_the_native_api_signature(self):
        def native(messages, *, processor_kwargs, **kwargs):
            assert (processor_kwargs['audio_kwargs']['sampling_rate']) == (16000)
            assert (processor_kwargs['padding'])
            assert ('sampling_rate') not in (kwargs)
            return {'input_ids': np.array([[1, 2]]), 'input_features': np.ones((1, 8, 10))}

        self.processor.apply_chat_template = native
        self.handler.preprocess(self.ctx, self.request)
