import base64
import contextlib
import io
import sys
import tempfile
import types
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from acprof.container.handlers.audio import AudioHandler
from acprof.workloads.audio import AudioWorkloadGenerator


class TextTokenizer:
    model_max_length = 6

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)

    def num_special_tokens_to_add(self, pair=False):
        return 2


class TestAudioTaskExtension:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.handler = AudioHandler()

    @pytest.mark.parametrize('task', ('text-to-speech', 'text-to-audio'))
    def test_text_audio_workloads_are_deterministic_text_and_keep_scale_kind(self, task):
        generator = AudioWorkloadGenerator("model", task, 1)
        payload = generator.generate(4)
        assert (payload) == (generator.generate_for_word_count(4))
        assert (len(payload["text"].split())) == (4)
        assert ("audio_base64") not in (payload)
        assert (generator.plan_metadata()["input_scale"]["type"]) == ("seq_length")
        assert (generator.scale_label(4)) == ("seq4")

    @pytest.mark.parametrize('task', ('audio-classification', 'audio-to-audio', 'voice-activity-detection'))
    def test_waveform_tasks_reuse_verified_speech_without_asr_inference_options(self, task):
        generator = AudioWorkloadGenerator("model", task, 1)
        payload = generator.generate(1)
        assert (payload["sample_rate"]) == (16000)
        assert ("asr_task") not in (payload["params"])
        metadata = generator.plan_metadata()
        assert (metadata["pipeline_tag"]) == (task)
        assert (metadata["provenance"]["license"]) == ("CC-BY-4.0")
        assert (generator.input_metadata(1, payload)["input_num_samples"]) == (16000)

    def test_text_audio_preprocessing_uses_text_token_count_and_limit(self):
        context = {"task_type": "text-to-speech", "pipeline": SimpleNamespace(tokenizer=TextTokenizer())}
        output = self.handler.preprocess(context, {"text": "one two three four five", "params": {}})
        assert (output["text"]) == ("one two three four")
        assert (output["_effective_input_scale"]) == (4)
        assert (output["_truncated_by_limit"])
        metadata = self.handler.get_scale_metadata(context, {})
        assert (metadata["input_scale_type"]) == ("seq_length")
        assert (metadata["max_effective_input_scale"]) == (4)

    def test_bark_input_limit_uses_semantic_generation_window_without_special_tokens(self):
        tokenizer = TextTokenizer()
        tokenizer.model_max_length = 1024
        context = {
            "task_type": "text-to-speech",
            "pipeline": SimpleNamespace(
                tokenizer=tokenizer,
                model=SimpleNamespace(config=SimpleNamespace(model_type="bark")),
                generation_config=SimpleNamespace(semantic_config={"max_input_semantic_length": 256}),
            ),
        }
        metadata = self.handler.get_scale_metadata(context, {})
        assert (metadata["max_effective_input_scale"]) == (256)
        output = self.handler.preprocess(context, {"text": "word " * 300})
        assert (output["_effective_input_scale"]) == (256)
        assert (output["_truncated_by_limit"])

    def test_text_audio_rejects_tokenizer_overrides_that_invalidate_measured_scale(self):
        pipe = Mock()
        context = {"task_type": "text-to-speech", "pipeline": pipe}
        with pytest.raises(ValueError, match="preprocess_params.*input scale"):
            self.handler.predict(context, {"text": "test", "params": {"pipeline_kwargs": {"preprocess_params": {"max_length": 1}}}})
        pipe.assert_not_called()

    def test_text_audio_predict_passes_generation_controls_and_returns_waveform_metadata(self):
        pipe = Mock(return_value={"audio": np.zeros((1, 24000)), "sampling_rate": 24000})
        context = {"task_type": "text-to-audio", "pipeline": pipe}
        processed = {"text": "calm music", "params": {"pipeline_kwargs": {"generate_kwargs": {"max_new_tokens": 64}}}}
        result = self.handler.postprocess(context, self.handler.predict(context, processed))
        pipe.assert_called_once_with("calm music", generate_kwargs={"max_new_tokens": 64})
        assert (result["output_type"]) == ("audio")
        assert (result["audio_duration_s"]) == (1.0)
        assert (result["audio_num_samples"]) == (24000)
        assert (result["audio_shape"]) == ([1, 24000])
        assert ("audio") not in (result)

    @pytest.mark.parametrize('result_case', range(2), ids=["{'audio': np.array([np.nan]), 'sampling_rate': 24000}", "{'sampling_rate': 24000}"])
    def test_audio_output_rejects_non_finite_or_missing_waveform(self, result_case):
        context = {"task_type": "text-to-speech"}
        result = tuple(({'audio': np.array([np.nan]), 'sampling_rate': 24000}, {'sampling_rate': 24000}))[result_case]
        with pytest.raises(ValueError, match="audio"):
            self.handler.postprocess(context, result)

    def test_audio_classification_resamples_in_preprocess_and_preserves_source_duration(self):
        context = {"task_type": "audio-classification", "audio_metadata": {"sampling_rate": 32000}}
        signal = types.ModuleType("scipy.signal")
        signal.resample_poly = Mock(side_effect=lambda value, up, down: np.repeat(value, up))
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(np.full(160, 8192, dtype="<i2").tobytes())
        request = {"audio_base64": base64.b64encode(buffer.getvalue()).decode(),
                   "audio_format": "wav", "sample_rate": 16000}
        with patch.dict(sys.modules, {"scipy": types.ModuleType("scipy"), "scipy.signal": signal}):
            output = self.handler.preprocess(context, request)
        assert (output["sample_rate"]) == (32000)
        assert (len(output["audio"])) == (320)
        assert (output["_effective_input_scale"]) == (0.01)
        assert (output["_input_num_samples"]) == (160)
        assert (output["_model_input_num_samples"]) == (320)
        assert (output["_source_sample_rate"]) == (16000)

    def test_codec_load_uses_explicit_supported_architecture_and_pinned_revision(self):
        transformer = types.ModuleType("transformers")
        transformer.__version__ = "4.57.6"
        transformer.AutoConfig = SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(model_type="encodec", audio_channels=1)))
        model = Mock(config=SimpleNamespace(model_type="encodec", sampling_rate=24000, audio_channels=1))
        model.to.return_value = model
        model.eval.return_value = model
        transformer.EncodecModel = SimpleNamespace(from_pretrained=Mock(return_value=model))
        processor = SimpleNamespace(sampling_rate=24000)
        transformer.AutoProcessor = SimpleNamespace(from_pretrained=Mock(return_value=processor))
        transformer.pipeline = Mock(side_effect=AssertionError("audio-to-audio is not a Transformers pipeline"))
        torch = types.ModuleType("torch")
        torch.float32 = "float32"
        with patch.dict(sys.modules, {"transformers": transformer, "torch": torch}):
            context = self.handler.load("model", "audio-to-audio", "transformers_model", "cpu", "pinned")
        assert (context["model"]) is (model)
        assert (context["processor"]) is (processor)
        assert (context["audio_metadata"]["sampling_rate"]) == (24000)
        assert (transformer.EncodecModel.from_pretrained.call_args.kwargs["revision"]) == ("pinned")

    def test_codec_predict_returns_model_waveform(self):
        model = Mock(return_value=SimpleNamespace(audio_values=np.zeros((1, 1, 24))))
        context = {"task_type": "audio-to-audio", "model": model, "audio_metadata": {"sampling_rate": 24000}}
        processed = {"model_inputs": {"input_values": np.ones((1, 1, 24))}, "params": {"pipeline_kwargs": {"bandwidth": 6.0}}}
        torch = types.ModuleType("torch")
        torch.inference_mode = contextlib.nullcontext
        with patch.dict(sys.modules, {"torch": torch}):
            output = self.handler.predict(context, processed)
        result = self.handler.postprocess(context, output)
        assert (result["audio_num_samples"]) == (24)
        assert (result["audio_duration_s"]) == (0.001)
        assert (model.call_args.kwargs["bandwidth"]) == (6.0)

    def test_vad_load_uses_local_jit_and_rejects_cuda(self):
        torch = types.ModuleType("torch")
        model = Mock()
        model.eval.return_value = model
        torch.jit = SimpleNamespace(load=Mock(return_value=model))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "silero_vad.jit"
            path.write_bytes(b"fake-model")
            with patch.dict(sys.modules, {"torch": torch}):
                context = self.handler.load(directory, "voice-activity-detection", "torchscript", "cpu")
                assert (context["model"]) is (model)
                assert (self.handler.get_scale_metadata(context, {"sample_rate": 16000})["short_form_fixed_padding"]) is (False)
                torch.jit.load.assert_called_once_with(str(path), map_location="cpu")
                with pytest.raises(ValueError, match="CPU"):
                    self.handler.load(directory, "voice-activity-detection", "torchscript", "cuda")

    def test_vad_postprocessing_joins_adjacent_positive_frames_and_clips_padding(self):
        result = self.handler.postprocess(
            {"task_type": "voice-activity-detection"},
            {"probabilities": [0.1, 0.8, 0.9, 0.2, 0.8], "frame_samples": 512, "sample_rate": 16000, "input_num_samples": 2200, "threshold": 0.5},
        )
        assert (result["output_type"]) == ("voice_activity")
        assert (result["segments"]) == ([{"start": 0.032, "end": 0.096}, {"start": 0.128, "end": 0.1375}])
        assert (result["n_results"]) == (2)
        assert (result["speech_duration_s"]) == (0.0735) or round(abs((result["speech_duration_s"]) - (0.0735)), 7) == 0

    def test_vad_predict_resets_recurrent_state_for_each_request(self):
        model = Mock(side_effect=[0.8, 0.2, 0.8, 0.2])
        context = {"task_type": "voice-activity-detection", "model": model}
        frames = [np.zeros(512), np.zeros(512)]
        processed = {"frames": frames, "audio": np.zeros(900), "params": {}}
        torch = types.ModuleType("torch")
        torch.inference_mode = contextlib.nullcontext
        with patch.dict(sys.modules, {"torch": torch}):
            first = self.handler.predict(context, processed)
            second = self.handler.predict(context, processed)
        assert (first["probabilities"]) == (second["probabilities"])
        assert (model.reset_states.call_count) == (2)
        assert (first["input_num_samples"]) == (900)

    def test_vad_load_rejects_ambiguous_model_files(self):
        with tempfile.TemporaryDirectory() as directory:
            for subdir in ("v5", "v6"):
                folder = Path(directory) / subdir
                folder.mkdir()
                (folder / "silero_vad.jit").write_bytes(b"model")
            with pytest.raises(ValueError, match="exactly one.*found 2"):
                self.handler.load(directory, "voice-activity-detection", "torchscript", "cpu")
