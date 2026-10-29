import base64
import io
import sys
import types
import wave
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from acprof.container.handlers.audio import AudioHandler


def wav_base64(
    samples,
    *,
    sample_rate=16000,
    channels=1,
    sample_width=2,
):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        if sample_width == 2:
            payload = np.asarray(samples, dtype="<i2").tobytes()
        elif sample_width == 1:
            payload = np.asarray(samples, dtype=np.uint8).tobytes()
        else:
            raise AssertionError("test helper only supports 8-bit and 16-bit WAV")
        wav_file.writeframes(payload)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class RecordingPipeline:
    def __init__(self, *, model_type="whisper", tokenizer=None):
        self.calls = []
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                model_type=model_type,
                num_mel_bins=128,
                max_source_positions=1500,
                max_target_positions=448,
            )
        )
        self.feature_extractor = SimpleNamespace(
            sampling_rate=16000,
            chunk_length=30,
            n_samples=480000,
            nb_max_frames=3000,
            hop_length=160,
        )
        self.tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"text": "hello world"}


class TestAudioHandler:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.handler = AudioHandler()

    def context(self, pipe=None, task_type="automatic-speech-recognition"):
        pipe = pipe or RecordingPipeline()
        return {
            "pipeline": pipe,
            "task_type": task_type,
            "audio_metadata": self.handler._extract_audio_metadata(pipe),
        }

    def valid_request(self, sample_count=16000, **overrides):
        request = {
            "audio_base64": wav_base64(np.arange(sample_count) % 100),
            "audio_format": "wav",
            "sample_rate": 16000,
            "params": {},
        }
        request.update(overrides)
        return request

    def test_load_extracts_whisper_audio_limits(self):
        pipe = RecordingPipeline()
        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_torch.float32 = "float32"
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.__version__ = "4.57.6"
        fake_transformers.pipeline = lambda **kwargs: pipe

        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            context = self.handler.load(
                "openai/whisper-large-v3",
                "automatic-speech-recognition",
                "transformers_pipeline",
                "cpu",
            )

        assert (context["audio_metadata"]) == ({
                "sampling_rate": 16000,
                "max_short_form_duration_s": 30.0,
                "model_input_num_samples": 480000,
                "model_input_frames": 3000,
                "short_form_fixed_padding": True,
                "fixed_frontend_num_samples": 480000,
                "fixed_frontend_num_frames": 3000,
                "frontend_feature_bins": 128,
                "encoder_positions": 1500,
                "decoder_output_token_limit": 448,
                "model_type": "whisper",
            })

    def test_scale_metadata_distinguishes_audio_and_decoder_limits(self):
        metadata = self.handler.get_scale_metadata(self.context(), {})

        assert (metadata["input_scale_type"]) == ("duration_s")
        assert (metadata["required_sampling_rate"]) == (16000)
        assert (metadata["max_short_form_duration_s"]) == (30.0)
        assert (metadata["max_effective_input_scale"]) == (30.0)
        assert (metadata["model_input_num_samples"]) == (480000)
        assert (metadata["model_input_frames"]) == (3000)
        assert (metadata["short_form_fixed_padding"])
        assert (metadata["fixed_frontend_num_samples"]) == (480000)
        assert (metadata["fixed_frontend_num_frames"]) == (3000)
        assert (metadata["frontend_feature_bins"]) == (128)
        assert (metadata["encoder_positions"]) == (1500)
        assert (metadata["decoder_output_token_limit"]) == (448)
        assert (metadata["model_type"]) == ("whisper")
        assert ("output-token limit") in (metadata["reason"])
        assert ("not an audio input-length limit") in (metadata["reason"])
        assert ("pads every accepted short-form") in (metadata["reason"])

    def test_preprocess_decodes_pcm16_mono_wav_and_reports_scale(self):
        samples = np.array([-32768, -1, 0, 16384, 32767], dtype=np.int16)
        processed = self.handler.preprocess(
            self.context(),
            {
                "audio_base64": wav_base64(samples),
                "audio_format": "wav",
                "sample_rate": 16000,
                "params": {"mode": "short_form"},
            },
        )

        np.testing.assert_allclose(
            processed["audio"],
            samples.astype(np.float32) / 32768.0,
        )
        assert (processed["sample_rate"]) == (16000)
        assert (processed["_input_num_samples"]) == (5)
        assert (processed["_duration_s"]) == (5 / 16000)
        assert (processed["_effective_input_scale"]) == (5 / 16000)
        assert not (processed["_truncated_by_limit"])
        assert ("within") in (processed["_probe_reason"])

    def test_preprocess_rejects_legacy_float_samples(self):
        with pytest.raises(ValueError, match="audio_base64"):
            self.handler.preprocess(self.context(), {"audio_samples": [0.0, 0.25], "sample_rate": 16000})

    def test_preprocess_rejects_missing_empty_and_ambiguous_audio(self):
        with pytest.raises(ValueError, match="audio_base64"):
            self.handler.preprocess(self.context(), {"params": {}})
        with pytest.raises(ValueError, match="audio_base64"):
            self.handler.preprocess(
                self.context(), {"audio_samples": [], "sample_rate": 16000}
            )
        with pytest.raises(ValueError, match="audio_base64"):
            self.handler.preprocess(
                self.context(),
                {
                    "audio_base64": self.valid_request()["audio_base64"],
                    "audio_samples": [0.0],
                    "audio_format": "wav",
                    "sample_rate": 16000,
                },
            )

    @pytest.mark.parametrize('request_case', range(7))
    def test_preprocess_strictly_validates_base64_wav_contract(self, request_case):
        cases = [
            (
                {"audio_base64": "%%%", "audio_format": "wav", "sample_rate": 16000},
                "not valid Base64",
            ),
            (
                {
                    "audio_base64": self.valid_request()["audio_base64"],
                    "audio_format": "flac",
                    "sample_rate": 16000,
                },
                "audio_format must be 'wav'",
            ),
            (
                {
                    "audio_base64": self.valid_request()["audio_base64"],
                    "audio_format": "WAV",
                    "sample_rate": 16000,
                },
                "audio_format must be 'wav'",
            ),
            (
                {
                    "audio_base64": self.valid_request()["audio_base64"],
                    "audio_format": "wav",
                },
                "sample_rate is required",
            ),
            (
                {
                    "audio_base64": wav_base64([0, 1], sample_rate=8000),
                    "audio_format": "wav",
                    "sample_rate": 16000,
                },
                "does not match the WAV header",
            ),
            (
                {
                    "audio_base64": wav_base64([0, 1, 2, 3], channels=2),
                    "audio_format": "wav",
                    "sample_rate": 16000,
                },
                "must be mono",
            ),
            (
                {
                    "audio_base64": wav_base64([0, 1], sample_width=1),
                    "audio_format": "wav",
                    "sample_rate": 16000,
                },
                "signed 16-bit PCM",
            ),
        ]
        (request, message) = tuple(cases)[request_case]
        with pytest.raises(ValueError, match=message):
            self.handler.preprocess(self.context(), request)

    def test_preprocess_rejects_model_sample_rate_mismatch(self):
        with pytest.raises(ValueError, match="model feature extractor"):
            self.handler.preprocess(
                self.context(),
                {
                    "audio_base64": wav_base64([0, 1], sample_rate=8000),
                    "audio_format": "wav",
                    "sample_rate": 8000,
                },
            )

    def test_probe_diagnoses_over_limit_without_truncating(self):
        processed = self.handler.preprocess(
            self.context(),
            {
                "audio_base64": wav_base64(np.zeros(480001, dtype=np.int16)),
                "audio_format": "wav",
                "sample_rate": 16000,
            },
        )

        assert (processed["_input_num_samples"]) == (480001)
        assert (processed["_truncated_by_limit"])
        assert ("exceeds") in (processed["_probe_reason"])
        assert (processed["audio"].size) == (480001)

    def test_predict_maps_whisper_semantics_and_pipeline_kwargs(self):
        pipe = RecordingPipeline()
        context = self.context(pipe)
        processed = self.handler.preprocess(
            context,
            self.valid_request(
                16000,
                params={
                    "mode": "short_form",
                    "asr_task": "transcribe",
                    "language": "en",
                    "return_timestamps": False,
                    "pipeline_kwargs": {
                        "batch_size": 1,
                        "generate_kwargs": {"max_new_tokens": 64},
                    },
                },
            ),
        )

        self.handler.predict(context, processed)

        args, kwargs = pipe.calls[0]
        assert (args[0]["sampling_rate"]) == (16000)
        np.testing.assert_array_equal(args[0]["raw"], processed["audio"])
        assert (kwargs["batch_size"]) == (1)
        assert not (kwargs["return_timestamps"])
        assert (kwargs["generate_kwargs"]) == ({"max_new_tokens": 64, "task": "transcribe", "language": "en"})

    def test_predict_rejects_translation_and_timestamp_modes(self):
        context = self.context()
        processed = {
            "audio": np.zeros(16000, dtype=np.float32),
            "sample_rate": 16000,
            "_duration_s": 1.0,
            "params": {"asr_task": "translate"},
        }
        with pytest.raises(ValueError, match="translation must be profiled"):
            self.handler.predict(context, processed)

        processed["params"] = {"return_timestamps": True}
        with pytest.raises(ValueError, match="requires return_timestamps=false"):
            self.handler.predict(context, processed)

        processed["params"] = {
            "pipeline_kwargs": {"stride_length_s": 2},
        }
        with pytest.raises(ValueError, match="chunked long-form setting"):
            self.handler.predict(context, processed)

    def test_predict_rejects_whisper_over_30_seconds(self):
        context = self.context()
        processed = {
            "audio": np.zeros(1, dtype=np.float32),
            "sample_rate": 16000,
            "_duration_s": 30.0001,
            "params": {},
        }

        with pytest.raises(ValueError, match="limited to 30s"):
            self.handler.predict(context, processed)

    def test_non_whisper_pipeline_does_not_receive_language_or_task(self):
        pipe = RecordingPipeline(model_type="wav2vec2")
        context = self.context(pipe)
        processed = {
            "audio": np.zeros(16000, dtype=np.float32),
            "sample_rate": 16000,
            "_duration_s": 1.0,
            "params": {
                "asr_task": "transcribe",
                "language": "en",
                "pipeline_kwargs": {"top_k": 3},
            },
        }

        self.handler.predict(context, processed)

        _, kwargs = pipe.calls[0]
        assert (kwargs) == ({"top_k": 3})

    def test_postprocess_returns_asr_text_character_and_token_counts(self):
        tokenizer = SimpleNamespace(
            encode=lambda text, add_special_tokens: [10, 11, 12]
        )
        context = self.context(RecordingPipeline(tokenizer=tokenizer))

        result = self.handler.postprocess(context, {"text": "hello world"})

        assert (result["output_type"]) == ("transcription")
        assert (result["text"]) == ("hello world")
        assert (result["output_length"]) == (11)
        assert (result["output_token_count"]) == (3)

    def test_postprocess_keeps_audio_classification_compatible(self):
        context = self.context(
            RecordingPipeline(model_type="wav2vec2"),
            task_type="audio-classification",
        )

        list_result = self.handler.postprocess(context, [{"label": "speech"}])
        dict_result = self.handler.postprocess(context, {"label": "speech"})

        assert (list_result["output_type"]) == ("classification")
        assert (list_result["n_results"]) == (1)
        assert (dict_result["output_type"]) == ("classification")
        assert (dict_result["n_results"]) == (1)
