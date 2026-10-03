import base64
import copy
import hashlib
import io
import json
import wave
from pathlib import Path

import pytest

from acprof.workloads import WorkloadGenerator, get_generator
from acprof.workloads.audio import AudioWorkloadGenerator

SAMPLE_RATE = 16000
NUM_SAMPLES = 30 * SAMPLE_RATE


def _write_wav(
    path: Path,
    *,
    num_samples: int = NUM_SAMPLES,
    sample_rate: int = SAMPLE_RATE,
    channels: int = 1,
    sample_width: int = 2,
) -> None:
    sample = b"\x01\x00" if sample_width == 2 else b"\x80"
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(sample * num_samples * channels)


def _read_wav_payload(payload):
    wav_bytes = base64.b64decode(payload["audio_base64"], validate=True)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        params = wav_file.getparams()
        pcm_bytes = wav_file.readframes(wav_file.getnframes())
    return wav_bytes, params, pcm_bytes


def _build_spec(
    asset_path: Path,
    *,
    pipeline_tag="automatic-speech-recognition",
    input_scales=None,
):
    with wave.open(str(asset_path), "rb") as wav_file:
        num_samples = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
    return {
        "schema_version": 1,
        "workload_id": "test-real-speech-short-v1",
        "task_family": "audio",
        "pipeline_tag": pipeline_tag,
        "asset": {
            "path": asset_path.name,
            "sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest(),
            "format": "wav",
            "pcm_subtype": "PCM_16",
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "num_samples": num_samples,
            "duration_s": num_samples / sample_rate,
        },
        "input_scale": {
            "type": "duration_s",
            "values": input_scales or [1, 2, 5, 10, 20, 30],
            "construction": "prefix",
        },
        "inference": {
            "mode": "short_form",
            "asr_task": "transcribe",
            "language": "en",
            "return_timestamps": False,
            "pipeline_kwargs": {"chunk_length_s": None},
        },
        "provenance": {
            "dataset_id": "test/dataset",
            "dataset_revision": "revision-sha",
            "config": "clean",
            "split": "test",
            "dataset_url": "https://example.test/dataset",
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "sources": [
                {
                    "id": "speaker-chapter-0001",
                    "sha256": "1" * 64,
                    "text": "real test speech transcript",
                }
            ],
            "transform": {
                "operation": "concatenate_then_prefix_crop",
                "source_order": ["speaker-chapter-0001"],
                "crop_start_sample": 0,
                "output_num_samples": num_samples,
                "output_duration_s": num_samples / sample_rate,
                "gain_applied": False,
                "normalized": False,
                "resampled": False,
            },
        },
    }


class _MinimalGenerator(WorkloadGenerator):
    def generate(self, scale_value):
        return {"scale": scale_value}

    def scale_label(self, scale_value):
        return str(scale_value)


class TestAudioWorkload:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.temp_dir = tmp_path
        self.root = Path(str(self.temp_dir))
        self.asset_path = self.root / "audio.wav"
        _write_wav(self.asset_path)
        self.spec = _build_spec(self.asset_path)
        self.spec_path = self.root / "source.json"
        self._write_spec(self.spec)

    def _write_spec(self, spec, path=None):
        target = path or self.spec_path
        target.write_text(json.dumps(spec), encoding="utf-8")
        return target

    def _generator(self, spec_path=None, task_type="automatic-speech-recognition"):
        return AudioWorkloadGenerator(
            "openai/whisper-large-v3",
            task_type,
            1,
            workload_spec_path=str(spec_path or self.spec_path),
        )

    def test_base_generator_metadata_extensions_preserve_legacy_defaults(self):
        generator = _MinimalGenerator("model", "task", 1)
        assert (generator.default_input_scales()) is None
        assert (generator.plan_metadata()) == ({})
        assert (generator.input_metadata(1, {"scale": 1})) == ({})

    def test_registry_accepts_optional_workload_spec_path(self):
        generator = get_generator(
            "audio",
            "openai/whisper-large-v3",
            "automatic-speech-recognition",
            1,
            workload_spec_path=str(self.spec_path),
        )
        assert isinstance(generator, AudioWorkloadGenerator)
        assert (generator.workload_spec_path) == (self.spec_path.resolve())

    def test_manifest_defaults_and_plan_metadata(self):
        generator = self._generator()
        assert (generator.default_input_scales()) == ([1, 2, 5, 10, 20, 30])
        assert (generator.max_input_scale()) == (30.0)
        metadata = generator.plan_metadata()
        assert (metadata["schema_version"]) == (1)
        assert (metadata["workload_id"]) == ("test-real-speech-short-v1")
        assert (metadata["asset"]["sha256"]) == (self.spec["asset"]["sha256"])
        assert (metadata["provenance"]) == (self.spec["provenance"])
        assert (metadata["inference"]) == (self.spec["inference"])
        assert (metadata["workload_constraints"]["max_short_form_duration_s"]) == (30.0)
        json.dumps(metadata)

    def test_builtin_manifest_keeps_pinned_librispeech_provenance(self):
        generator = AudioWorkloadGenerator(
            "openai/whisper-large-v3",
            "automatic-speech-recognition",
            1,
        )
        provenance = generator.plan_metadata()["provenance"]

        assert (provenance["dataset_id"]) == ("openslr/librispeech_asr")
        assert (provenance["dataset_revision"]) == ("71cacbfb7e2354c4226d01e70d77d5fca3d04ba1")
        assert ([source["id"] for source in provenance["sources"]]) == (["6930-75918-0001", "6930-75918-0002", "6930-75918-0003"])
        assert not (provenance["transform"]["resampled"])
        assert not (provenance["transform"]["normalized"])

    def test_builtin_manifest_records_project_relative_spec_path(self):
        generator = AudioWorkloadGenerator(
            "openai/whisper-large-v3",
            "automatic-speech-recognition",
            1,
        )

        workload_spec_path = generator.plan_metadata()["workload_spec_path"]

        assert (workload_spec_path) == ("assets/audio/librispeech-clean-test-en-30s/source.json")
        assert not (Path(workload_spec_path).is_absolute())

    def test_custom_manifest_allows_arbitrary_auditable_provenance(self):
        custom = copy.deepcopy(self.spec)
        custom["workload_id"] = "custom-single-source-v1"
        custom["provenance"] = {
            "license": "CC0-1.0",
            "sources": [{"uri": "recorder://session-42"}],
            "transform": {
                "operation": "resample_to_contract",
                "resampled": True,
                "gain_db": -1.5,
            },
        }
        self._write_spec(custom)

        generator = self._generator()

        assert (generator.plan_metadata()["provenance"]["transform"]["resampled"])
        assert (generator.default_input_scales()) == ([1, 2, 5, 10, 20, 30])

    def test_generate_returns_pcm16_wav_prefix_and_inference_params(self):
        generator = self._generator()
        payload = generator.generate(1)
        wav_bytes, params, pcm_bytes = _read_wav_payload(payload)

        assert (payload["audio_format"]) == ("wav")
        assert (payload["sample_rate"]) == (SAMPLE_RATE)
        assert (payload["params"]) == ({
                "mode": "short_form",
                "asr_task": "transcribe",
                "language": "en",
                "return_timestamps": False,
                "pipeline_kwargs": {"chunk_length_s": None},
            })
        assert (params.nchannels) == (1)
        assert (params.sampwidth) == (2)
        assert (params.framerate) == (SAMPLE_RATE)
        assert (params.nframes) == (SAMPLE_RATE)

        with wave.open(str(self.asset_path), "rb") as source:
            expected_prefix = source.readframes(SAMPLE_RATE)
        assert (pcm_bytes) == (expected_prefix)

        metadata = generator.input_metadata(1, payload)
        assert (metadata) == ({
                "duration_s": 1.0,
                "input_num_samples": SAMPLE_RATE,
                "audio_wav_bytes": len(wav_bytes),
                "audio_sha256": hashlib.sha256(wav_bytes).hexdigest(),
            })

    def test_every_scale_is_a_prefix_of_the_same_audio(self):
        generator = self._generator()
        _, _, one_second = _read_wav_payload(generator.generate(1))
        _, _, two_seconds = _read_wav_payload(generator.generate(2))
        assert (two_seconds[:len(one_second)]) == (one_second)

    def test_effective_scale_comes_from_materialized_sample_count(self):
        generator = self._generator()
        requested = 1.00001
        payload = generator.generate(requested)
        metadata = generator.input_metadata(requested, payload)
        assert (metadata["input_num_samples"]) == (SAMPLE_RATE)
        assert (generator.effective_input_scale(requested, payload)) == (1.0)
        assert (generator.effective_input_scale(99, payload)) == (1.0)

    @pytest.mark.parametrize('value_case', range(6), ids=['0', '-1', '30.1', "float('inf')", "float('nan')", 'True'])
    def test_rejects_invalid_manual_scales(self, value_case):
        generator = self._generator()
        value = tuple((0, -1, 30.1, float('inf'), float('nan'), True))[value_case]
        with pytest.raises(ValueError):
            generator.generate(value)

    def test_rejects_batching_and_unsupported_audio_tasks(self):
        with pytest.raises(ValueError, match="batch_size=1"):
            AudioWorkloadGenerator(
                "openai/whisper-large-v3",
                "automatic-speech-recognition",
                2,
                workload_spec_path=str(self.spec_path),
            )
        with pytest.raises(ValueError, match="unsupported audio workload"):
            AudioWorkloadGenerator("model", "unsupported-audio-task", 1)

    def test_rejects_pipeline_mismatch(self):
        with pytest.raises(ValueError, match="does not match"):
            self._generator(task_type="audio-classification")

    def test_rejects_asset_hash_mismatch(self):
        invalid = copy.deepcopy(self.spec)
        invalid["asset"]["sha256"] = "0" * 64
        self._write_spec(invalid)
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            self._generator()

    @pytest.mark.parametrize('filename_case', range(4))
    def test_rejects_invalid_wav_properties_and_short_asset(self, filename_case):
        cases = [
            ("stereo.wav", NUM_SAMPLES, SAMPLE_RATE, 2, 2, "must be mono"),
            ("8khz.wav", NUM_SAMPLES, 8000, 1, 2, "16000 Hz"),
            ("pcm8.wav", NUM_SAMPLES, SAMPLE_RATE, 1, 1, "16-bit PCM"),
            (
                "short.wav",
                NUM_SAMPLES - 1,
                SAMPLE_RATE,
                1,
                2,
                "short-form/asset limit",
            ),
        ]
        (filename, frames, rate, channels, width, message) = tuple(cases)[filename_case]
        asset = self.root / filename
        _write_wav(
            asset,
            num_samples=frames,
            sample_rate=rate,
            channels=channels,
            sample_width=width,
        )
        spec = _build_spec(asset)
        spec_path = self.root / f"{filename}.json"
        self._write_spec(spec, spec_path)
        with pytest.raises(ValueError, match=message):
            self._generator(spec_path)

    def test_shorter_asset_is_valid_when_all_manifest_scales_fit(self):
        asset = self.root / "ten-seconds.wav"
        _write_wav(asset, num_samples=10 * SAMPLE_RATE)
        spec = _build_spec(asset, input_scales=[1, 5, 10])
        spec_path = self.root / "ten-seconds.json"
        self._write_spec(spec, spec_path)

        generator = self._generator(spec_path)

        assert (generator.default_input_scales()) == ([1.0, 5.0, 10.0])
        assert (generator.max_input_scale()) == (10.0)
        with pytest.raises(ValueError, match="short-form/asset limit"):
            generator.generate(10.1)

    def test_manifest_rejects_unsupported_inference_modes_early(self):
        translated = copy.deepcopy(self.spec)
        translated["inference"]["asr_task"] = "translate"
        self._write_spec(translated)
        with pytest.raises(ValueError, match="only supports.*transcribe"):
            self._generator()

        timestamps = copy.deepcopy(self.spec)
        timestamps["inference"]["return_timestamps"] = True
        self._write_spec(timestamps)
        with pytest.raises(ValueError, match="return_timestamps=false"):
            self._generator()

        chunked = copy.deepcopy(self.spec)
        chunked["inference"]["pipeline_kwargs"] = {"chunk_length_s": 10}
        self._write_spec(chunked)
        with pytest.raises(ValueError, match="separate long-form workload"):
            self._generator()

    def test_rejects_asset_path_escape_and_unknown_schema_fields(self):
        nested = self.root / "nested"
        nested.mkdir()
        escaped = copy.deepcopy(self.spec)
        escaped["asset"]["path"] = "../audio.wav"
        nested_spec = nested / "source.json"
        self._write_spec(escaped, nested_spec)
        with pytest.raises(ValueError, match="must stay within"):
            self._generator(nested_spec)

        invalid = copy.deepcopy(self.spec)
        invalid["input_scale"]["typo"] = True
        self._write_spec(invalid)
        with pytest.raises(ValueError, match="unknown keys: typo"):
            self._generator()

    def test_input_metadata_rejects_malformed_payload(self):
        generator = self._generator()
        payload = generator.generate(1)
        payload["audio_base64"] = "not base64!"
        with pytest.raises(ValueError, match="invalid base64"):
            generator.input_metadata(1, payload)
