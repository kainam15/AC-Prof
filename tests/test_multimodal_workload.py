import base64
import io
import json
import tempfile
import wave
from pathlib import Path

import pytest
from PIL import Image

from acprof.workloads import get_generator


class TestMultimodalWorkload:
    def generator(self, task, spec=None, batch=1):
        return get_generator("multimodal", "example/model", task, batch, workload_spec_path=spec)

    @pytest.mark.parametrize('task', ('image-text-to-text', 'visual-question-answering', 'document-question-answering', 'visual-document-retrieval'))
    def test_image_tasks_share_reproducible_scene_and_document_annotations(self, task):
        gen = self.generator(task)
        payload = gen.generate(224)
        assert (payload) == (self.generator(task).generate(224))
        sample = payload["samples"][0]
        image = Image.open(io.BytesIO(base64.b64decode(sample["image_base64"])))
        assert (image.size) == ((224, 224))
        assert (sample["text"])
        assert (payload["input_scale_type"]) == ("resolution_px")
        assert (gen.input_metadata(224, payload)["input_num_samples"]) == (1)
        if task == "document-question-answering":
            assert (len(sample["words"])) == (len(sample["boxes"]))
            assert ("42") in (sample["words"])

    def test_audio_is_prefix_of_same_real_waveform_with_provenance(self):
        gen = self.generator("audio-text-to-text")
        def read(payload):
            with wave.open(io.BytesIO(base64.b64decode(payload["samples"][0]["audio_base64"]))) as audio:
                assert (audio.getframerate()) == (16000)
                return audio.readframes(audio.getnframes())
        short, long = gen.generate(1), gen.generate(2)
        assert (len(read(short))) == (32000)
        assert (read(long)[:32000]) == (read(short))
        assert (short["input_scale_type"]) == ("duration_s")
        assert (gen.plan_metadata()["assets"][0]["sha256"]) == ("c67f163f3b1aa88157123dfa0264d80a34d7a5b51e4d3f2b6718b60bcca34157")
        with pytest.raises(ValueError, match="duration|时长"):
            gen.generate(31)

    def test_video_scales_frame_prefix_and_records_fps(self):
        gen = self.generator("video-text-to-text")
        small, large = gen.generate(2), gen.generate(4)
        assert (small["samples"][0]["video_frames_base64"]) == (large["samples"][0]["video_frames_base64"][:2])
        assert (len(large["samples"][0]["video_frames_base64"])) == (4)
        assert (large["input_scale_type"]) == ("frame_count")
        assert (gen.input_metadata(4, large)["video_num_frames"]) == (4)
        assert (gen.input_metadata(4, large)["video_frame_width"]) == (224)
        assert (gen.input_metadata(4, large)["video_frame_height"]) == (224)

    def test_custom_manifest_resolves_relative_assets_and_freezes_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (20, 30), "red").save(root / "page.png")
            spec = root / "workload.json"
            spec.write_text(json.dumps({"schema_version": 1, "task": "image-text-to-text", "image_path": "page.png", "text": "What color?", "input_scales": [64, 128], "params": {"max_new_tokens": 8}}))
            gen = self.generator("image-text-to-text", str(spec))
            first = gen.generate(64)
            Image.new("RGB", (20, 30), "blue").save(root / "page.png")
            assert (first) == (gen.generate(64))
            assert (first["samples"][0]["text"]) == ("What color?")
            assert (first["params"]["max_new_tokens"]) == (8)
            assert (gen.default_input_scales()) == ([64.0, 128.0])
            assert (len(gen.plan_metadata()["assets"][0]["sha256"])) == (64)

    @pytest.mark.parametrize('scale_case', range(5), ids=['0', '-1', "float('nan')", '1.5', 'True'])
    def test_invalid_batch_scale_and_unknown_manifest_keys_are_rejected(self, scale_case):
        with pytest.raises(ValueError, match="batch"):
            self.generator("image-text-to-text", batch=2)
        gen = self.generator("image-text-to-text")
        scale = tuple((0, -1, float('nan'), 1.5, True))[scale_case]
        with pytest.raises(ValueError):
            gen.generate(scale)
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp) / "invalid.json"
            spec.write_text('{"schema_version":1,"promtp":"typo"}')
            with pytest.raises(ValueError, match="promtp"):
                self.generator("image-text-to-text", str(spec))

    def test_any_to_any_can_mix_image_and_audio_with_one_scaling_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp) / "any.json"
            spec.write_text(json.dumps({"schema_version": 1, "modalities": ["image", "audio"], "scale_modality": "audio", "params": {"return_audio": True}}))
            gen = self.generator("any-to-any", str(spec))
            small, large = gen.generate(1), gen.generate(2)
            assert (small["samples"][0]["image_base64"]) == (large["samples"][0]["image_base64"])
            assert (small["samples"][0]["audio_base64"]) != (large["samples"][0]["audio_base64"])
            assert (small["params"]["return_audio"])
            assert (gen.plan_metadata()["scale_modality"]) == ("audio")

    @pytest.mark.parametrize('extra', ({'audio_path': 'unused.wav'}, {'video_frames': ['unused.png']}, {'image_resolution': 128}))
    def test_manifest_rejects_unused_assets_and_active_axis_fixed_values(self, extra):
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp) / "invalid.json"
            spec.write_text(json.dumps({"schema_version": 1, **extra}))
            with pytest.raises(ValueError):
                self.generator("image-text-to-text", str(spec))
        gen = self.generator("image-text-to-text")
        assert ("image_resolution") not in (gen.plan_metadata()["fixed_media"])

    def test_custom_video_frames_respect_fixed_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (20, 30), "red").save(root / "frame.png")
            spec = root / "video.json"
            spec.write_text(json.dumps({"schema_version": 1, "video_frames": ["frame.png"], "image_resolution": 64, "input_scales": [1]}))
            gen = self.generator("video-text-to-text", str(spec))
            frame = gen.generate(1)["samples"][0]["video_frames_base64"][0]
            assert (Image.open(io.BytesIO(base64.b64decode(frame))).size) == ((64, 64))
