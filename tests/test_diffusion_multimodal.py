import base64
import hashlib
import io
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from acprof.container.handlers.diffusion import DiffusionHandler
from acprof.workloads.diffusion import DiffusionWorkloadGenerator


class FakeGenerator:
    def __init__(self, *, device):
        self.device = device

    def manual_seed(self, seed):
        self.seed = seed
        return self


class ImagePipeline:
    vae_scale_factor = 8

    def __call__(self, prompt, image, num_inference_steps=20, guidance_scale=7.5,
                 generator=None, output_type="pil", return_dict=True, strength=0.8):
        self.kwargs = locals()
        return types.SimpleNamespace(images=[image.copy()])


class VideoPipeline:
    vae_scale_factor_spatial = 8
    vae_scale_factor_temporal = 4
    transformer = types.SimpleNamespace(config=types.SimpleNamespace(patch_size=2))

    def __call__(self, prompt, image, height, width, num_frames=17,
                 num_inference_steps=20, guidance_scale=7.5, generator=None,
                 output_type="pil", return_dict=True):
        self.kwargs = locals()
        return types.SimpleNamespace(frames=[[
            Image.new("RGB", (width, height)) for _ in range(num_frames)
        ]])


def test_image_condition_and_its_hash_are_reproducible_across_generators():
    first = DiffusionWorkloadGenerator("example/edit", "image-text-to-image", 1)
    second = DiffusionWorkloadGenerator("example/edit", "image-text-to-image", 1)
    payload = first.generate(128)
    assert (payload) == (second.generate(128))
    image_bytes = base64.b64decode(payload["image_base64"], validate=True)
    with Image.open(io.BytesIO(image_bytes)) as image:
        assert (image.size) == ((128, 128))
        assert (len(image.getcolors(maxcolors=128 * 128))) > (1)
    metadata = first.input_metadata(128, payload)
    assert (metadata["condition_image_sha256"]) == (hashlib.sha256(image_bytes).hexdigest())
    assert (metadata["input_num_samples"]) == (1)
    assert (metadata["input_scale_type"]) == ("resolution_px")
    assert (first.plan_metadata()["input_scale_type"]) == ("resolution_px")

def test_manifest_materializes_relative_image_prompt_steps_and_frames():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        Image.new("RGB", (200, 120), (200, 30, 10)).save(root / "condition.png")
        spec = {
            "schema_version": 1,
            "workload_id": "custom-video-v1",
            "image_path": "condition.png",
            "prompt": "pan slowly to the left",
            "input_scales": [128, 256],
            "params": {"num_inference_steps": 5, "num_frames": 9, "seed": 99},
        }
        manifest = root / "workload.json"
        manifest.write_text(json.dumps(spec))
        generator = DiffusionWorkloadGenerator(
            "example/video", "image-text-to-video", 1, workload_spec_path=str(manifest)
        )
        payload = generator.generate(128)
        assert (payload["prompt"]) == (["pan slowly to the left"])
        assert (payload["params"]["num_frames"]) == (9)
        assert (payload["params"]["seed"]) == (99)
        assert (generator.default_input_scales()) == ([128.0, 256.0])
        assert (generator.input_metadata(128, payload)["output_num_frames"]) == (9)
        assert (generator.plan_metadata()["workload_id"]) == ("custom-video-v1")
        assert (generator.plan_metadata()["workload_spec_sha256"]) == (hashlib.sha256(manifest.read_bytes()).hexdigest())

@pytest.mark.parametrize('spec_case', range(2), ids=["({'params': {'strenth': 0.7}}, 'strenth')", "({'image_path': 'condition.png', 'image_sha256': '0' * 64}, 'SHA256')"])
def test_manifest_rejects_unknown_params_and_image_hash_mismatch(spec_case):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        Image.new("RGB", (64, 64)).save(root / "condition.png")
        manifest = root / "workload.json"
        (spec, message) = tuple([({'params': {'strenth': 0.7}}, 'strenth'), ({'image_path': 'condition.png', 'image_sha256': '0' * 64}, 'SHA256')])[spec_case]
        manifest.write_text(json.dumps(spec))
        with pytest.raises(ValueError, match=message):
            DiffusionWorkloadGenerator("example/edit", "image-text-to-image", 1, workload_spec_path=str(manifest))

@pytest.mark.parametrize('task', ('image-to-image', 'image-to-video'))
def test_hub_aliases_generate_and_execute_conditioned_workloads(task):
    payload = DiffusionWorkloadGenerator("example/model", task, 1).generate(128)
    assert (payload["prompt_optional"])
    assert ("image_base64") in (payload)
    assert (len(payload["prompt"])) == (1)

@pytest.mark.parametrize('task', ['image-text-to-image', 'image-text-to-video'])
def test_conditioned_tasks_reject_batch_greater_than_one(task):
    with pytest.raises(ValueError, match="batch_size=1"):
        DiffusionWorkloadGenerator("example/model", task, 2)

@pytest.mark.parametrize('params', [{'num_frames': '9'}, {'num_inference_steps': 0}, {'guidance_scale': True}, {'strength': 1.1}, {'seed': 1.5}, {'negative_prompt': ['first', 'second']}])
def test_manifest_rejects_invalid_generation_values_before_materializing(params):
    with tempfile.TemporaryDirectory() as directory:
        manifest = Path(directory) / "workload.json"
        manifest.write_text(json.dumps({"params": params}))
        with pytest.raises(ValueError):
            DiffusionWorkloadGenerator(
                "example/video", "image-text-to-video", 1,
                workload_spec_path=str(manifest),
            )


class TestDiffusionMultimodalHandler:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.handler = DiffusionHandler()
        self.fake_torch = types.ModuleType("torch")
        self.fake_torch.Generator = FakeGenerator

    @staticmethod
    def payload(task="image-text-to-image", resolution=128):
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (100, 20, 30)).save(buf, format="PNG")
        params = {"seed": 27, "num_inference_steps": 10}
        if task == "image-text-to-video":
            params["num_frames"] = 9
        return {"prompt": ["make the scene brighter"], "resolution": resolution,
                "image_base64": base64.b64encode(buf.getvalue()).decode("ascii"), "params": params}

    def run_pipeline(self, pipeline, task="image-text-to-image", payload=None):
        context = {"task_type": task, "pipeline": pipeline, "device": "cpu"}
        processed = self.handler.preprocess(context, payload or self.payload(task))
        with patch.dict(sys.modules, {"torch": self.fake_torch}):
            output = self.handler.predict(context, processed)
        return self.handler.postprocess(context, output)

    def test_image_prompt_seed_strength_and_resolution_reach_native_pipeline(self):
        pipe = ImagePipeline()
        payload = self.payload()
        payload["params"]["strength"] = 0.6
        response = self.run_pipeline(pipe, payload=payload)
        assert (pipe.kwargs["prompt"]) == (["make the scene brighter"])
        assert (pipe.kwargs["image"].size) == ((128, 128))
        assert (pipe.kwargs["generator"].seed) == (27)
        assert (pipe.kwargs["strength"]) == (0.6)
        assert (response["output_shape"]) == ([1, 128, 128, 3])
        assert ("images") not in (response)

    def test_video_metadata_counts_decoded_frames_without_encoding(self):
        pipe = VideoPipeline()
        payload = self.payload("image-text-to-video")
        with patch.object(Image.Image, "save", side_effect=AssertionError("must not encode output")):
            response = self.run_pipeline(pipe, "image-text-to-video", payload)
        assert (pipe.kwargs["num_frames"]) == (9)
        assert (response["output_type"]) == ("video")
        assert (response["n_results"]) == (1)
        assert (response["output_length"]) == (9)
        assert (response["video_frame_count"]) == (9)
        assert (response["output_shape"]) == ([1, 9, 128, 128, 3])
        assert ("frames") not in (response)

    @pytest.mark.parametrize('value', [None, '', 'not base64!'])
    def test_missing_or_invalid_condition_is_rejected_instead_of_generated(self, value):
        context = {"task_type": "image-text-to-image", "pipeline": ImagePipeline()}
        payload = self.payload()
        payload["image_base64"] = value
        with pytest.raises(ValueError, match="image_base64"):
            self.handler.preprocess(context, payload)

    def test_pipeline_cannot_swallow_missing_image_support_through_kwargs(self):
        def text_only(prompt, num_inference_steps=20, guidance_scale=7.5,
                      generator=None, output_type="pil", return_dict=True, **kwargs):
            raise AssertionError("unsupported pipeline must not execute")
        with pytest.raises(ValueError, match="image"):
            self.run_pipeline(text_only)

    def test_unsupported_explicit_parameter_is_rejected_even_with_kwargs(self):
        def no_strength(prompt, image, num_inference_steps=20, guidance_scale=7.5,
                        generator=None, output_type="pil", return_dict=True, **kwargs):
            raise AssertionError("unsupported strength must not be ignored")
        payload = self.payload()
        payload["params"]["strength"] = 0.5
        with pytest.raises(ValueError, match="strength"):
            self.run_pipeline(no_strength, payload=payload)

    @pytest.mark.parametrize('resolution,frames,message', [(72, 9, '16'), (128, 8, 'num_frames')])
    def test_video_enforces_model_spatial_and_temporal_alignment(self, resolution, frames, message):
        context = {"task_type": "image-text-to-video", "pipeline": VideoPipeline()}
        payload = self.payload("image-text-to-video", resolution)
        payload["params"]["num_frames"] = frames
        with pytest.raises(ValueError, match=message):
            self.handler.preprocess(context, payload)

    def test_pipeline_output_cannot_silently_change_requested_frames_or_size(self):
        class WrongOutput(VideoPipeline):
            def __call__(self, prompt, image, height, width, num_frames=17,
                         num_inference_steps=20, guidance_scale=7.5, generator=None,
                         output_type="pil", return_dict=True):
                return types.SimpleNamespace(frames=[[Image.new("RGB", (64, 64))]])
        with pytest.raises(ValueError, match="requested|expected"):
            self.run_pipeline(WrongOutput(), "image-text-to-video")

    def test_prediction_does_not_run_postprocess_inside_profiler_window(self):
        context = {"task_type": "image-text-to-video", "pipeline": VideoPipeline(), "device": "cpu"}
        processed = self.handler.preprocess(context, self.payload("image-text-to-video"))
        with patch.dict(sys.modules, {"torch": self.fake_torch}), patch.object(
            self.handler, "postprocess", side_effect=AssertionError("metadata belongs in postprocess")
        ):
            output = self.handler.predict(context, processed)
        assert (self.handler.postprocess(context, output)["video_frame_count"]) == (9)

    def test_strength_that_produces_no_denoising_steps_is_rejected(self):
        payload = self.payload()
        payload["params"]["strength"] = 0.01
        with pytest.raises(ValueError, match="strength.*num_inference_steps|denoising"):
            self.run_pipeline(ImagePipeline(), payload=payload)

    def test_handler_rejects_batched_conditioned_request(self):
        payload = self.payload()
        payload["prompt"] = ["first", "second"]
        with pytest.raises(ValueError, match="one prompt|batch_size=1"):
            self.run_pipeline(ImagePipeline(), payload=payload)

    def test_load_rejects_image_only_video_pipeline_and_preserves_revision(self):
        calls = []
        class ImageOnly:
            def __call__(self, image, num_frames=17):
                pass
        class Loader:
            @classmethod
            def from_pretrained(cls, model_source, **kwargs):
                calls.append(kwargs)
                return ImageOnly()
        fake_diffusers = types.ModuleType("diffusers")
        fake_diffusers.DiffusionPipeline = Loader
        self.fake_torch.float16 = "float16"
        self.fake_torch.float32 = "float32"
        with patch.dict(sys.modules, {"torch": self.fake_torch, "diffusers": fake_diffusers}):
            with pytest.raises(ValueError, match="prompt"):
                self.handler.load("example/video", "image-text-to-video", "diffusers", "cpu", "revision123")
        assert (calls[0]["revision"]) == ("revision123")
        assert (calls[0]["local_files_only"])

    def test_native_video_load_keeps_transformer_and_refuses_unverified_eager_flops(self):
        class LoadableVideo(VideoPipeline):
            def to(self, device):
                self.device = device
                return self

            def set_progress_bar_config(self, *, disable):
                self.progress_disabled = disable

        pipe = LoadableVideo()
        calls = []

        class Loader:
            @classmethod
            def from_pretrained(cls, model_source, **kwargs):
                calls.append(kwargs)
                return pipe

        fake_diffusers = types.ModuleType("diffusers")
        fake_diffusers.DiffusionPipeline = Loader
        self.fake_torch.float16 = "float16"
        self.fake_torch.float32 = "float32"
        with tempfile.TemporaryDirectory() as snapshot, patch.dict(
            sys.modules, {"torch": self.fake_torch, "diffusers": fake_diffusers}
        ):
            context = self.handler.load(snapshot, "image-to-video", "diffusers", "cpu", "revision123")
            with pytest.raises(ValueError, match="eager UNet"):
                self.handler.load(
                    snapshot, "image-to-video", "diffusers", "cpu", "revision123",
                    load_options={"attention_implementation": "eager"},
                )
        assert (context["model"]) is (pipe.transformer)
        assert (context["task_type"]) == ("image-to-video")
        assert (pipe.device) == ("cpu")
        assert (pipe.progress_disabled)
        assert ("revision") not in (calls[0])
        assert (calls[0]["local_files_only"])
