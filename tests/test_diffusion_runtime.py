"""Offline CPU checks using real Diffusers; no pretrained weights are downloaded."""
import functools
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from acprof.container.handlers.diffusion import (
    DiffusionHandler,
    _conditioned_pipeline_parameters,
    _native_pipeline_parameters,
)
from acprof.workloads.diffusion import DiffusionWorkloadGenerator

_RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "diffusers", "transformers")
)
pytestmark = pytest.mark.runtime


@pytest.mark.skipif(not (_RUNTIME_AVAILABLE), reason="requires the diffusion image runtime")
class TestDiffusionRuntime:
    @pytest.fixture(scope="class", autouse=True)
    def _class_setup(self, request):
        import torch
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        yield
        torch.set_num_threads(previous)

    @pytest.mark.parametrize('pipeline_class_case', range(4), ids=["(StableDiffusionImg2ImgPipeline, 'image-text-to-image')", "(StableDiffusionInstructPix2PixPipeline, 'image-text-to-image')", "(CogVideoXImageToVideoPipeline, 'image-text-to-video')", "(WanImageToVideoPipeline, 'image-text-to-video')"])
    def test_native_image_and_video_signatures_preserve_both_conditions(self, pipeline_class_case):
        from diffusers import (
            CogVideoXImageToVideoPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInstructPix2PixPipeline,
            WanImageToVideoPipeline,
        )
        (pipeline_class, task) = tuple([(StableDiffusionImg2ImgPipeline, 'image-text-to-image'), (StableDiffusionInstructPix2PixPipeline, 'image-text-to-image'), (CogVideoXImageToVideoPipeline, 'image-text-to-video'), (WanImageToVideoPipeline, 'image-text-to-video')])[pipeline_class_case]
        call = functools.partial(pipeline_class.__call__, None)
        parameters = _conditioned_pipeline_parameters(call, task)
        assert ("image") in (parameters)
        assert ("prompt") in (parameters)
        if task == "image-text-to-video":
            assert ("num_frames") in (parameters)
            assert ("height") in (parameters)
            assert ("width") in (parameters)

    @staticmethod
    def _tiny_pipeline(directory, instruct):
        import torch
        from diffusers import (
            AutoencoderKL,
            DDIMScheduler,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInstructPix2PixPipeline,
            UNet2DConditionModel,
        )
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

        torch.manual_seed(123)
        vocab_path = Path(directory) / "vocab.json"
        merges_path = Path(directory) / "merges.txt"
        vocab_path.write_text(json.dumps({
            "<|startoftext|>": 0, "<|endoftext|>": 1, "a</w>": 2,
        }))
        merges_path.write_text("#version: 0.2\n")
        tokenizer = CLIPTokenizer(
            vocab_file=str(vocab_path), merges_file=str(merges_path),
            model_max_length=16,
        )
        text_encoder = CLIPTextModel(CLIPTextConfig(
            vocab_size=3, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2,
            max_position_embeddings=16, bos_token_id=0, eos_token_id=1,
            pad_token_id=1,
        ))
        unet = UNet2DConditionModel(
            sample_size=32, in_channels=8 if instruct else 4, out_channels=4,
            layers_per_block=1, block_out_channels=(16, 32),
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=16, attention_head_dim=4, norm_num_groups=8,
        )
        vae = AutoencoderKL(
            in_channels=3, out_channels=3, latent_channels=4,
            block_out_channels=(16, 32), layers_per_block=1,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
            norm_num_groups=8, sample_size=64,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            clip_sample=False, set_alpha_to_one=False, steps_offset=1,
        )
        pipeline_class = (
            StableDiffusionInstructPix2PixPipeline if instruct
            else StableDiffusionImg2ImgPipeline
        )
        return pipeline_class(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            unet=unet, scheduler=scheduler, safety_checker=None,
            feature_extractor=None, requires_safety_checker=False,
        )

    @pytest.mark.parametrize('instruct', (False, True))
    def test_tiny_native_image_pipelines_load_offline_and_generate_seeded_images(self, instruct):
        import torch

        handler = DiffusionHandler()
        with tempfile.TemporaryDirectory() as directory:
            pipe = self._tiny_pipeline(directory, instruct)
            snapshot = Path(directory) / "snapshot"
            pipe.save_pretrained(snapshot)
            context = handler.load(
                str(snapshot), "image-to-image", "diffusers", "cpu",
                model_revision="offline-random-weights",
                load_options={"attention_implementation": "eager"},
            )
            payload = DiffusionWorkloadGenerator(
                "local/tiny", "image-to-image", 1,
            ).generate(64)
            payload["prompt"] = ["a"]
            payload["params"].update(num_inference_steps=2, guidance_scale=2.0)
            if not instruct:
                payload["params"]["strength"] = 0.8
            processed = handler.preprocess(context, payload)
            with torch.inference_mode():
                first = handler.predict(context, processed)
                second = handler.predict(context, processed)
            metadata = handler.postprocess(context, first)
            assert (metadata["output_shape"]) == ([1, 64, 64, 3])
            assert (metadata["n_results"]) == (1)
            assert (first.result.images[0].tobytes()) == (second.result.images[0].tobytes())
            assert (context["model"].config._attn_implementation) == ("eager")

    @pytest.mark.parametrize('output_class_case', range(2), ids=['CogVideoXPipelineOutput', 'WanPipelineOutput'])
    def test_native_video_output_containers_return_frame_metadata(self, output_class_case):
        from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
        from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
        from PIL import Image

        frames = [[Image.new("RGB", (64, 64), (10, 20, 30)) for _ in range(5)]]
        output_class = tuple((CogVideoXPipelineOutput, WanPipelineOutput))[output_class_case]
        metadata = DiffusionHandler().postprocess(
            {"task_type": "image-to-video"}, output_class(frames=frames),
        )
        assert (metadata["output_shape"]) == ([1, 5, 64, 64, 3])
        assert (metadata["video_frame_count"]) == (5)
        assert (metadata["output_length"]) == (5)
        assert ("frames") not in (metadata)

    @pytest.mark.parametrize('root_layout', (False, True))
    def test_tiny_ddpm_runs_offline_with_native_resolution_and_step_scales(self, root_layout):
        import torch
        from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

        torch.manual_seed(42)
        pipe = DDPMPipeline(
            unet=UNet2DModel(
                sample_size=32, in_channels=3, out_channels=3,
                block_out_channels=(8, 16), layers_per_block=1,
                down_block_types=("DownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "UpBlock2D"), norm_num_groups=4,
            ), scheduler=DDPMScheduler(num_train_timesteps=20),
        )
        handler = DiffusionHandler()
        task = "unconditional-image-generation"
        from acprof.container.model_files import plan_download
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "source")
            snapshot = Path(directory, "selected")
            snapshot.mkdir()
            pipe.save_pretrained(source)
            if root_layout:
                for component in ("unet", "scheduler"):
                    for path in (source / component).iterdir():
                        path.rename(source / path.name)
                    (source / component).rmdir()
            plan = plan_download(model_id="local/ddpm", revision="a" * 40,
                family="diffusion", backend="diffusers",
                files={str(p.relative_to(source)): {"size": p.stat().st_size} for p in source.rglob("*") if p.is_file()},
                read_json=lambda name: json.loads((source / name).read_text()))
            for record in plan["files"]:
                target = snapshot / record["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / record["path"], target)
            context = handler.load(str(snapshot), task, "diffusers", "cpu")
            payload = DiffusionWorkloadGenerator("local/ddpm", task, 1).generate(2)
            processed = handler.preprocess(context, payload)
            first = handler.predict(context, processed)
            second = handler.predict(context, processed)
            metadata = handler.postprocess(context, first)
            assert (first.images[0].tobytes()) == (second.images[0].tobytes())
            assert (metadata["output_shape"]) == ([1, 32, 32, 3])
            assert (handler.get_scale_metadata(context, payload)["native_output_width"]) == (32)

    @pytest.mark.parametrize('pipeline_class_case', range(7), ids=["(CogVideoXPipeline, 'text-to-video')", "(CogVideoXVideoToVideoPipeline, 'video-to-video')", "(StableVideoDiffusionPipeline, 'image-to-video')", "(TextToVideoSDPipeline, 'text-to-video')", "(VideoToVideoSDPipeline, 'video-to-video')", "(ShapEPipeline, 'text-to-3d')", "(ShapEImg2ImgPipeline, 'image-to-3d')"])
    def test_new_native_video_and_shap_e_signatures_match_pinned_runtime(self, pipeline_class_case):
        import types

        from diffusers import (
            CogVideoXPipeline,
            CogVideoXVideoToVideoPipeline,
            ShapEImg2ImgPipeline,
            ShapEPipeline,
            StableVideoDiffusionPipeline,
            TextToVideoSDPipeline,
            VideoToVideoSDPipeline,
        )

        (pipeline_class, task) = tuple([(CogVideoXPipeline, 'text-to-video'), (CogVideoXVideoToVideoPipeline, 'video-to-video'), (StableVideoDiffusionPipeline, 'image-to-video'), (TextToVideoSDPipeline, 'text-to-video'), (VideoToVideoSDPipeline, 'video-to-video'), (ShapEPipeline, 'text-to-3d'), (ShapEImg2ImgPipeline, 'image-to-3d')])[pipeline_class_case]
        call = functools.partial(pipeline_class.__call__, None)
        call.shap_e_renderer = types.SimpleNamespace(decode_to_mesh=lambda: None)
        parameters = _native_pipeline_parameters(call, task)
        assert ("num_inference_steps") in (parameters)
        if task in {"text-to-3d", "image-to-3d"}:
            assert ("frame_size") in (parameters)
            assert ("height") not in (parameters)
        elif task == "video-to-video":
            assert ("video") in (parameters)
            assert ("num_frames") not in (parameters)

    @pytest.mark.parametrize('task', ('text-to-video', 'video-to-video'))
    def test_tiny_native_video_generation_and_task_conversion(self, task):
        from diffusers import TextToVideoSDPipeline, UNet3DConditionModel

        with tempfile.TemporaryDirectory() as directory:
            image_pipe = self._tiny_pipeline(directory, False)
            pipe = TextToVideoSDPipeline(
                vae=image_pipe.vae, tokenizer=image_pipe.tokenizer,
                text_encoder=image_pipe.text_encoder, scheduler=image_pipe.scheduler,
                unet=UNet3DConditionModel(
                    sample_size=32, in_channels=4, out_channels=4,
                    layers_per_block=1, block_out_channels=(32, 32),
                    down_block_types=("DownBlock3D", "CrossAttnDownBlock3D"),
                    up_block_types=("CrossAttnUpBlock3D", "UpBlock3D"),
                    cross_attention_dim=16, attention_head_dim=4,
                    norm_num_groups=8,
                ),
            )
            snapshot = Path(directory) / "video"
            pipe.save_pretrained(snapshot)
            handler = DiffusionHandler()
            context = handler.load(str(snapshot), task, "diffusers", "cpu")
            payload = DiffusionWorkloadGenerator("local/video", task, 1).generate(64)
            payload["prompt"] = ["a"]
            payload["params"].update(num_inference_steps=2, num_frames=5, guidance_scale=2.0)
            if task == "video-to-video":
                payload["frames_base64"] = payload["frames_base64"][:5]
            processed = handler.preprocess(context, payload)
            output = handler.predict(context, processed)
            assert (handler.postprocess(context, output)["output_shape"]) == ([1, 5, 64, 64, 3])
