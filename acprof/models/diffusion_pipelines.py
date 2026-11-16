"""Reviewed Diffusers task hints and artifact capabilities, without framework imports."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffusersPipeline:
    task: str | None = None
    component_layout: bool = False
    root_layout: bool = False


# A task hint alone does not authorize file selection or prove runtime compatibility.
_PIPELINES = {
    "DDPMPipeline": DiffusersPipeline("unconditional-image-generation", component_layout=True, root_layout=True),
    "DDIMPipeline": DiffusersPipeline("unconditional-image-generation", component_layout=True, root_layout=True),
    "PNDMPipeline": DiffusersPipeline("unconditional-image-generation"),
    "ShapEPipeline": DiffusersPipeline("text-to-3d"),
    "ShapEImg2ImgPipeline": DiffusersPipeline("image-to-3d"),
    "StableDiffusionPipeline": DiffusersPipeline("text-to-image", component_layout=True),
    "StableDiffusionXLPipeline": DiffusersPipeline("text-to-image", component_layout=True),
    "StableDiffusionImg2ImgPipeline": DiffusersPipeline("image-to-image", component_layout=True),
    "StableDiffusionXLImg2ImgPipeline": DiffusersPipeline("image-to-image", component_layout=True),
    "StableDiffusionInpaintPipeline": DiffusersPipeline(component_layout=True),
    "StableDiffusionXLInpaintPipeline": DiffusersPipeline(component_layout=True),
    "StableDiffusionInstructPix2PixPipeline": DiffusersPipeline("image-text-to-image"),
    "StableDiffusionUpscalePipeline": DiffusersPipeline("image-to-image"),
    "StableVideoDiffusionPipeline": DiffusersPipeline("image-to-video"),
    "CogVideoXPipeline": DiffusersPipeline("text-to-video"),
    "CogVideoXImageToVideoPipeline": DiffusersPipeline("image-to-video"),
    "CogVideoXVideoToVideoPipeline": DiffusersPipeline("video-to-video"),
    "WanPipeline": DiffusersPipeline("text-to-video"),
    "WanImageToVideoPipeline": DiffusersPipeline("image-to-video"),
    "WanVideoToVideoPipeline": DiffusersPipeline("video-to-video"),
    "TextToVideoSDPipeline": DiffusersPipeline("text-to-video"),
    "VideoToVideoSDPipeline": DiffusersPipeline("video-to-video"),
}


def diffusers_pipeline(class_name: object) -> DiffusersPipeline | None:
    """Return explicit capabilities; unknown or custom classes remain unregistered."""
    return _PIPELINES.get(class_name) if isinstance(class_name, str) else None
