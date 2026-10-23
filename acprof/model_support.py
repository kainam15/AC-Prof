"""Reviewed Transformers capabilities and task-to-AutoModel mapping; no model imports."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

# Tasks select upstream operations; checkpoint names never take part in this map.
TASK_MAPPINGS = {
    "text-generation": ("CAUSAL_LM",), "conversational": ("CAUSAL_LM",),
    "text2text-generation": ("SEQ_TO_SEQ_CAUSAL_LM",), "translation": ("SEQ_TO_SEQ_CAUSAL_LM",),
    "summarization": ("SEQ_TO_SEQ_CAUSAL_LM",), "fill-mask": ("MASKED_LM",),
    "replaced-token-detection": ("PRETRAINING",),
    "text-classification": ("SEQUENCE_CLASSIFICATION",), "text-ranking": ("SEQUENCE_CLASSIFICATION",),
    "zero-shot-classification": ("SEQUENCE_CLASSIFICATION",), "token-classification": ("TOKEN_CLASSIFICATION",),
    "question-answering": ("QUESTION_ANSWERING",), "feature-extraction": ("",),
    "sentence-similarity": ("",), "image-classification": ("IMAGE_CLASSIFICATION",),
    "image-feature-extraction": ("IMAGE", ""), "image-to-text": ("VISION_2_SEQ", "IMAGE_TEXT_TO_TEXT"),
    "image-text-to-text": ("IMAGE_TEXT_TO_TEXT",), "video-text-to-text": ("IMAGE_TEXT_TO_TEXT",),
    "visual-question-answering": ("VISUAL_QUESTION_ANSWERING",),
    "document-question-answering": ("DOCUMENT_QUESTION_ANSWERING",),
    "table-question-answering": ("TABLE_QUESTION_ANSWERING", "SEQ_TO_SEQ_CAUSAL_LM"),
    "object-detection": ("OBJECT_DETECTION",), "zero-shot-object-detection": ("ZERO_SHOT_OBJECT_DETECTION",),
    "image-segmentation": ("IMAGE_SEGMENTATION", "SEMANTIC_SEGMENTATION", "INSTANCE_SEGMENTATION", "UNIVERSAL_SEGMENTATION"),
    "depth-estimation": ("DEPTH_ESTIMATION",), "video-classification": ("VIDEO_CLASSIFICATION",),
    "zero-shot-image-classification": ("ZERO_SHOT_IMAGE_CLASSIFICATION",), "mask-generation": ("MASK_GENERATION",),
    "keypoint-detection": ("KEYPOINT_DETECTION", "KEYPOINT_MATCHING"),
    "automatic-speech-recognition": ("CTC", "SPEECH_SEQ_2_SEQ"), "audio-classification": ("AUDIO_CLASSIFICATION",),
    "text-to-speech": ("TEXT_TO_WAVEFORM", "TEXT_TO_SPECTROGRAM"),
}


def _transformers_support_catalog(version: str) -> dict:
    path = Path(__file__).parent / "extensions" / "transformers" / f"{version}.json"
    if not path.is_file():
        raise ValueError(f"No reviewed support catalog for transformers=={version}; register and validate this runtime first")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("version") != version:
        raise ValueError(f"Invalid Transformers support catalog: {path}")
    return data


@lru_cache(maxsize=None)
def transformers_support(version: str) -> dict:
    return _transformers_support_catalog(version)["mappings"]


def supports_pipeline_task(version: str, task: str) -> bool:
    registry = _transformers_support_catalog(version).get("pipeline_registry")
    if not isinstance(registry, dict) or not isinstance(registry.get("tasks"), list):
        raise ValueError(f"Unreviewed pipeline registry for transformers=={version}")
    return registry.get("aliases", {}).get(task, task) in registry["tasks"]


def transformers_capabilities(version: str) -> dict[str, bool]:
    """Read explicitly reviewed local-loader capabilities; missing/unknown is an error."""
    capabilities = _transformers_support_catalog(version).get("capabilities")
    required = ("local_dynamic_transitive_imports", "local_dynamic_symlink_safe")
    reviewed = {}
    for name in required:
        value = capabilities.get(name) if isinstance(capabilities, dict) else None
        if not isinstance(value, bool):
            raise ValueError(
                f"Unreviewed Transformers capability {name} for transformers=={version}; "
                "complete the dynamic-module runtime acceptance before loading custom code"
            )
        reviewed[name] = value
    return reviewed


def audio_text_loader(version: str, config: dict) -> tuple[str, str | None] | None:
    """Select a native text-output Auto interface, shared by preflight and load.

    Composite models may expose a separately registered multimodal text head.
    Select that head without instantiating speech synthesis components. Never
    select a plain language submodel, which would discard the audio encoder.
    """
    mappings = transformers_support(version)
    model_type = config.get("model_type")
    for operation, auto_class in (
        ("SEQ_TO_SEQ_CAUSAL_LM", "AutoModelForSeq2SeqLM"),
        ("IMAGE_TEXT_TO_TEXT", "AutoModelForImageTextToText"),
    ):
        if model_type in mappings.get(f"MODEL_FOR_{operation}_MAPPING_NAMES", {}):
            return auto_class, None
    if model_type not in mappings.get("MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES", {}):
        return None
    multimodal = mappings.get("MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES", {})
    text_heads = [key for key, value in config.items()
                  if isinstance(value, dict) and value.get("model_type") in multimodal]
    if len(text_heads) == 1:
        return "AutoModelForImageTextToText", text_heads[0]
    return None


def supports_transformers_task(
    version: str, task: str, model_type: str, model_config: dict | None = None,
) -> bool | None:
    if task == "audio-text-to-text" and model_type:
        return audio_text_loader(version, model_config or {"model_type": model_type}) is not None
    if not model_type or task not in TASK_MAPPINGS:
        return None
    mappings = transformers_support(version)
    return any(model_type in mappings.get(f"MODEL_FOR_{name}_MAPPING_NAMES" if name else "MODEL_MAPPING_NAMES", {})
               for name in TASK_MAPPINGS[task])
