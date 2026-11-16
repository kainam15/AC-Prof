"""Offline Kokoro 0.9.4 adapter; official synthesis stays inside predict."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from acprof.container.handlers import BaseHandler, InputLimitError
from acprof.failures import Failure, RuntimeFailure, failure_from_exception
from acprof.models.files import ModelFilesError
from acprof.models.kokoro import snapshot_contract, validate_snapshot


def _check_g2p_dependencies() -> None:
    import spacy
    from misaki import espeak

    # Misaki otherwise calls spacy.cli.download, even in HF offline mode.
    if not spacy.util.is_package("en_core_web_sm"):
        raise RuntimeFailure(Failure(
            "load", "runtime_dependency_missing", "Kokoro requires the locked en_core_web_sm package; rebuild the kokoro-cpu environment.",
            evidence={"dependency": "en_core_web_sm", "next_action": "Prepare the complete kokoro-cpu dependency lock before loading."}))
    # KPipeline catches this failure and silently disables English fallback.
    # Exercise the exact official backend first, before constructing any model.
    try:
        espeak.EspeakFallback(british=False)
    except (RuntimeError, OSError) as exc:
        raise RuntimeFailure(Failure(
            "load", "runtime_dependency_missing", "Kokoro requires a working espeak-ng library and data from espeakng-loader.",
            evidence={"next_action": "Repair the locked espeakng-loader package and its native library dependencies."},
            exception_type=type(exc).__name__)) from exc


def _array(audio: Any) -> np.ndarray:
    if callable(getattr(audio, "detach", None)):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio)


class KokoroHandler(BaseHandler):
    def load(self, model_source: str, task_type: str, backend: str, device: str,
             model_revision: str = "main", load_options: dict[str, Any] | None = None,
             *, adapter: str = "kokoro") -> dict[str, Any]:
        if (task_type, backend, adapter, device) != ("text-to-speech", "kokoro", "kokoro", "cpu"):
            raise ValueError("Kokoro supports text-to-speech / kokoro on CPU FP32 only")
        options = dict(load_options or {})
        if (set(options) - {"dtype", "trust_remote_code"} or options.get("dtype", "FP32") != "FP32"
                or options.get("trust_remote_code", False) is not False):
            raise ValueError("Kokoro supports only FP32 with trust_remote_code=false")
        contract = snapshot_contract()
        root = Path(model_source).resolve()
        if not root.is_dir():
            raise ModelFilesError("Kokoro requires a fixed local snapshot directory")
        config_file = root / "config.json"
        if not config_file.is_file():
            raise ModelFilesError("Kokoro snapshot is incomplete: missing config.json")
        names = [path.relative_to(root).as_posix() for path in [config_file, root / contract["weights"], *root.glob("voices/*.pt")]
                 if path.is_file()]
        voices = validate_snapshot(contract["model_id"], model_revision,
                                   json.loads(config_file.read_text(encoding="utf-8")), names)
        # Model Store views intentionally link to verified blobs outside the
        # snapshot directory. Their identity/hash checks belong to preparation.
        _check_g2p_dependencies()
        import torch
        from kokoro import KModel, KPipeline

        model = KModel(repo_id=contract["model_id"], config=str(config_file),
                       model=str(root / contract["weights"])).to(device="cpu", dtype=torch.float32).eval()
        pipelines = {language: KPipeline(lang_code=language, repo_id=contract["model_id"], model=model, device="cpu")
                     for language in sorted({Path(name).stem[0] for name in voices})}
        packs = {}
        for name in voices:
            voice = Path(name).stem
            pack = pipelines[voice[0]].load_voice(str(root / name))
            array = _array(pack)
            if (array.ndim != 3 or array.shape[0] < 510 or array.shape[1:] != (1, 256)
                    or array.dtype != np.float32 or not np.isfinite(array).all()):
                raise ModelFilesError(f"Invalid Kokoro voice pack: {name}; expected finite FP32 [>=510, 1, 256]")
            packs[voice] = pack
        return {"model": model, "pipelines": pipelines, "voices": packs, "task_type": task_type,
                "device": device, "model_revision": model_revision, "load_options": options,
                "sample_rate": contract["sample_rate"]}

    def preprocess(self, model_ctx: dict[str, Any], raw_input: dict[str, Any]) -> dict[str, Any]:
        params = raw_input.get("params", {})
        if not isinstance(params, dict) or set(params) - {"language", "voice", "speed"}:
            raise ValueError("Kokoro params accept language, voice and speed only")
        language, voice, speed = params.get("language", "a"), params.get("voice", "af_heart"), params.get("speed", 1.0)
        if not isinstance(language, str) or language not in model_ctx["pipelines"]:
            raise ValueError(f"Kokoro language {language!r} has no prepared pipeline/voice")
        if not isinstance(voice, str) or voice not in model_ctx["voices"] or not voice.startswith(language):
            raise ValueError(f"Kokoro voice {voice!r} is missing from the snapshot or mismatches language {language!r}")
        if type(speed) not in (int, float) or not 0.25 <= speed <= 4:
            raise ValueError("Kokoro speed must be finite and between 0.25 and 4")
        if raw_input.get("batch_size", 1) != 1:
            raise ValueError("Kokoro supports batch_size=1")
        if ("text" in raw_input) == ("phonemes" in raw_input):
            raise ValueError("Kokoro requires exactly one of text or phonemes")
        pipeline = model_ctx["pipelines"][language]
        text = raw_input.get("text")
        phonemes = raw_input.get("phonemes")
        segments: list[str] = []
        if "phonemes" in raw_input:
            if isinstance(phonemes, str):
                segments = [phonemes]
            elif isinstance(phonemes, list):
                segments = phonemes
            else:
                raise ValueError("Kokoro phonemes must be a non-empty string or list of non-empty strings")
            if not segments or any(not isinstance(item, str) or not item.strip() for item in segments):
                raise ValueError("Kokoro phonemes must be a non-empty string or list of non-empty strings")
        else:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Kokoro requires non-empty text")
            for paragraph in text.splitlines():
                if not paragraph.strip():
                    continue
                ps, tokens = pipeline.g2p(paragraph)
                if language in {"a", "b"}:
                    if any(not token.phonemes and any(char.isalpha() for char in token.text) for token in tokens):
                        raise ValueError("Kokoro G2P could not phonemize a word; input would be omitted")
                    segments.extend(ps for _, ps, _ in pipeline.en_tokenize(tokens) if ps)
                else:
                    segments.append(ps)
        if not segments or any(not ps for ps in segments):
            raise ValueError("Kokoro G2P produced no phonemes")
        vocab = model_ctx["model"].vocab
        for ps in segments:
            if len(ps) > 510 or len(ps) + 2 > model_ctx["model"].context_length:
                raise InputLimitError(
                    "Kokoro segment exceeds 510 phoneme characters; split the input at sentence/newline boundaries",
                    effective_input_scale=sum(map(len, segments)))
            if any(vocab.get(char) is None for char in ps):
                raise ValueError("Kokoro phonemes contain characters outside the model vocabulary; refusing silent omission")
        count = sum(len(ps) for ps in segments)
        facts = {"tokens": None, "token_scope": "not_applicable", "characters": len(text) if text is not None else None,
                 "words": len(text.split()) if text is not None else None, "phoneme_characters": count,
                 "phoneme_ids": count, "model_input_ids_including_boundaries": count + 2 * len(segments),
                 "segments": len(segments)}
        return {"segments": list(segments), "language": language, "voice": voice, "speed": speed,
                "_effective_input_scale": count, "_truncated_by_limit": False,
                "_probe_reason": "Kokoro vocabulary IDs from actual G2P segments, excluding two boundary IDs per segment; not tokenizer tokens",
                "_workload": {"input": {"scale_unit": "phoneme_ids", "text": facts},
                              "generation": {"language": language, "voice": voice, "speed": speed}}}

    def predict(self, model_ctx: dict[str, Any], processed_input: dict[str, Any]) -> dict[str, Any]:
        return self.synthesize(model_ctx, processed_input["segments"], language=processed_input["language"],
                               voice=processed_input["voice"], speed=processed_input["speed"])

    def synthesize(self, model_ctx: dict[str, Any], phoneme_segments: list[str], *,
                   language: str, voice: str, speed: float) -> dict[str, Any]:
        """Observable adapter consumer for frozen language, voice and speed."""
        pipeline = model_ctx["pipelines"][language]
        segments = []
        for index, phonemes in enumerate(phoneme_segments):
            try:
                # Exhaust each official generator here. Late exceptions belong to
                # inference, never to serialization or a subsequent request.
                segments.extend(pipeline.generate_from_tokens(
                    phonemes, voice=model_ctx["voices"][voice], speed=speed, model=model_ctx["model"]))
            except Exception as exc:
                raise RuntimeFailure(failure_from_exception(exc, stage="predict", device="cpu",
                    runtime_profile="kokoro-cpu", evidence={"segment_index": index})) from exc
        return {"segments": segments, "sample_rate": model_ctx["sample_rate"],
                "language": language, "voice": voice, "speed": speed}

    def postprocess(self, model_ctx: dict[str, Any], raw_output: dict[str, Any]) -> dict[str, Any]:
        # Only summarize existing shapes in the request; full numeric validation
        # runs independently outside formal measurement windows.
        shapes = [tuple(segment.audio.shape) for segment in raw_output["segments"]]
        if not shapes or any(len(shape) != 1 or shape[0] <= 0 for shape in shapes):
            raise ValueError("Kokoro output must contain non-empty mono audio segments")
        samples = sum(shape[0] for shape in shapes)
        return {"task": model_ctx["task_type"], "output_type": "audio", "audio_shape": [samples],
                "audio_sample_rate": raw_output["sample_rate"], "audio_num_samples": samples,
                "audio_duration_s": samples / raw_output["sample_rate"], "output_length": samples,
                "output_token_count": None, "n_results": 1, "segment_count": len(shapes),
                "language": raw_output["language"], "voice": raw_output["voice"], "speed": raw_output["speed"]}

    def validate_output(self, model_ctx, raw_input, processed_input, raw_output, response):
        return validate_output(model_ctx, raw_input, processed_input, raw_output, response)

    def get_scale_metadata(self, model_ctx, raw_input):
        return {"input_scale_type": "phoneme_ids", "max_effective_input_scale": 510,
                "reason": "Automatic planning targets at most 510 phoneme IDs; this is an auto-planning cap, not a total request limit. Each segment is checked separately."}


def validate_output(model_ctx, raw_input, processed_input, raw_output, response) -> dict[str, Any]:
    arrays = [_array(segment.audio) for segment in raw_output["segments"]]
    if not arrays or any(array.ndim != 1 or array.size == 0 or array.dtype.kind != "f" or not np.isfinite(array).all() for array in arrays):
        raise ValueError("Kokoro output must be non-empty, finite floating-point mono audio")
    rate = raw_output["sample_rate"]
    if type(rate) is not int or rate != 24000 or response.get("audio_sample_rate") != rate:
        raise ValueError("Kokoro output sample rate must be 24000 Hz")
    count = sum(array.size for array in arrays)
    if (response.get("output_type") != "audio" or response.get("audio_shape") != [count]
            or response.get("audio_num_samples") != count or response.get("audio_duration_s") != count / rate):
        raise ValueError("Kokoro audio summary disagrees with the actual output")
    from acprof.container.validation import validate_output as validate_protocol
    validation = validate_protocol(model_ctx, raw_input, processed_input, raw_output, response)
    validation["task"]["checks"].extend(["kokoro_mono_finite_audio", "kokoro_sample_rate", "kokoro_audio_summary"])
    return validation
