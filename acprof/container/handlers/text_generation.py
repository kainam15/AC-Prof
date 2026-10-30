"""NLP text generation and decoded response metrics."""
from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any


def text_output_metrics(tokenizer: Any, raw_output: Any) -> dict[str, int | None]:
    """Summarize all returned texts, independently of generated token IDs."""
    metrics: dict[str, int | None] = {"output_length": None, "output_token_count": None}
    texts: list[str] = []
    pending = [raw_output]
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(reversed(value))
        elif isinstance(value, Mapping):
            text = next((value[key] for key in ("generated_text", "summary_text", "translation_text")
                         if key in value), None)
            if not isinstance(text, str):
                return metrics
            texts.append(text)
        else:
            return metrics
    if not texts:
        return metrics
    metrics["output_length"] = sum(len(text) for text in texts)
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            metrics["output_token_count"] = sum(len(encode(text, add_special_tokens=False)) for text in texts)
        except (TypeError, ValueError, RuntimeError):
            # Optional decoded-text statistics may be unavailable; retain the
            # known character count and never publish a partial token total.
            pass
    return metrics


def load_seq2seq(source, revision, *, device, dtype, model_kwargs):
    import transformers

    from acprof.container.handlers import model_revision_kwargs
    options = {**model_revision_kwargs(source, revision), "local_files_only": os.path.isdir(source),
               "trust_remote_code": False}
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **options)
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        source, **options, **{dtype_key: dtype}, **model_kwargs).to(device).eval()
    if not model.config.is_encoder_decoder:
        raise ValueError("seq2seq generation requires an Encoder-Decoder checkpoint")
    transformers.set_seed(0)
    return SimpleNamespace(model=model, tokenizer=tokenizer, generation_config=model.generation_config)


def generate(context, inputs):
    import torch

    pipe = context["pipeline"]
    encoded = pipe.tokenizer([inputs["text"]] * inputs["batch_size"], return_tensors="pt",
                             padding=True, truncation=False)
    from acprof.container.handlers.generation_evidence import (
        count_input_tokens,
        generation_evidence,
    )
    input_tokens = count_input_tokens(encoded)
    encoded = {name: tensor.to(pipe.model.device) for name, tensor in encoded.items()}
    params = dict(inputs["params"])
    with torch.inference_mode():
        tokens = pipe.model.generate(**encoded, **params)
    # Transfer and decode follow the same task-pipeline protocol as the existing
    # handler. Evidence counts generated IDs, never retokenizes decoded text.
    tokens = tokens.cpu() if callable(getattr(tokens, "cpu", None)) else tokens
    decoded = pipe.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    config = copy.deepcopy(pipe.generation_config)
    for key, value in params.items():
        setattr(config, key, value)
    context["generation_evidence"] = generation_evidence(
        tokens, config, encoder_decoder=True,
        max_new_tokens=params.get("max_new_tokens", params.get("max_length", getattr(config, "max_length", 20)) - 1),
        input_tokens=input_tokens)
    context["generation_evidence"]["generation_parameters"] = params
    context["generation_evidence"]["inference_interface"] = "AutoModelForSeq2SeqLM.generate"
    return [{"generated_text": text} for text in decoded]
