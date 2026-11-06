"""Shared request contracts for streaming execution and offline evidence."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from acprof.artifacts import reject_isolated_surrogates

OPENAI_STREAMING_ENDPOINTS = frozenset({"/v1/chat/completions", "/v1/completions"})


def request_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_request_credentials(payload: dict[str, Any]) -> None:
    if any(str(key).lower() in {"authorization", "api_key", "apikey", "headers"} for key in payload):
        raise ValueError("credentials and headers are not request body fields; use --api-key-env")


def _validate_text_input(payload: dict[str, Any], endpoint: str) -> None:
    if endpoint == "/v1/completions":
        if not isinstance(payload.get("prompt"), str):
            raise ValueError("completions requires one text prompt")
        if "best_of" in payload and (type(payload["best_of"]) is not int or payload["best_of"] != 1):
            raise ValueError("streaming completions supports best_of=1 only")
    else:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages or any(
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str) or not message["role"].strip()
            or not isinstance(message.get("content"), str)
            for message in messages
        ):
            raise ValueError("chat completions requires nonempty text messages")


def prepare_streaming_payload(
    payload: dict[str, Any], *, endpoint: str, tokenizer_model: str | None = None,
) -> dict[str, Any]:
    """Validate text-only input and return normalized streaming fields without mutation."""
    if endpoint not in OPENAI_STREAMING_ENDPOINTS:
        raise ValueError("unsupported OpenAI streaming endpoint")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), str) or not payload["model"].strip():
        raise ValueError("OpenAI streaming requires a nonempty model")
    reject_isolated_surrogates(payload)
    if tokenizer_model is not None and payload["model"] != tokenizer_model:
        raise ValueError("tokenizer fallback model does not match the request model")
    if "n" in payload and (type(payload["n"]) is not int or payload["n"] != 1):
        raise ValueError("OpenAI streaming currently supports n=1 only")
    if "stream" in payload and payload["stream"] is not True:
        raise ValueError("OpenAI streaming requires stream=true")
    if any(payload.get(name) for name in ("tools", "functions", "tool_choice", "function_call", "audio")):
        raise ValueError("OpenAI streaming currently supports text output only")
    if payload.get("echo"):
        raise ValueError("echo output cannot be counted as generated tokens")
    _validate_text_input(payload, endpoint)
    options = payload.get("stream_options", {})
    if not isinstance(options, dict):
        raise ValueError("stream_options must be an object")
    if "include_usage" in options and options["include_usage"] is not True:
        raise ValueError("streaming measurement requires stream_options.include_usage=true")
    return {**payload, "stream": True, "n": 1, "stream_options": {**options, "include_usage": True}}
