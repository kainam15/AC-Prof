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


def _is_single_prompt(prompt: object) -> bool:
    if isinstance(prompt, str):
        return True
    if not isinstance(prompt, list) or not prompt:
        return False
    tokens = prompt
    if len(prompt) == 1:
        first = prompt[0]
        if isinstance(first, str):
            return True
        if isinstance(first, list):
            tokens = first
    return bool(tokens) and all(type(token) is int and token >= 0 for token in tokens)


def _validate_chat_message(message: object, index: int) -> None:
    if not isinstance(message, dict):
        raise ValueError(f"messages[{index}] must be an object")
    role = message.get("role")
    if not isinstance(role, str) or not role.strip():
        raise ValueError(f"messages[{index}].role requires nonempty text")
    content = message.get("content")
    if isinstance(content, str):
        return
    field = f"messages[{index}].content"
    if role not in {"developer", "system", "user", "assistant", "tool"}:
        raise ValueError(f"{field} must be a string for this role")
    if not isinstance(content, list) or not content:
        raise ValueError(f"{field} requires a string or a nonempty list of text parts")
    kinds = ("text", "refusal") if role == "assistant" else ("text",)
    for position, part in enumerate(content):
        if not isinstance(part, dict) or part.get("type") not in kinds:
            raise ValueError(f"{field}[{position}] requires a part of type {' or '.join(kinds)}")
        kind = part["type"]
        if not isinstance(part.get(kind), str):
            raise ValueError(f"{field}[{position}].{kind} must be a string")


def _validate_text_input(payload: dict[str, Any], endpoint: str) -> None:
    if endpoint == "/v1/completions":
        if not _is_single_prompt(payload.get("prompt")):
            raise ValueError("completions requires one prompt: a string or a nonempty sequence of nonnegative "
                             "integer token IDs, optionally wrapped in a single-item list")
        if "best_of" in payload and (type(payload["best_of"]) is not int or payload["best_of"] != 1):
            raise ValueError("streaming completions supports best_of=1 only")
    else:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("chat completions requires nonempty text messages")
        for index, message in enumerate(messages):
            _validate_chat_message(message, index)


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
