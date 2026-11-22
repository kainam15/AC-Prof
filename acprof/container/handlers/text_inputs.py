"""Pure token budgets and text preparation shared by NLP handlers."""
from __future__ import annotations

from typing import Any

from acprof.container.handlers import InputLimitError
from acprof.input_budget import InputBudget

# Host workloads use one model-independent mask placeholder.
_FILL_MASK_PLACEHOLDER = "[MASK]"


def get_tokenizer_max_length(tokenizer: Any) -> int | None:
    max_len = getattr(tokenizer, "model_max_length", None)
    if max_len is None:
        return None
    try:
        value = int(max_len)
        return value if 0 < value < 1_000_000 else None
    except (TypeError, ValueError, OverflowError):
        return None


def get_model_max_length(pipe: Any) -> int | None:
    limits = [get_tokenizer_max_length(getattr(pipe, "tokenizer", None))]
    # Sentence Transformers may configure a shorter sequence limit than
    # the underlying tokenizer; CrossEncoder exposes max_length instead.
    for name in ("max_seq_length", "max_length"):
        value = getattr(pipe, name, None)
        if isinstance(value, (int, float)) and 0 < value < 1_000_000:
            limits.append(int(value))
    config = getattr(getattr(pipe, "model", None), "config", None)
    for name in ("max_position_embeddings", "n_positions"):
        value = getattr(config, name, None)
        if isinstance(value, (int, float)) and 0 < value < 1_000_000:
            limits.append(int(value))
    known = [value for value in limits if value is not None]
    return min(known) if known else None


def _single_text_budget(max_len: int, special_tokens: int, reserved_tokens: int,
                        max_total_tokens: int | None) -> tuple[int, str]:
    if max_total_tokens is not None and max_total_tokens > max_len:
        raise ValueError(f"max_length={max_total_tokens} exceeds the usable model context limit of {max_len}")
    limit = max_len if max_total_tokens is None else max_total_tokens
    available = limit - special_tokens - reserved_tokens
    if reserved_tokens and available < 1:
        name = "max_new_tokens" if max_total_tokens is None else "max_length"
        raise ValueError(f"{name} leaves no input token budget within the model context limit")
    available = max(1, available)
    detail = f"max_length={max_len},available={available}"
    if max_total_tokens is not None:
        detail += f",generation_max_length={max_total_tokens}"
    return available, detail


def _prepare_fill_mask_input(
    tokenizer: Any,
    text: str,
) -> tuple[str, list, Any]:
    raw_mask_token = getattr(tokenizer, "mask_token", None)
    if raw_mask_token is None or not str(raw_mask_token):
        raise ValueError(
            "fill-mask task requires a tokenizer with a configured mask_token"
        )
    mask_token = str(raw_mask_token)

    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(convert_tokens_to_ids):
            mask_token_id = convert_tokens_to_ids(mask_token)
    if mask_token_id is None:
        raise ValueError(
            "fill-mask task requires a tokenizer with a configured mask_token_id"
        )

    if mask_token != _FILL_MASK_PLACEHOLDER:
        text = text.replace(_FILL_MASK_PLACEHOLDER, mask_token)

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if mask_token_id not in token_ids:
        separator = "" if not text or text[-1].isspace() else " "
        text = f"{text}{separator}{mask_token}"
        token_ids = tokenizer.encode(text, add_special_tokens=False)

    if mask_token_id not in token_ids:
        raise ValueError(
            "tokenizer could not encode its configured mask_token "
            f"{mask_token!r} as mask_token_id={mask_token_id!r}"
        )

    return text, token_ids, mask_token_id


def truncate_single_text(
    pipe: Any,
    text: str,
    task_type: str,
    reserved_tokens: int = 0,
    *, max_total_tokens: int | None = None,
) -> tuple[str, int | None, bool | None, str]:
    tokenizer = getattr(pipe, "tokenizer", None)
    if not tokenizer:
        return text, None, None, "tokenizer_missing"

    mask_token_id = None
    if task_type == "fill-mask":
        text, token_ids, mask_token_id = _prepare_fill_mask_input(
            tokenizer,
            text,
        )
    else:
        token_ids = tokenizer.encode(text, add_special_tokens=False)

    max_len = get_model_max_length(pipe)
    if max_len is None:
        return text, len(token_ids), None, "max_length_unknown"

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    available, detail = _single_text_budget(max_len, int(special_tokens), reserved_tokens, max_total_tokens)
    truncated_by_limit = len(token_ids) > available
    if truncated_by_limit:
        if task_type == "fill-mask":
            mask_pos = token_ids.index(mask_token_id)
            if mask_pos < available:
                token_ids = token_ids[:available]
            else:
                token_ids = token_ids[:available - 1] + [mask_token_id]
            text = tokenizer.decode(token_ids, skip_special_tokens=False)
        else:
            token_ids = token_ids[:available]
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if task_type == "fill-mask" and mask_token_id not in token_ids:
            raise ValueError(
                "tokenizer decode/encode round trip removed the configured "
                "mask token while truncating fill-mask input"
            )

    status = "truncated_to_model_limit" if truncated_by_limit else "within_model_limit"
    reason = f"{status}({detail})"
    if reserved_tokens:
        reason += f"; reserved_output_tokens={reserved_tokens}"
    return text, len(token_ids), truncated_by_limit, reason


def truncate_qa_context(
    pipe: Any,
    question: str,
    context: str,
) -> tuple[str, int | None, bool | None, str]:
    tokenizer = getattr(pipe, "tokenizer", None)
    if not tokenizer:
        return context, None, None, "tokenizer_missing"

    context_ids = tokenizer.encode(context, add_special_tokens=False)
    max_len = get_model_max_length(pipe)
    if max_len is None:
        return context, len(context_ids), None, "max_length_unknown"

    question_ids = tokenizer.encode(question, add_special_tokens=False)
    special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    available = max(0, max_len - len(question_ids) - int(special_tokens))
    truncated_by_limit = len(context_ids) > available
    if truncated_by_limit:
        context_ids = context_ids[:available]
        context = tokenizer.decode(context_ids, skip_special_tokens=True)
        context_ids = tokenizer.encode(context, add_special_tokens=False)

    reason = (
        f"truncated_context_to_model_limit(max_length={max_len},available={available})"
        if truncated_by_limit
        else f"within_model_limit(max_length={max_len},available={available})"
    )
    return context, len(context_ids), truncated_by_limit, reason


def chat_messages(text: str) -> list[dict[str, str]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("conversational requires nonempty text for a single user turn")
    return [{"role": "user", "content": text}]


def prepare_chat_prompt(pipe: Any, text: str, batch: int, budget: InputBudget,
                        reserved_tokens: int, max_total_tokens: int | None) -> dict:
    # Match TextGenerationPipeline's final user message, including the assistant generation prompt.
    tokenizer = pipe.tokenizer
    rendered = tokenizer.apply_chat_template(chat_messages(text), tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered, str) or not rendered.strip():
        raise ValueError("chat template produced no text for the user turn")
    budget.text(len(rendered) * batch, batch_items=batch)
    # apply_chat_template tokenizes its rendering without adding a second layer of special tokens.
    length = len(tokenizer.encode(rendered, add_special_tokens=False))
    if not length:
        raise ValueError("chat template produced no input tokens")
    available, reason = max_effective_single_text_length(
        pipe, reserved_tokens, max_total_tokens=max_total_tokens, prompt_includes_special_tokens=True,
    )
    if available is not None and length > available:
        raise InputLimitError(f"chat prompt requires {length} tokens, available={available} after output reservation",
                              effective_input_scale=length)
    return {"text": text, "_effective_input_scale": length, "_truncated_by_limit": False,
            "_input_token_scope": "templated_prompt_per_sample_including_special_tokens", "_probe_reason": reason}


def max_effective_single_text_length(
    pipe: Any, reserved_tokens: int = 0, *, max_total_tokens: int | None = None,
    prompt_includes_special_tokens: bool = False,
) -> tuple[int | None, str]:
    tokenizer = getattr(pipe, "tokenizer", None)
    if not tokenizer:
        return None, "tokenizer_missing"

    max_len = get_model_max_length(pipe)
    if max_len is None:
        if prompt_includes_special_tokens and max_total_tokens is not None:
            available, _ = _single_text_budget(max_total_tokens, 0, reserved_tokens, max_total_tokens)
            return available, (f"model context limit unknown; generation max_length={max_total_tokens},"
                               f"reserved_output_tokens={reserved_tokens},available={available}")
        return None, "max_length_unknown"

    special_tokens = 0 if prompt_includes_special_tokens else tokenizer.num_special_tokens_to_add(pair=False)
    available, detail = _single_text_budget(max_len, int(special_tokens), reserved_tokens, max_total_tokens)
    reason = f"max_effective_single_text_length({detail})"
    if reserved_tokens:
        reason += f"; reserved_output_tokens={reserved_tokens}"
    return available, reason


def max_effective_qa_context_length(
    pipe: Any,
    question: str,
) -> tuple[int | None, str]:
    tokenizer = getattr(pipe, "tokenizer", None)
    if not tokenizer:
        return None, "tokenizer_missing"

    max_len = get_model_max_length(pipe)
    if max_len is None:
        return None, "max_length_unknown"

    question_ids = tokenizer.encode(question, add_special_tokens=False)
    special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    available = max(0, max_len - len(question_ids) - int(special_tokens))
    return available, (
        "max_effective_qa_context_length("
        f"max_length={max_len},question_tokens={len(question_ids)},available={available})"
    )


def string_list(raw_input: dict[str, Any], field: str) -> list[str]:
    values = raw_input.get(field)
    if not isinstance(values, list) or not values or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise ValueError(f"{field} must be a nonempty list of nonempty strings")
    return list(values)
