"""Pure token budgets and text preparation shared by NLP handlers."""
from __future__ import annotations

from typing import Any

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


def _single_text_budget(max_len: int, special_tokens: int, reserved_tokens: int) -> int:
    available = max_len - special_tokens - reserved_tokens
    if reserved_tokens and available < 1:
        raise ValueError("max_new_tokens leaves no input token budget within the model context limit")
    return max(1, available)


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
    available = _single_text_budget(max_len, int(special_tokens), reserved_tokens)
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

    reason = (
        f"truncated_to_model_limit(max_length={max_len},available={available})"
        if truncated_by_limit
        else f"within_model_limit(max_length={max_len},available={available})"
    )
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


def max_effective_single_text_length(
    pipe: Any, reserved_tokens: int = 0,
) -> tuple[int | None, str]:
    tokenizer = getattr(pipe, "tokenizer", None)
    if not tokenizer:
        return None, "tokenizer_missing"

    max_len = get_model_max_length(pipe)
    if max_len is None:
        return None, "max_length_unknown"

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    available = _single_text_budget(max_len, int(special_tokens), reserved_tokens)
    reason = f"max_effective_single_text_length(max_length={max_len},available={available})"
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
