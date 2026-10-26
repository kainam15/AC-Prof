"""Count observed token IDs without retokenizing decoded model output."""
from __future__ import annotations

from numbers import Integral
from typing import Any


def _token_rows(value: Any) -> list[list[int]]:
    value = value.tolist() if callable(getattr(value, "tolist", None)) else value
    if not isinstance(value, (list, tuple)) or not value:
        return []
    if all(isinstance(token, Integral) for token in value):
        return [[int(token) for token in value]]
    return [row for child in value for row in _token_rows(child)]


def count_input_tokens(prepared: Any) -> int | None:
    mask = prepared.get("attention_mask")
    rows = _token_rows(mask if mask is not None else prepared.get("input_ids"))
    return (sum(sum(row) if mask is not None else len(row) for row in rows) if rows else None)


def generation_evidence(generated: Any, config: Any, *, prompt_length: int = 0,
                        encoder_decoder: bool = False, max_new_tokens: Any = None,
                        input_tokens: int | None = None) -> dict[str, Any]:
    eos = getattr(config, "eos_token_id", None)
    eos_ids = {int(value) for value in (eos if isinstance(eos, (list, tuple)) else [eos])
               if isinstance(value, Integral)}
    start = getattr(config, "decoder_start_token_id", None)
    counts, reasons = [], []
    for row in _token_rows(generated):
        if encoder_decoder and not isinstance(start, Integral):
            continue
        row = row[1:] if encoder_decoder and row and isinstance(start, Integral) and row[0] == start else row
        if not encoder_decoder:
            row = row[prompt_length:]
        end = next((index + 1 for index, token in enumerate(row) if token in eos_ids), None)
        counts.append(end if end is not None else len(row))
        reasons.append("eos" if end is not None else "length" if
                       isinstance(max_new_tokens, Integral) and len(row) == max_new_tokens else None)
    return {"actual_output_tokens": sum(counts) if counts else None,
            "actual_output_tokens_per_sequence": counts or None,
            "actual_input_tokens": input_tokens,
            "stop_reason": reasons[0] if reasons and len(set(reasons)) == 1 else None,
            "stop_reason_per_sequence": reasons or None}
