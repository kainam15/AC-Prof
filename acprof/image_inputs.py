"""Decode bounded image collections before allocating expanded pixel buffers.

Host assets and container payloads share the same header-first checks. Limits
apply to the sum of source pixels, independently of any later output resize.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Collection, Mapping, Sequence
from contextlib import ExitStack
from typing import TYPE_CHECKING

from acprof.input_budget import InputBudget

if TYPE_CHECKING:
    from PIL.Image import Image


def decode_image_groups(
    groups: Mapping[str, Sequence[object]], *, budget: InputBudget,
    same_size: Collection[str] = (), mode: str = "RGB",
) -> dict[str, list[Image]]:
    """Decode Base64 fields, counting all groups as one request."""
    count = sum(len(values) for values in groups.values())
    if count:
        budget.batch(count)
    data: dict[str, list[bytes]] = {}
    for name, values in groups.items():
        data[name] = []
        for value in values:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must contain a non-empty Base64 image")
            try:
                data[name].append(base64.b64decode(value, validate=True))
            except ValueError as exc:
                raise ValueError(f"{name} must contain a valid Base64 image") from exc
    return load_image_groups(data, budget=budget, same_size=same_size, mode=mode)


def load_image_groups(
    groups: Mapping[str, Sequence[bytes]], *, budget: InputBudget,
    same_size: Collection[str] = (), mode: str = "RGB",
    additional_sizes: Sequence[tuple[int, int]] = (),
) -> dict[str, list[Image]]:
    """Inspect every source header and the aggregate budget before any convert."""
    from PIL import Image

    count = sum(len(values) for values in groups.values()) + len(additional_sizes)
    if count:
        budget.batch(count)
    with ExitStack() as sources, ExitStack() as decoded:
        opened: dict[str, list[Image.Image]] = {}
        sizes = list(additional_sizes)
        for name, values in groups.items():
            opened[name] = []
            for value in values:
                stream = sources.enter_context(io.BytesIO(value))
                try:
                    source = sources.enter_context(Image.open(stream))
                except Image.DecompressionBombError as exc:
                    raise ValueError(f"{name}: decoder pixel limit exceeded: {exc}") from exc
                except (ValueError, OSError) as exc:
                    raise ValueError(f"{name} must decode to a valid image") from exc
                opened[name].append(source)
                sizes.append(source.size)
        budget.image_collection(sizes)
        for name in same_size:
            if len({source.size for source in opened.get(name, [])}) > 1:
                raise ValueError(f"{name}: all video frames must have the same resolution (identical dimensions)")
        result: dict[str, list[Image.Image]] = {}
        for name, images in opened.items():
            result[name] = []
            for source in images:
                try:
                    image = source.convert(mode)
                except (ValueError, OSError) as exc:
                    raise ValueError(f"{name} must decode to a valid image") from exc
                decoded.callback(image.close)
                result[name].append(image)
        decoded.pop_all()  # Ownership of successful images transfers to the caller.
        return result
