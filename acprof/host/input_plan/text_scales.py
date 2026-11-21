"""Select distinct text payloads using lengths observed by the runtime probe."""

import math
from bisect import bisect_right
from collections.abc import Callable
from functools import cache
from typing import Any

from acprof.host.input_plan.probe import probe_exceeds_limit
from acprof.host.input_plan.scales import AUTO_INPUT_SCALE_COUNT, integer_auto_scales


def select_text_scale_probes(maximum: int, probe: Callable[[int], dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the nearest nominal scales and fill collisions from actual usable payloads."""
    observed: dict[float, dict[str, Any]] = {}

    @cache
    def sample(words: int) -> tuple[float, dict[str, Any]]:
        result = probe(words)
        if probe_exceeds_limit(result):
            return math.inf, result
        scale = result["effective_input_scale"]
        if not math.isfinite(scale) or scale < 0:
            raise RuntimeError(f"invalid effective text input scale {scale} for word_count={words}")
        if scale > maximum:
            return math.inf, result
        if scale > 0:
            observed.setdefault(scale, result)
        return scale, result

    def scale_for(words: int) -> float:
        return sample(words)[0]

    domain = range(1, maximum + 1)
    if maximum < AUTO_INPUT_SCALE_COUNT:
        for words in domain:
            sample(words)
        selected = observed
    else:
        selected = {}
        for target in integer_auto_scales(maximum):
            # Preserve the original search bounds, favoring the larger scale on ties.
            upper = int(target)
            split = bisect_right(domain, target, hi=upper, key=scale_for)
            neighbors = (sample(words) for words in (split, split + 1) if 1 <= words <= upper)
            usable = [(scale, result) for scale, result in neighbors if 0 < scale <= maximum]
            if usable:
                scale, result = min(usable, key=lambda item: (abs(item[0] - target), -item[0]))
                selected.setdefault(scale, result)

        # If the nominal searches saw too few lengths, skip equal-length runs
        # by bisection. Each iteration advances to a distinct length or the limit.
        index = 0
        while len(observed) < AUTO_INPUT_SCALE_COUNT and index < len(domain):
            scale = scale_for(domain[index])
            if scale == math.inf:
                break
            index = bisect_right(domain, scale, lo=index + 1, key=scale_for)

        def separation(scale: float) -> tuple[float, float]:
            return min((abs(scale - other) for other in selected), default=maximum), scale

        while len(selected) < min(AUTO_INPUT_SCALE_COUNT, len(observed)):
            scale = max(observed.keys() - selected.keys(), key=separation)
            selected[scale] = observed[scale]

    if not selected:
        raise RuntimeError(f"no non-truncated NLP payload fits the input limit of {maximum}")
    return [selected[scale] for scale in sorted(selected)]
