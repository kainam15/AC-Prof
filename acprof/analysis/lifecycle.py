"""Deduplicate reused startup evidence without inferring independent starts."""

import math
from collections.abc import Iterable, Mapping

_MISSING_IDENTITIES = {"", "none", "null", "nan", "nat", "n/a", "unknown", "<na>"}


class LifecycleConflict(ValueError):
    """Recorded startup evidence cannot support an unambiguous aggregate."""


def unique_startup_observations(
    observations: Iterable[tuple[object, Mapping[str, float]]],
) -> list[dict[str, float]]:
    """Keep one consistent value per metric/startup, including legacy reuse.

    Callers provide eligible, finite metric values within one configuration.
    Without identity only one consistent reused observation is defensible;
    mixing anonymous and identified observations cannot establish independence.
    """
    startups: dict[str | None, dict[str, float]] = {}
    for raw_identity, fields in observations:
        if not fields:
            continue
        text = str(raw_identity).strip()
        identity = None if raw_identity is None or text.lower() in _MISSING_IDENTITIES else text
        if startups and (identity is None) != (None in startups):
            raise LifecycleConflict("Cannot combine missing startup identity with identified startups")
        recorded = startups.setdefault(identity, {})
        for field, value in fields.items():
            if not math.isfinite(value):
                raise LifecycleConflict(f"Non-finite {field} for startup {identity!r}")
            if field in recorded and recorded[field] != value:
                raise LifecycleConflict(
                    f"Conflicting {field} for startup {identity!r}: {recorded[field]!r} != {value!r}"
                )
            recorded[field] = value
    return list(startups.values())
