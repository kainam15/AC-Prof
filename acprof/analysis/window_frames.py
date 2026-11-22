"""Filter window values used in paired and median aggregations without changing their estimator."""
from collections.abc import Sequence

import pandas as pd

from acprof.results.profile import (
    OBSERVATION_CONTEXT_FIELDS,
    USABLE_QUALITIES,
    metric_observation,
    observation_signature,
)


def window_evidence_frame(source: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Copy the requested data and only the context consumed by metric_observation."""
    context = [field for field in OBSERVATION_CONTEXT_FIELDS if field in source] if "resource_metric_quality" in source else []
    return source.loc[:, list(dict.fromkeys([*columns, *context]))].copy()


def _window_signatures(row: dict, fields: tuple[str, ...], required: tuple[str, ...]) -> list[str | None]:
    missing: list[str | None] = [None] * len(fields)
    if any(pd.isna(row[field]) for field in required):
        return missing
    signatures = []
    for field in fields:
        signature = None
        if pd.notna(row[field]):
            observation = metric_observation(row, field)
            if observation["quality"] in USABLE_QUALITIES:
                signature = observation_signature(observation)
        if signature is None and field in required:
            return missing
        signatures.append(signature)
    return signatures


def mask_incompatible_windows(
    frame: pd.DataFrame, group_columns: Sequence[str], required: tuple[str, ...], *, optional: tuple[str, ...] = (),
) -> None:
    """Mask in place after numeric checks. Only complete pairs participate in each case's signature check."""
    if frame.empty or "resource_metric_quality" not in frame:
        return
    fields = required + optional
    context = [field for field in OBSERVATION_CONTEXT_FIELDS if field in frame]
    columns = list(dict.fromkeys([*fields, *context]))
    rows = frame[columns].itertuples(index=False, name=None)
    signatures = pd.DataFrame(
        (_window_signatures(dict(zip(columns, row, strict=True)), fields, required) for row in rows),
        index=frame.index, columns=fields,
    )
    case_keys = [frame[column] for column in group_columns]
    distinct = signatures.groupby(case_keys, sort=False, dropna=False).transform("nunique")
    usable = signatures.notna() & distinct.eq(1)
    usable = usable.where(usable[list(required)].all(axis=1), False, axis=0)
    frame[list(fields)] = frame[list(fields)].where(usable)
