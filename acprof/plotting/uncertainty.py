"""Render window statistics using the same estimator as stats/CSV exports.

The optional cache belongs to one plotting invocation. Its keys contain actual
observations and measurement identity, not a pathname or mutable DataFrame id.
Lifecycle/profiler observations and medians deliberately keep their own semantics.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from acprof.analysis.uncertainty import summarize_windows
from acprof.plotting.data import aggregate_metric_series, is_gpu_on, make_config_labels
from acprof.results.csv import KEY_FIELDS
from acprof.results.metrics import METRICS

_CONTEXT = ("environment_class", "runtime_kind", "runtime_id", "resource_scope")
_INTERVAL_FIELDS = ("ci_low", "ci_high", "n_windows", "missing_windows", "std", "reason")
INTERVAL_NOTE = "Mean; 95% within-run bootstrap CI. No band/bar when unavailable."


class WindowStatistics:
    """Memoize identical groups only for the lifetime of a figure pipeline."""

    def __init__(self) -> None:
        self.groups: dict[tuple, dict | None] = {}

    def summarize(self, rows: list[dict], metric: str) -> dict | None:
        fields = (*KEY_FIELDS, *_CONTEXT, "status", metric)
        key = (metric, tuple(sorted(tuple(str(row.get(name, "")) for name in fields) for row in rows)))
        if key not in self.groups:
            groups = summarize_windows(rows, [metric])["groups"]
            if len(groups) > 1:
                raise ValueError("Plot series mixes runtime or measurement identities; split its inputs")
            self.groups[key] = groups[0] if groups else None
        return self.groups[key]


def _window_metric(frame: pd.DataFrame, metric: str) -> tuple[str, float] | None:
    factor = 1.0
    source = metric
    if metric.endswith("_gib") and metric[:-4] + "_bytes" in frame:
        source, factor = metric[:-4] + "_bytes", 1 / 1024 ** 3
    spec = METRICS.get(source)
    if (spec is None or spec.kind != "number"
            or spec.window not in {"request_window", "matched_control"}):
        return None
    if not {*KEY_FIELDS, "status", source}.issubset(frame.columns):
        # Direct library plots may be supplied already-aggregated values. They
        # do not establish independent windows, so must never invent intervals.
        return None
    return source, factor


def aggregate_plot_series(
    frame: pd.DataFrame, metric: str, *, config_columns: Sequence[str],
    agg_func: str = "mean", sort: bool = True, statistics: WindowStatistics | None = None,
) -> pd.DataFrame:
    specification = _window_metric(frame, metric)
    if agg_func != "mean" or specification is None:
        return aggregate_metric_series(frame, metric, config_columns=config_columns, agg_func=agg_func, sort=sort)
    source, factor = specification
    cache = statistics if statistics is not None else WindowStatistics()
    group_columns = [*config_columns, "input_scale"]
    fields = list(dict.fromkeys([*KEY_FIELDS, *[name for name in _CONTEXT if name in frame], "status", source]))
    records = []
    for coordinates, group in frame.groupby(group_columns, dropna=False, sort=sort):
        summary = cache.summarize(group[fields].to_dict("records"), source)
        if summary is None:
            continue
        values = {name: summary[name] for name in _INTERVAL_FIELDS}
        for name in ("ci_low", "ci_high", "std"):
            values[name] = np.nan if values[name] is None else values[name] * factor
        records.append({**dict(zip(group_columns, coordinates, strict=True)), **values,
                        metric: np.nan if summary["mean"] is None else summary["mean"] * factor})
    result = pd.DataFrame(records, columns=[*group_columns, metric, *_INTERVAL_FIELDS])
    if not result.empty:
        usable = result.groupby(list(config_columns), dropna=False, sort=False)[metric].transform("count") > 0
        result = result[usable].reset_index(drop=True)
    return result


def aggregate_plot_metric(
    frame: pd.DataFrame, metric: str, *, agg_func: str = "mean", statistics: WindowStatistics | None = None,
) -> pd.DataFrame:
    columns = ["cpu_cores", "mem_cap_gb", "gpu_mode"]
    if "environment_class" in frame:
        columns.insert(0, "environment_class")
    result = aggregate_plot_series(frame, metric, config_columns=columns, agg_func=agg_func, statistics=statistics)
    if not result.empty:
        result["config"] = make_config_labels(result)
        result["gpu_on"] = result["gpu_mode"].map(is_gpu_on)
    return result


def draw_window_interval(axis, series: pd.DataFrame, *, color) -> None:
    """Keep missing scales as gaps; vertical bounds also show isolated estimates."""
    if "ci_low" not in series or series.empty:
        return
    x = series["input_scale"].to_numpy(dtype=float)
    low = series["ci_low"].to_numpy(dtype=float, na_value=np.nan)
    high = series["ci_high"].to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(low) & np.isfinite(high)
    if valid.any():
        axis.fill_between(x, low, high, where=valid, color=color, alpha=0.16)
        axis.vlines(x[valid], low[valid], high[valid], color=color, alpha=0.6, linewidth=1)
