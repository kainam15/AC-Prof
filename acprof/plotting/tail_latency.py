"""Window-percentile aggregation and tail-latency plots."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from acprof.plotting.data import (
    is_gpu_on,
    make_config_labels,
    representative_max_memory_rows,
)
from acprof.plotting.styles import configurations_with_colors

_FAMILIES = (
    ("Packet latency percentiles", "latency_p50_s", "latency_p90_s", "latency_p95_s"),
    ("Application latency percentiles", "latency_app_p50_s", "latency_app_p90_s", "latency_app_p95_s"),
)


def _mask_invalid_percentiles(frame: pd.DataFrame) -> pd.Series:
    """Mask invalid pairs in an owned frame, keeping missing scale positions."""
    available = pd.Series(False, index=frame.index)
    for _, p50, p90, p95 in _FAMILIES:
        paired = (
            np.isfinite(frame[p50]) & np.isfinite(frame[p95])
            & (frame[p50] > 0) & (frame[p95] >= frame[p50])
        ).fillna(False)
        available |= paired
        frame[[p50, p90, p95]] = frame[[p50, p90, p95]].where(paired, axis=0)
        valid_p90 = (
            np.isfinite(frame[p90]) & frame[p90].between(frame[p50], frame[p95])
        ).fillna(False)
        frame[p90] = frame[p90].where(valid_p90)
    return available


def _aggregate_tail_latency(df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for family in _FAMILIES for column in family[1:]]
    present = [column for column in metric_columns if column in df.columns]
    if not present:
        return pd.DataFrame()
    group_cols = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale"]
    usable = df[[*group_cols, *present]].reindex(columns=[*group_cols, *metric_columns])
    available = _mask_invalid_percentiles(usable)
    usable = representative_max_memory_rows(usable, available=available)
    if usable.empty:
        return usable
    # Pair before taking medians: complementary missing repeats must not
    # manufacture a percentile band that no window actually supplied.
    aggregated = usable.groupby(group_cols, as_index=False)[metric_columns].median()
    # A sparse optional P90 can have a median outside the paired P50/P95 band.
    # Keep it missing instead of clipping it or drawing an inconsistent curve.
    _mask_invalid_percentiles(aggregated)
    aggregated["config"] = make_config_labels(aggregated)
    aggregated["gpu_on"] = aggregated["gpu_mode"].map(is_gpu_on)
    return aggregated


def _mark_isolated_values(line: Line2D) -> None:
    """Make a valid percentile visible when neither neighbor can form a line."""
    valid = np.isfinite(line.get_ydata())
    isolated = valid.copy()
    isolated[1:] &= ~valid[:-1]
    isolated[:-1] &= ~valid[1:]
    if isolated.any():
        line.set_marker("o")
        line.set_markersize(4)
        line.set_markevery(np.flatnonzero(isolated).tolist())


def plot_tail_latency_overview(
    df: pd.DataFrame,
    *,
    xlabel: str,
    out_png: str | None,
    show_plots: bool = False,
) -> bool:
    """Show paired window P50/P90/P95 and P95/P50 with explicit data gaps."""
    aggregated = _aggregate_tail_latency(df)
    if aggregated.empty:
        print("[skip] Tail latency percentiles have no valid P50/P95 pairs")
        return False

    configs, config_colors = configurations_with_colors(aggregated)
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharex="col", squeeze=False)
    plotted_any = False
    plotted_p90 = False
    for column_index, (title, p50_column, p90_column, p95_column) in enumerate(_FAMILIES):
        percentile_axis = axes[0, column_index]
        ratio_axis = axes[1, column_index]
        percentile_axis.set_title(title)
        ratio_axis.set_title("P95 / P50 tail inflation")
        family_plotted = False
        for cpu, mem, gpu_on in configs:
            sub_df = aggregated[
                (aggregated["cpu_cores"] == cpu)
                & (aggregated["mem_cap_gb"] == mem)
                & (aggregated["gpu_on"] == gpu_on)
            ].sort_values("input_scale")
            if not sub_df[p50_column].notna().any():
                continue
            family_plotted = True
            color = config_colors[(cpu, mem, gpu_on)]
            x_values = sub_df["input_scale"].to_numpy(dtype=float)
            p50_values = sub_df[p50_column].to_numpy(dtype=float, na_value=np.nan)
            p95_values = sub_df[p95_column].to_numpy(dtype=float, na_value=np.nan)
            percentile_axis.fill_between(
                x_values,
                p50_values,
                p95_values,
                color=color,
                alpha=0.12,
            )
            p50_line, = percentile_axis.plot(
                x_values,
                p50_values,
                color=color,
                linestyle="-",
                linewidth=1.8,
            )
            _mark_isolated_values(p50_line)
            if sub_df[p90_column].notna().any():
                plotted_p90 = True
                p90_line, = percentile_axis.plot(
                    x_values,
                    sub_df[p90_column].to_numpy(dtype=float, na_value=np.nan),
                    color=color,
                    linestyle="--",
                    linewidth=1.4,
                )
                _mark_isolated_values(p90_line)
            p95_line, = percentile_axis.plot(
                x_values,
                p95_values,
                color=color,
                linestyle=":",
                linewidth=1.8,
            )
            _mark_isolated_values(p95_line)
            ratio_axis.plot(
                x_values,
                p95_values / p50_values,
                color=color,
                marker="o",
                linewidth=1.8,
                markersize=5,
            )

        percentile_axis.set_ylabel("Latency (s)")
        ratio_axis.set_ylabel("P95 / P50")
        ratio_axis.set_xlabel(xlabel)
        for axis in (percentile_axis, ratio_axis):
            axis.grid(True, which="both", linestyle="-", alpha=0.35)
            if not family_plotted:
                axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
        plotted_any |= family_plotted

    if not plotted_any:
        plt.close(fig)
        print("[skip] Tail latency percentiles have no valid P50/P95 pairs")
        return False

    config_handles = [
        Line2D([0], [0], color=config_colors[config], linewidth=2)
        for config in configs
    ]
    config_labels = [
        f"{'GPU' if gpu_on else 'CPU'}+CPU{cpu}+Mem{mem}"
        for cpu, mem, gpu_on in configs
    ]
    percentile_handles: list[Line2D | Patch] = [Line2D([0], [0], color="black", linestyle="-", label="P50")]
    if plotted_p90:
        percentile_handles.append(Line2D([0], [0], color="black", linestyle="--", label="P90"))
    percentile_handles.extend([
        Line2D([0], [0], color="black", linestyle=":", label="P95"),
        Patch(facecolor="gray", alpha=0.18, label="P50–P95 band"),
    ])
    fig.suptitle(
        "Tail Latency at Representative Max-Memory Configurations",
        y=0.995,
    )
    fig.legend(
        handles=config_handles + percentile_handles,
        labels=config_labels + [handle.get_label() for handle in percentile_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(6, max(1, len(config_handles + percentile_handles))),
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    if out_png:
        fig.savefig(out_png, dpi=200)
        print(f"[saved] {out_png}")
    if show_plots:
        plt.show()
    plt.close(fig)
    return True
