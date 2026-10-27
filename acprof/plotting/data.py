"""结果 CSV 整理、分组和基础聚合。"""

import math
import os

import pandas as pd

from acprof.analysis.inputs import resource_coordinate
from acprof.gpu_mode import normalize_gpu_mode
from acprof.metric_registry import NUMERIC_FIELDS
from acprof.pixel_metrics import (
    PIXEL_COUNT_FIELDS,
    PIXEL_RATE_SOURCES,
    pixel_rate_metrics,
)
from acprof.platform import recorded_identity
from acprof.plotting.config import BYTES_PER_GIB, COMPUTE_NUMERIC_COLUMNS, PLOT_OUTPUT_DIRS


def _make_config_label(row) -> str:
    cpu = resource_coordinate(row["cpu_cores"]) if pd.notna(row["cpu_cores"]) else "inherited"
    mem = resource_coordinate(row["mem_cap_gb"]) if pd.notna(row["mem_cap_gb"]) else "inherited"
    if is_gpu_on(row["gpu_mode"]):
        label = f"GPU+CPU{cpu}+Mem{mem}"
    else:
        label = f"CPU+CPU{cpu}+Mem{mem}"
    return f"{row['environment_class']} / {label}" if "environment_class" in row else label


def make_config_labels(df: pd.DataFrame) -> pd.Series:
    """Only materialize resource identity columns for the row formatter."""
    if len(df.index) == 0:
        return pd.Series(index=df.index, dtype="object")
    columns = ["cpu_cores", "mem_cap_gb", "gpu_mode"]
    if "environment_class" in df.columns:
        columns.append("environment_class")
    return df.loc[:, columns].apply(_make_config_label, axis=1)


def is_gpu_on(gpu_mode: object) -> bool:
    return normalize_gpu_mode(gpu_mode) == "on"


def is_gpu_off(gpu_mode: object) -> bool:
    return normalize_gpu_mode(gpu_mode) == "off"


def build_plot_groups(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Split plot inputs into CPU-only, GPU-only, and comparison groups."""
    df = df.copy()
    df["gpu_mode"] = df["gpu_mode"].map(normalize_gpu_mode)
    if "environment_class" in df.columns:
        groups = []
        for environment, subset in df.groupby("environment_class", dropna=False):
            for name, group in build_plot_groups(subset.drop(columns="environment_class")):
                group["environment_class"] = environment
                groups.append((f"{environment}/{name}", group))
        return groups
    gpu_on = df["gpu_mode"].map(is_gpu_on)
    gpu_off = df["gpu_mode"].map(is_gpu_off)
    cpu_df = df[gpu_off].copy()
    gpu_df = df[gpu_on].copy()
    combined_df = (
        df[gpu_off | gpu_on].copy()
        if not cpu_df.empty and not gpu_df.empty
        else df.iloc[0:0].copy()
    )
    return list(zip(PLOT_OUTPUT_DIRS, (cpu_df, gpu_df, combined_df), strict=True))


def _with_pixel_metrics(df: pd.DataFrame, static_meta: dict) -> pd.DataFrame:
    """只使用 CSV 的显式像素数；不从历史计划补写缺失的测量字段。"""
    source_fields = list(dict.fromkeys([
        *PIXEL_COUNT_FIELDS,
        *(field for dependencies in PIXEL_RATE_SOURCES.values() for field in dependencies),
    ]))
    # Keep pandas scalar boxing, but only for inputs consumed by pixel metrics.
    # Missing columns stay NaN; even an all-missing projection retains its rows.
    inputs = df.loc[:, df.columns.isin(source_fields)].reindex(columns=source_fields)
    derived_rows = []
    for row in inputs.to_dict("records"):
        counts = {}
        for field in PIXEL_COUNT_FIELDS:
            try:
                number = float(row.get(field, "nan"))
            except (TypeError, ValueError, OverflowError):
                number = float("nan")
            counts[field] = number if math.isfinite(number) and number > 0 and number.is_integer() else float("nan")
        derived_rows.append({**counts, **pixel_rate_metrics({**row, **counts})})

    derived = pd.DataFrame(derived_rows, index=df.index,
                           columns=[*PIXEL_COUNT_FIELDS, *PIXEL_RATE_SOURCES])
    result = df.drop(columns=derived.columns, errors="ignore").join(derived)
    family = static_meta.get("task_family", "")
    result.attrs["pixel_scale_kind"] = {"diffusion": "output", "cv": "input", "multimodal": "input"}.get(family, "")
    result.attrs["input_scale_type"] = str(static_meta.get("input_scale_type") or "")
    return result


def normalized_metric_spec(
    df: pd.DataFrame, metric: str, title: str, ylabel: str, xlabel: str,
) -> tuple[str, str, str]:
    """分辨率图使用像素指标；缺少几何信息时显示 No data，不退回边长分母。"""
    sources = {
        "container_attributed_j_per_input_unit": ("container_attributed_j", "Estimated energy (J/Mpixel)"),
        "latency_s_per_input_unit": ("latency_s", "Latency (s/Mpixel)"),
        "latency_app_s_per_input_unit": ("latency_app_s", "Latency (s/Mpixel)"),
    }
    scale_type = df.attrs.get("input_scale_type") or xlabel
    if metric not in sources or scale_type not in {"resolution_px", "resolution_scale"}:
        return metric, title, ylabel
    direction = df.attrs.get("pixel_scale_kind")
    if direction not in {"input", "output"}:
        available = [
            kind for kind in ("input", "output")
            if f"{kind}_pixels_per_request" in df
            and pd.to_numeric(df[f"{kind}_pixels_per_request"], errors="coerce").gt(0).any()
        ]
        direction = available[0] if len(available) == 1 else "unknown"
    prefix, pixel_ylabel = sources[metric]
    unit_title = f"{direction.title()} Megapixel" if direction != "unknown" else "Megapixel (geometry unavailable)"
    return (
        f"{prefix}_per_{direction}_megapixel",
        title.replace("Input Unit", unit_title),
        pixel_ylabel if ylabel else "",
    )


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric = {
        name: pd.to_numeric(df[name], errors="coerce")
        for name in dict.fromkeys([*NUMERIC_FIELDS, "gpu_mem_total_bytes", *COMPUTE_NUMERIC_COLUMNS])
        if name in df.columns
    }
    # Construct together so same-dtype columns share blocks rather than splitting
    # the original frame once per assignment. Positions preserve duplicate labels
    # on unknown columns; ambiguous numeric labels still fail in to_numeric above.
    result = pd.DataFrame({
        position: numeric[name] if name in numeric else df.iloc[:, position]
        for position, name in enumerate(df.columns)
    }, index=df.index)
    result.columns = df.columns
    return result


def prepare_df(
    csv_path: str,
    *,
    only_ok: bool = True,
    exclude_warmup: bool = True,
) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find {csv_path}")

    from acprof.result_csv import open_result_text
    with open_result_text(csv_path) as stream:
        df = pd.read_csv(stream, skipinitialspace=True, float_precision="round_trip")
    df.columns = [str(col).strip() for col in df.columns]
    from acprof.result_csv import require_current_fields
    require_current_fields(df.columns)
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)

    if "gpu_mode" in df:
        df["gpu_mode"] = df["gpu_mode"].map(normalize_gpu_mode)

    static_meta = read_static_meta(csv_path)
    kinds = set(df.get("runtime_kind", pd.Series(dtype=str)).dropna()) - {"", "nan"}
    if len(kinds) > 1 or kinds and kinds != {static_meta.get("runtime_kind", "docker")}:
        raise ValueError("mixed or inconsistent execution runtime; split results by runtime")
    environment = recorded_identity(static_meta)["environment_class"]
    if "environment_class" not in df:
        df["environment_class"] = environment
    else:
        df["environment_class"] = df["environment_class"].where(
            df["environment_class"].isin(["native_linux", "wsl2", "vm", "cloud", "container_host"]), "unknown")
    if len(set(df["environment_class"])) > 1 or any(
        value not in {"unknown", environment} for value in df["environment_class"]
    ):
        raise ValueError("mixed or inconsistent environment_class; split results by recorded environment")
    if "gpu_mem_total_bytes" not in df.columns and static_meta.get("gpu_mem_total_bytes"):
        df["gpu_mem_total_bytes"] = pd.to_numeric(
            static_meta["gpu_mem_total_bytes"],
            errors="coerce",
        )

    df = _convert_numeric_columns(df)

    bytes_to_gib = {
        "container_mem_usage_avg_bytes": "container_mem_usage_avg_gib",
        "container_mem_usage_peak_bytes": "container_mem_usage_peak_gib",
        "container_mem_peak_cgroup_bytes": "container_mem_peak_cgroup_gib",
        "container_mem_anon_bytes_end": "container_mem_anon_end_gib",
        "container_mem_file_bytes_end": "container_mem_file_end_gib",
        "container_mem_slab_bytes_end": "container_mem_slab_end_gib",
        "container_swap_limit_bytes": "container_swap_limit_gib",
        "container_swap_usage_avg_bytes": "container_swap_usage_avg_gib",
        "container_swap_usage_peak_bytes": "container_swap_usage_peak_gib",
        "gpu_mem_used_avg_bytes": "gpu_mem_used_avg_gib",
        "gpu_mem_used_peak_bytes": "gpu_mem_used_peak_gib",
        "gpu_mem_total_bytes": "gpu_mem_total_gib",
        "cpu_heap_peak_total_bytes_massif": "cpu_heap_peak_total_gib_massif",
    }
    converted = {target: df[source] / float(BYTES_PER_GIB)
                 for source, target in bytes_to_gib.items() if source in df.columns}
    df = pd.concat([df.drop(columns=list(converted), errors="ignore"), pd.DataFrame(converted, index=df.index)], axis=1)

    df = _with_pixel_metrics(df, static_meta)

    if only_ok and "status" in df.columns:
        normalized_status = df["status"].astype(str).str.strip().str.lower()
        df = df[normalized_status == "ok"].copy()

    if exclude_warmup and "warmup" in df.columns:
        df = df[df["warmup"].fillna(0).astype(int) == 0].copy()

    need = {"input_scale", "cpu_cores", "mem_cap_gb", "gpu_mode"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["config"] = make_config_labels(df)
    df = df[df["input_scale"].notna() & (df["input_scale"] > 0)].copy()

    return df


def read_static_meta(csv_path: str) -> dict[str, object]:
    from acprof.artifact_layout import ArtifactLayout
    from acprof.artifacts import read_static_metadata
    return read_static_metadata(ArtifactLayout.from_csv(csv_path).root)


def aggregate_metric(df: pd.DataFrame, metric: str, *, agg_func: str = "mean") -> pd.DataFrame:
    group_cols = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale"]
    if "environment_class" in df:
        group_cols.insert(0, "environment_class")
    metric_df = df[group_cols + [metric]].copy()
    metric_df = metric_df[metric_df[metric].notna()].copy()
    if metric_df.empty:
        return metric_df

    if agg_func == "median":
        agg_df = metric_df.groupby(group_cols, as_index=False)[metric].median()
    else:
        agg_df = metric_df.groupby(group_cols, as_index=False)[metric].mean()

    agg_df["config"] = make_config_labels(agg_df)
    agg_df["gpu_on"] = agg_df["gpu_mode"].map(is_gpu_on)
    return agg_df


def aggregate_cold_start(df: pd.DataFrame, *, agg_func: str = "mean") -> pd.DataFrame:
    group_cols = ["cpu_cores", "mem_cap_gb", "gpu_mode"]
    if "environment_class" in df:
        group_cols.insert(0, "environment_class")
    metric_df = df[group_cols + ["cold_start_s"]].copy()
    metric_df = metric_df[metric_df["cold_start_s"].notna()].copy()
    if metric_df.empty:
        return metric_df

    if agg_func == "median":
        agg_df = metric_df.groupby(group_cols, as_index=False)["cold_start_s"].median()
    else:
        agg_df = metric_df.groupby(group_cols, as_index=False)["cold_start_s"].mean()

    agg_df["config"] = make_config_labels(agg_df)
    agg_df["gpu_on"] = agg_df["gpu_mode"].map(is_gpu_on)
    return agg_df
