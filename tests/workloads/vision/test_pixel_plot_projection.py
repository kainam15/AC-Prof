"""Pixel preparation must not expand unrelated wide-table cells into row maps."""

import gc
import tracemalloc
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from acprof.pixel_metrics import PIXEL_COUNT_FIELDS, PIXEL_RATE_SOURCES
from acprof.plotting.data import _with_pixel_metrics


def pixel_peak(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    gc.collect()
    tracemalloc.start()
    try:
        result = _with_pixel_metrics(frame, {"task_family": "cv"})
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def test_pixel_preparation_memory_does_not_expand_unrelated_cells():
    count = 1024
    narrow = pd.DataFrame({
        "input_pixels_per_request": [1_000_000] * count,
        "output_pixels_per_request": [2_000_000] * count,
        "latency_s": [0.25] * count,
        "latency_app_s": [0.125] * count,
        "container_attributed_energy_eff_j": [2.5] * count,
    })
    unrelated = pd.DataFrame({
        f"unrelated_{index}": np.arange(count, dtype=float) + index
        for index in range(256)
    })
    wide = narrow.join(unrelated)
    original = wide.copy(deep=True)
    _with_pixel_metrics(narrow, {})  # Warm pandas caches before comparing allocations.

    small_result, small_peak = pixel_peak(narrow)
    result, wide_peak = pixel_peak(wide)

    pd.testing.assert_frame_equal(result.loc[:, small_result.columns], small_result)
    pd.testing.assert_frame_equal(result.loc[:, unrelated.columns], unrelated)
    pd.testing.assert_frame_equal(wide, original)
    assert wide_peak <= 2 * small_peak + 1_000_000, (small_peak, wide_peak)


@pytest.mark.parametrize("count", [0, 2])
def test_all_missing_pixel_inputs_preserve_rows_and_nan_dtypes(count):
    index = pd.Index(["first", "last"][:count], name="sample")
    frame = pd.DataFrame([[7, 11]] * count, index=index, columns=["unrelated", "unrelated"])
    original = frame.copy(deep=True)

    result = _with_pixel_metrics(frame, {"task_family": "diffusion", "input_scale_type": "resolution_px"})

    assert result.index.equals(index)
    assert result.shape == (count, 10)
    assert result.iloc[:, :2].equals(frame)
    derived = result.iloc[:, 2:]
    assert derived.isna().all().all()
    assert all(dtype == ("float64" if count else "object") for dtype in derived.dtypes)
    assert result.attrs == {"pixel_scale_kind": "output", "input_scale_type": "resolution_px"}
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("dtype", ["object", "Float64", "Int64", "boolean", "category"])
def test_projection_retains_pandas_scalar_boxing(dtype):
    frame = pd.DataFrame({
        "input_pixels_per_request": pd.Series([1, None], dtype=dtype),
        "container_attributed_energy_eff_j": pd.Series([1, None], dtype=dtype),
        "unrelated": ["text", "other"],
    })
    original = frame.copy(deep=True)

    result = _with_pixel_metrics(frame, {})

    assert result["input_pixels_per_request"].iloc[0] == 1.0
    assert pd.isna(result["input_pixels_per_request"].iloc[1])
    energy = result["container_attributed_j_per_input_megapixel"]
    if dtype == "boolean":
        assert pd.isna(energy.iloc[0])  # A boolean numerator is unavailable, never 1 J.
    else:
        assert energy.iloc[0] == 1_000_000.0
    assert pd.isna(energy.iloc[1])
    assert result["output_pixels_per_request"].isna().all()
    assert "latency_s" not in result.columns
    pd.testing.assert_frame_equal(frame, original)


def test_object_numpy_booleans_remain_invalid_energy_values():
    frame = pd.DataFrame({
        "input_pixels_per_request": [1_000_000, 1_000_000],
        "container_attributed_energy_eff_j": pd.Series([np.bool_(True), np.bool_(False)], dtype=object),
    })
    result = _with_pixel_metrics(frame, {})
    assert result["container_attributed_j_per_input_megapixel"].isna().all()


@pytest.mark.parametrize("count", [None, float("nan"), float("inf"), -1, 0, 0.5, "invalid"])
def test_invalid_pixel_counts_never_become_a_fallback_denominator(count):
    frame = pd.DataFrame({
        "input_pixels_per_request": [count], "output_pixels_per_request": [2_000_000],
        "latency_s": [0.25], "latency_app_s": [0.5], "container_attributed_energy_eff_j": [0],
        "input_scale": [64], "latency_s_per_input_megapixel": [999],
    })
    result = _with_pixel_metrics(frame, {"task_family": "cv"})
    assert pd.isna(result["input_pixels_per_request"].iloc[0])
    assert pd.isna(result["latency_s_per_input_megapixel"].iloc[0])
    assert pd.isna(result["container_attributed_j_per_input_megapixel"].iloc[0])
    assert result["latency_s_per_output_megapixel"].iloc[0] == 0.125
    assert result["latency_app_s_per_output_megapixel"].iloc[0] == 0.25
    assert result["container_attributed_j_per_output_megapixel"].iloc[0] == 0.0


def test_projection_uses_the_metric_dependency_mapping():
    frame = pd.DataFrame({"additional_source": [3], "additional_count": [2_000_000]})
    with patch.dict(PIXEL_RATE_SOURCES, {"additional_rate": ("additional_source", "additional_count")}):
        result = _with_pixel_metrics(frame, {})
    assert result["additional_rate"].iloc[0] == 1.5
    assert result.loc[:, list(PIXEL_COUNT_FIELDS)].isna().all().all()
