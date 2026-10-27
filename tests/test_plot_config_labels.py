"""Configuration labels preserve identity without boxing an entire wide frame."""

import gc
import hashlib
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from acprof.plotting.data import (
    aggregate_cold_start,
    aggregate_metric,
    make_config_labels,
    prepare_df,
)


def test_wide_plot_preparation_avoids_a_full_object_matrix_for_labels(tmp_path):
    count = 2048
    frame = pd.DataFrame({
        "cpu_cores": np.full(count, 1.5), "mem_cap_gb": np.full(count, 2.25),
        "gpu_mode": ["off"] * count, "environment_class": ["unknown"] * count,
        "input_scale": np.arange(count) + 1,
        **{f"unrelated_{i}": np.arange(count, dtype=float) + i for i in range(256)},
    })
    path = tmp_path / "result_all.csv"
    frame.to_csv(path, index=False)
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    del frame
    prepare_df(str(path), only_ok=False, exclude_warmup=False)
    gc.collect()

    tracemalloc.start()
    try:
        result = prepare_df(str(path), only_ok=False, exclude_warmup=False)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    retained = int(result.memory_usage(deep=True).sum())
    assert result["config"].tolist() == ["unknown / CPU+CPU1.5+Mem2.25"] * count
    assert result["unrelated_255"].tolist() == (np.arange(count) + 255).tolist()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original
    # Allow the numeric input/output arrays and normal parser buffers, but not
    # an additional object matrix boxing every unrelated numeric cell.
    assert peak <= 4 * retained + 1_000_000, (peak, retained)


def test_prepared_labels_keep_fractional_and_inherited_coordinates(tmp_path):
    frame = pd.DataFrame({
        "cpu_cores": [1.0000000000000002, float("nan"), 3.5],
        "mem_cap_gb": [1e-8, float("nan"), 2.25],
        "gpu_mode": [" YES ", "false", "1.0"], "input_scale": [1, 2, 3],
        "config": ["stale"] * 3, "unrelated": ["a", "b", "c"],
    })
    path = tmp_path / "external.csv"
    frame.to_csv(path, index=False)

    result = prepare_df(str(path))

    assert result["config"].tolist() == [
        "unknown / GPU+CPU1.0000000000000002+Mem1e-08",
        "unknown / CPU+CPUinherited+Meminherited",
        "unknown / GPU+CPU3.5+Mem2.25",
    ]
    assert result.columns[:len(frame.columns)].tolist() == frame.columns.tolist()
    assert result["unrelated"].tolist() == ["a", "b", "c"]
    assert result["input_scale"].tolist() == [1, 2, 3]


@pytest.mark.parametrize("with_environment", [False, True])
def test_batch_labels_preserve_duplicate_index_and_optional_environment(with_environment):
    frame = pd.DataFrame({
        "cpu_cores": [1.5, 2.5, float("nan")], "mem_cap_gb": [4.25, 8.25, float("nan")],
        "gpu_mode": ["on", "off", "off"], "metric": [1, 2, 3],
    }, index=pd.Index([9, 2, 9], name="row_id"))
    labels = ["GPU+CPU1.5+Mem4.25", "CPU+CPU2.5+Mem8.25", "CPU+CPUinherited+Meminherited"]
    if with_environment:
        frame["environment_class"] = ["native_linux", "wsl2", "unknown"]
        labels = [f"{environment} / {label}" for environment, label in
                  zip(["native_linux", "wsl2", "unknown"], labels, strict=True)]
    original = frame.copy(deep=True)

    result = make_config_labels(frame)

    pd.testing.assert_series_equal(result, pd.Series(labels, index=frame.index))
    pd.testing.assert_frame_equal(frame, original)


def test_empty_batch_labels_keep_the_object_dtype_and_index():
    index = pd.Index([], name="empty_rows")
    result = make_config_labels(pd.DataFrame(index=index))
    pd.testing.assert_series_equal(result, pd.Series(index=index, dtype="object"))


@pytest.mark.parametrize("with_environment", [False, True])
@pytest.mark.parametrize("agg_func", ["mean", "median"])
@pytest.mark.parametrize("cold_start", [False, True])
def test_aggregate_labels_keep_grouping_and_numeric_results(with_environment, agg_func, cold_start):
    metric = "cold_start_s" if cold_start else "latency_s"
    frame = pd.DataFrame({
        "cpu_cores": [1.5] * 3, "mem_cap_gb": [2.25] * 3, "gpu_mode": ["on"] * 3,
        "input_scale": [64, 64, 128], metric: [2.0, 4.0, 8.0],
    })
    if with_environment:
        frame["environment_class"] = ["native_linux", "native_linux", "wsl2"]
    original = frame.copy(deep=True)

    result = (aggregate_cold_start(frame, agg_func=agg_func) if cold_start
              else aggregate_metric(frame, metric, agg_func=agg_func))

    if with_environment:
        assert result["environment_class"].tolist() == ["native_linux", "wsl2"]
        assert result["config"].tolist() == [
            "native_linux / GPU+CPU1.5+Mem2.25", "wsl2 / GPU+CPU1.5+Mem2.25",
        ]
        assert result[metric].tolist() == [3.0, 8.0]
    else:
        expected_values = ([14 / 3 if agg_func == "mean" else 4.0] if cold_start else [3.0, 8.0])
        assert result[metric].tolist() == expected_values
        assert result["config"].tolist() == ["GPU+CPU1.5+Mem2.25"] * len(expected_values)
    assert result["gpu_on"].tolist() == [True] * len(result)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("cold_start", [False, True])
def test_empty_aggregates_keep_the_existing_column_contract(empty, cold_start):
    metric = "cold_start_s" if cold_start else "latency_s"
    columns = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", metric]
    frame = pd.DataFrame([[1.5, 2.25, "off", 64, float("nan")]], columns=columns)
    if empty:
        frame = frame.iloc[:0]

    result = aggregate_cold_start(frame) if cold_start else aggregate_metric(frame, metric)

    assert result.empty
    assert result.columns.tolist() == (["cpu_cores", "mem_cap_gb", "gpu_mode", metric] if cold_start else columns)
    assert "config" not in result.columns


@pytest.mark.parametrize("cold_start", [False, True])
def test_aggregate_labels_reject_unknown_gpu_mode(cold_start):
    frame = pd.DataFrame({"cpu_cores": [1.5], "mem_cap_gb": [2.25], "gpu_mode": ["mystery"],
                          "input_scale": [64], "latency_s": [1.0], "cold_start_s": [1.0]})
    if cold_start:
        with pytest.raises(ValueError, match="gpu_mode"):
            aggregate_cold_start(frame)
    else:
        with pytest.raises(ValueError, match="gpu_mode"):
            aggregate_metric(frame, "latency_s")


@pytest.mark.parametrize("field", ["cpu_cores", "mem_cap_gb"])
@pytest.mark.parametrize("invalid", [0, -1, float("inf")])
def test_prepared_labels_reject_invalid_resource_coordinates(tmp_path, field, invalid):
    row = {"cpu_cores": 1.5, "mem_cap_gb": 2.25, "gpu_mode": "off", "input_scale": 1}
    row[field] = invalid
    path = tmp_path / "invalid.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="resource coordinate must be finite and positive"):
        prepare_df(str(path))
