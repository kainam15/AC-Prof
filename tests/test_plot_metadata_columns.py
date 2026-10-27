"""Metadata fallback must preserve wide CSV identity without fragmented inserts."""

import csv
import json

import pandas as pd
import pytest

from acprof.config import STATIC_META_SCHEMA_VERSION
from acprof.plotting.data import prepare_df


def write_input(tmp_path, rows, metadata, *, fields=None):
    path = tmp_path / "result_all.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if metadata is not None:
        (tmp_path / "static_meta.json").write_text(json.dumps({
            "schema_version": STATIC_META_SCHEMA_VERSION, **metadata,
        }), encoding="utf-8")
    return path


def native_metadata() -> dict[str, object]:
    return {
        "environment_class": "native_linux", "runtime_kind": "docker",
        "collection_tier": "full", "comparability_class": "native_linux",
        "platform": {"environment": "native_linux", "native": True},
        "gpu_mem_total_bytes": 8 * 1024**3,
    }


@pytest.mark.parametrize("existing_environment", [False, True])
@pytest.mark.parametrize("metadata_kind", ["native", "absent", "unknown"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.filterwarnings("error::pandas.errors.PerformanceWarning")
def test_wide_metadata_fallback_preserves_rows_columns_and_sources(
    tmp_path, existing_environment, metadata_kind, empty,
):
    metadata = native_metadata() if metadata_kind == "native" else None
    if metadata_kind == "unknown":
        metadata = {"gpu_mem_total_bytes": 8 * 1024**3}
    environment = "native_linux" if metadata_kind == "native" else "unknown"
    rows = []
    for index in range(4):
        row = {
            "cpu_cores": 1.5, "mem_cap_gb": 2.25, "gpu_mode": " off ",
            "input_scale": index + 1, "status": "error" if index == 1 else " ok ",
            "warmup": int(index == 2), "runtime_kind": " docker ",
            **{f"extension_{column}": index + column for column in range(128)},
        }
        if existing_environment:
            row["environment_class"] = f" {environment} "
        rows.append(row)
    fields = list(rows[0])
    path = write_input(tmp_path, [] if empty else rows, metadata, fields=fields)
    originals = {file.name: file.read_bytes() for file in tmp_path.iterdir()}

    frame = prepare_df(str(path))

    expected_index = [] if empty else [0, 3]
    assert frame.index.tolist() == expected_index
    assert frame.columns[:len(fields)].tolist() == fields
    additions = [] if existing_environment else ["environment_class"]
    if metadata is not None:
        additions.append("gpu_mem_total_bytes")
        assert frame["gpu_mem_total_bytes"].dtype == "int64"
        assert frame["gpu_mem_total_gib"].tolist() == [8.0] * len(expected_index)
    else:
        assert "gpu_mem_total_bytes" not in frame.columns
    assert frame.columns[len(fields):len(fields) + len(additions)].tolist() == additions
    assert frame["environment_class"].tolist() == [environment] * len(expected_index)
    assert frame["config"].tolist() == [f"{environment} / CPU+CPU1.5+Mem2.25"] * len(expected_index)
    for column in range(128):
        assert frame[f"extension_{column}"].tolist() == [index + column for index in expected_index]
    assert {file.name: file.read_bytes() for file in tmp_path.iterdir()} == originals


@pytest.mark.parametrize("capacity", [0, None, "", "bad", "18446744073709551615"])
def test_capacity_fallback_preserves_missing_and_large_integer_semantics(tmp_path, capacity):
    row = {"cpu_cores": 1, "mem_cap_gb": 2, "gpu_mode": "off", "input_scale": 1}
    metadata = native_metadata()
    metadata["gpu_mem_total_bytes"] = capacity
    frame = prepare_df(str(write_input(tmp_path, [row], metadata)))
    if not capacity:
        assert "gpu_mem_total_bytes" not in frame.columns
        assert "gpu_mem_total_gib" not in frame.columns
    elif capacity == "bad":
        assert pd.isna(frame["gpu_mem_total_bytes"].iloc[0])
        assert pd.isna(frame["gpu_mem_total_gib"].iloc[0])
    else:
        assert frame["gpu_mem_total_bytes"].dtype == "uint64"
        assert frame["gpu_mem_total_bytes"].iloc[0] == 2**64 - 1


def test_recorded_capacity_is_not_replaced_by_metadata(tmp_path):
    row = {"cpu_cores": 1, "mem_cap_gb": 2, "gpu_mode": "off", "input_scale": 1,
           "gpu_mem_total_bytes": "NA"}
    frame = prepare_df(str(write_input(tmp_path, [row], native_metadata())))
    assert pd.isna(frame["gpu_mem_total_bytes"].iloc[0])
    assert pd.isna(frame["gpu_mem_total_gib"].iloc[0])


@pytest.mark.parametrize(("gpu", "runtime", "environment", "error"), [
    ("invalid", "process", "wsl2", "gpu_mode"),
    ("off", "process", "wsl2", "execution runtime"),
    ("off", "docker", "wsl2", "environment_class"),
])
@pytest.mark.filterwarnings("error::pandas.errors.PerformanceWarning")
def test_invalid_identity_is_rejected_before_metadata_and_row_filtering(
    tmp_path, gpu, runtime, environment, error,
):
    row = {
        "cpu_cores": 1, "mem_cap_gb": 2, "gpu_mode": gpu, "input_scale": 1,
        "runtime_kind": runtime, "environment_class": environment, "status": "error",
        "warmup": 1, "latency_s": 7, "latency_s ": 11,
        **{f"extension_{column}": column for column in range(128)},
    }
    metadata = native_metadata()
    metadata["gpu_mem_total_bytes"] = {"invalid": "capacity"}
    path = write_input(tmp_path, [row], metadata)
    with pytest.raises(ValueError, match=error):
        prepare_df(str(path))


@pytest.mark.parametrize("environments", [["native_linux", "unknown"], ["native_linux", "wsl2"]])
def test_mixed_environment_is_rejected_even_on_filtered_rows(tmp_path, environments):
    rows = [{"cpu_cores": 1, "mem_cap_gb": 2, "gpu_mode": "off", "input_scale": 1,
             "environment_class": environment, "status": "error"} for environment in environments]
    path = write_input(tmp_path, rows, native_metadata())
    with pytest.raises(ValueError, match="environment_class"):
        prepare_df(str(path))
