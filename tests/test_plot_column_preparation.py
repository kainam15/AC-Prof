"""Wide plot inputs retain numeric identity without fragmented column insertion."""

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from acprof.config import CSV_FIELDS, STATIC_META_SCHEMA_VERSION
from acprof.metric_registry import NUMERIC_FIELDS
from acprof.plotting.data import prepare_df


def write_plot_csv(
    root: Path, rows: list[dict], *, fields: list[str] | None = None,
) -> Path:
    path = root / "result_all.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields if fields is not None else list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    (root / "static_meta.json").write_text(json.dumps({
        "schema_version": STATIC_META_SCHEMA_VERSION,
        "environment_class": "native_linux", "runtime_kind": "docker",
        "collection_tier": "full", "comparability_class": "native_linux",
        "platform": {"environment": "native_linux", "native": True},
        "task_family": "cv", "input_scale_type": "resolution_px",
    }), encoding="utf-8")
    return path


@pytest.mark.parametrize("only_ok", [False, True])
@pytest.mark.parametrize("exclude_warmup", [False, True])
@pytest.mark.filterwarnings("error::pandas.errors.PerformanceWarning")
def test_wide_plot_input_preserves_filters_without_fragmentation(
    tmp_path, only_ok, exclude_warmup,
):
    rows = []
    for index, (status, warmup, scale) in enumerate([
        ("ok", 0, 64), ("error", 0, 128), ("ok", 1, 256), ("ok", 0, 0),
    ]):
        row: dict[str, object] = dict.fromkeys(NUMERIC_FIELDS, 1)
        row.update(environment_class="native_linux", runtime_kind="docker",
                   cpu_cores=1.5, mem_cap_gb=2.25, gpu_mode="off",
                   status=status, warmup=warmup, input_scale=scale,
                   case_id=f"case-{index}", input_pixels_per_request=1_000_000,
                   output_pixels_per_request=2_000_000, latency_s=0.25)
        rows.append(row)
    path = write_plot_csv(tmp_path, rows)
    original = path.read_bytes()

    frame = prepare_df(str(path), only_ok=only_ok, exclude_warmup=exclude_warmup)

    expected = [0]
    if not only_ok:
        expected.append(1)
    if not exclude_warmup:
        expected.append(2)
    assert frame.index.tolist() == expected
    assert frame["case_id"].tolist() == [f"case-{index}" for index in expected]
    assert frame["config"].tolist() == ["native_linux / CPU+CPU1.5+Mem2.25"] * len(expected)
    assert frame["latency_s_per_input_megapixel"].tolist() == [0.25] * len(expected)
    assert frame["latency_s_per_output_megapixel"].tolist() == [0.125] * len(expected)
    assert frame.attrs == {"pixel_scale_kind": "input", "input_scale_type": "resolution_px"}
    assert path.read_bytes() == original


def test_numeric_conversion_preserves_integer_precision_and_unknown_dtypes(tmp_path):
    row = {
        "cpu_cores": "1.5", "mem_cap_gb": "2.25", "gpu_mode": " off ",
        "input_scale": "1.0000000000000002", "status": " ok ", "warmup": "0",
        "container_mem_pgfault_delta": str(2**53 + 1),
        "container_mem_pgmajfault_delta": str(2**64 - 1),
        "latency_s": "0.12345678901234566", "cpu_ipc": " invalid ",
        "extension_bool": "True", "extension_text": " 中文,\"quote\"\nline ",
        "extension_missing": "NA",
    }
    path = write_plot_csv(tmp_path, [row], fields=list(row))
    frame = prepare_df(str(path))

    assert frame.columns[:len(row)].tolist() == list(row)
    assert frame["container_mem_pgfault_delta"].dtype == "int64"
    assert frame["container_mem_pgfault_delta"].iloc[0] == 2**53 + 1
    assert frame["container_mem_pgmajfault_delta"].dtype == "uint64"
    assert frame["container_mem_pgmajfault_delta"].iloc[0] == 2**64 - 1
    assert frame["input_scale"].iloc[0] == 1.0000000000000002
    assert frame["latency_s"].iloc[0] == 0.12345678901234566
    assert pd.isna(frame["cpu_ipc"].iloc[0])
    assert frame["extension_bool"].dtype == "bool"
    assert bool(frame["extension_bool"].iloc[0])
    assert frame["extension_text"].iloc[0] == '中文,"quote"\nline'
    assert pd.isna(frame["extension_missing"].iloc[0])


@pytest.mark.parametrize("all_filtered", [False, True])
@pytest.mark.filterwarnings("error::pandas.errors.PerformanceWarning")
def test_empty_wide_plot_input_retains_fields_and_attributes(tmp_path, all_filtered):
    row = {"cpu_cores": 1, "mem_cap_gb": 2, "gpu_mode": "off", "input_scale": 1,
           "status": "error", "warmup": 0}
    path = write_plot_csv(tmp_path, [row] if all_filtered else [])
    frame = prepare_df(str(path), only_ok=True, exclude_warmup=False)
    assert frame.empty
    assert set(CSV_FIELDS).issubset(frame.columns)
    assert frame["config"].dtype == "object"
    assert frame.attrs == {"pixel_scale_kind": "input", "input_scale_type": "resolution_px"}


def test_existing_derived_fields_are_replaced_in_the_original_order(tmp_path):
    row = {
        "cpu_cores": 1, "mem_cap_gb": 2, "gpu_mode": "off", "input_scale": 64,
        "gpu_mem_used_avg_gib": -1, "gpu_mem_used_avg_bytes": 3 * 1024**3,
        "input_pixels_per_request": 1_000_000, "latency_s": 0.25,
        "latency_s_per_input_megapixel": -1, "extension": "preserved",
    }
    path = write_plot_csv(tmp_path, [row], fields=list(row))
    metadata_path = tmp_path / "static_meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["gpu_mem_total_bytes"] = 8 * 1024**3
    metadata_path.write_text(json.dumps(metadata))
    frame = prepare_df(str(path))

    assert frame.columns.tolist() == [
        "cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "gpu_mem_used_avg_bytes",
        "latency_s", "extension", "environment_class", "gpu_mem_total_bytes",
        "gpu_mem_used_avg_gib", "gpu_mem_total_gib", "input_pixels_per_request",
        "output_pixels_per_request", "container_attributed_j_per_input_megapixel",
        "latency_s_per_input_megapixel", "latency_app_s_per_input_megapixel",
        "container_attributed_j_per_output_megapixel", "latency_s_per_output_megapixel",
        "latency_app_s_per_output_megapixel", "config",
    ]
    assert frame["gpu_mem_used_avg_gib"].iloc[0] == 3
    assert frame["gpu_mem_total_gib"].iloc[0] == 8
    assert frame["latency_s_per_input_megapixel"].iloc[0] == 0.25
    assert pd.isna(frame["output_pixels_per_request"].iloc[0])


@pytest.mark.parametrize("column", ["extension", "latency_s"])
@pytest.mark.filterwarnings("error:DataFrame columns are not unique:UserWarning")
def test_padded_duplicate_labels_are_not_silently_dropped(tmp_path, column):
    row = {"cpu_cores": 1, "mem_cap_gb": 2, "gpu_mode": "off", "input_scale": 1,
           column: 7, f"{column} ": 11}
    path = write_plot_csv(tmp_path, [row], fields=list(row))
    if column == "latency_s":
        with pytest.raises(TypeError, match="1-d array, or Series"):
            prepare_df(str(path))
    else:
        frame = prepare_df(str(path))
        assert frame.columns.tolist().count("extension") == 2
        assert frame.loc[:, "extension"].iloc[0].tolist() == [7, 11]
