"""Invalid scale identities fail before a plotting filter can hide records."""

import csv
import hashlib

import numpy as np
import pandas as pd
import pytest

from acprof.plotting.data import prepare_df


def _row(scale=1, **extra):
    return {"cpu_cores": 1.5, "mem_cap_gb": 2.25, "gpu_mode": "off", "input_scale": scale,
            "status": "ok", "warmup": 0, "latency_s": .1, **extra}


def _write(root, rows):
    target = root / "result_all.csv"
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(_row()))
        writer.writeheader()
        writer.writerows(rows)
    return target


@pytest.mark.parametrize("invalid", [0, -1, "", None, "NaN", "inf", "-inf", "1e400", "not-a-scale", "False"])
def test_invalid_scale_names_its_data_row_and_leaves_the_source_unchanged(tmp_path, invalid):
    path = _write(tmp_path, [_row(), _row(invalid)])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="input_scale.*finite and positive.*data row 2"):
        prepare_df(str(path))

    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("invalid", [True, False])
def test_boolean_scale_is_not_a_numeric_coordinate(tmp_path, invalid):
    path = _write(tmp_path, [_row(invalid)])

    with pytest.raises(ValueError, match="input_scale.*data row 1"):
        prepare_df(str(path))


@pytest.mark.parametrize("extra", [{"status": "error"}, {"warmup": 1}])
def test_excluded_rows_cannot_hide_an_invalid_scale(tmp_path, extra):
    path = _write(tmp_path, [_row(), _row("inf", **extra)])

    with pytest.raises(ValueError, match="input_scale.*data row 2"):
        prepare_df(str(path), only_ok=True, exclude_warmup=True)


def test_positive_fractional_and_adjacent_float_scales_keep_exact_identity(tmp_path):
    scales = [.125, 1., np.nextafter(1., 2.), np.nextafter(np.nextafter(1., 2.), 2.), 1e-8]
    path = _write(tmp_path, [_row(repr(float(scale))) for scale in scales])
    original = path.read_bytes()

    frame = prepare_df(str(path))

    assert frame["input_scale"].tolist() == scales
    assert frame.index.tolist() == list(range(len(scales)))
    assert path.read_bytes() == original


def test_validation_preserves_integer_dtype_and_precision(tmp_path):
    scales = [2**63 + 7, 2**63 + 9]
    path = _write(tmp_path, [_row(scale) for scale in scales])

    frame = prepare_df(str(path))

    assert pd.api.types.is_integer_dtype(frame["input_scale"])
    assert frame["input_scale"].tolist() == scales


def test_missing_metric_is_allowed_with_a_valid_scale(tmp_path):
    path = _write(tmp_path, [_row(latency_s="not measured")])

    frame = prepare_df(str(path))

    assert frame["input_scale"].tolist() == [1]
    assert frame["latency_s"].isna().all()


def test_header_only_csv_keeps_an_empty_plot_frame(tmp_path):
    frame = prepare_df(str(_write(tmp_path, [])))

    assert frame.empty
    assert {"input_scale", "config"} <= set(frame.columns)


@pytest.mark.parametrize("invalid", [0, "inf"])
def test_plot_entrypoint_rejects_invalid_scales_before_creating_artifacts(tmp_path, invalid):
    from acprof.cli.plot import main

    path = _write(tmp_path, [_row(), _row(invalid)])

    with pytest.raises(ValueError, match="input_scale.*data row 2"):
        main([str(path)])

    assert list(tmp_path.iterdir()) == [path]
