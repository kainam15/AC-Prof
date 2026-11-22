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


@pytest.mark.filterwarnings("error:.*float_precision.*")
def test_csv_plot_statistics_keep_adjacent_scales_in_separate_windows(tmp_path):
    from acprof.plotting.uncertainty import aggregate_plot_metric

    scales = [0.12345678901234565, 0.12345678901234566, 0.12345678901234568]
    path = tmp_path / "result_all.csv"
    rows = [_row(repr(scale), repeat_idx=0, latency_app_s=repr(scale)) for scale in scales]
    pd.DataFrame(rows).to_csv(path, index=False)

    summary = aggregate_plot_metric(prepare_df(str(path)), "latency_app_s")

    assert summary["input_scale"].tolist() == scales
    assert summary["latency_app_s"].tolist() == scales
    assert summary["n_windows"].tolist() == [1, 1, 1]


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
@pytest.mark.parametrize("from_directory", [True, False], ids=["experiment", "file"])
def test_plot_entrypoint_rejects_invalid_scales_before_creating_artifacts(tmp_path, invalid, from_directory):
    from acprof.cli.plot import main

    path = _write(tmp_path, [_row(), _row(invalid)])

    with pytest.raises(ValueError, match="input_scale.*data row 2"):
        main([str(tmp_path if from_directory else path)])

    assert list(tmp_path.iterdir()) == [path]


def test_default_plot_directory_uses_the_same_result_selection(tmp_path, monkeypatch):
    from acprof.cli.plot import main

    directory = tmp_path / "results"
    directory.mkdir()
    _write(directory, [_row(), _row("inf")])
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="input_scale.*data row 2"):
        main([])
