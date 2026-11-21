"""Scalar metric curves preserve gaps in recorded, eligible input scales."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from acprof.plotting.data import aggregate_metric
from acprof.plotting.metrics import plot_metric, plot_metric_overview
from acprof.plotting.process import plot_process_results


def _row(scale, value, **extra):
    return {"cpu_cores": 1, "mem_cap_gb": 4, "gpu_mode": "off", "input_scale": scale,
            "latency_app_s": value, **extra}


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
@pytest.mark.parametrize("aggregation", ["mean", "median"])
def test_aggregate_retains_recorded_gaps_and_uses_only_finite_windows(dtype, aggregation):
    frame = pd.DataFrame([
        _row(scale, value) for scale, value in [
            (1, np.nan), (2, 2.), (2, 4.), (2, np.nan), (3, np.nan),
            (4, 10.), (4, 12.), (4, 100.), (5, np.inf), (5, -np.inf),
            (6, np.inf), (6, 8.), (7, -2.), (8, 0.),
        ]
    ])
    frame = frame.rename(columns={"latency_app_s": "signed_metric"})
    frame["signed_metric"] = frame["signed_metric"].astype(dtype)
    frame["unrelated"] = [{"preserved": True}] * len(frame)
    frame.index = [7] * len(frame)
    before = frame.copy(deep=True)

    result = aggregate_metric(frame, "signed_metric", agg_func=aggregation)

    assert result["input_scale"].tolist() == list(range(1, 9))
    middle = 122. / 3 if aggregation == "mean" else 12.
    np.testing.assert_allclose(result["signed_metric"].to_numpy(dtype=float, na_value=np.nan),
                               [np.nan, 3., np.nan, middle, np.nan, 8., -2., 0.], equal_nan=True)
    assert result["config"].tolist() == ["CPU+CPU1+Mem4"] * 8
    assert "unrelated" not in result
    pd.testing.assert_frame_equal(frame, before)


def test_aggregate_drops_empty_configurations_without_inventing_unrecorded_scales():
    frame = pd.DataFrame([
        _row(1, 2., environment_class="native_linux"),
        _row(3, 4., environment_class="native_linux"),
        _row(2, np.nan, cpu_cores=2, environment_class="native_linux"),
        _row(4, np.inf, cpu_cores=2, environment_class="native_linux"),
        _row(2, np.nan, environment_class="wsl2"),
    ])

    result = aggregate_metric(frame, "latency_app_s")

    assert result["input_scale"].tolist() == [1, 3]
    assert result["cpu_cores"].tolist() == [1, 1]
    assert result["environment_class"].tolist() == ["native_linux"] * 2


@pytest.mark.parametrize("values", [[np.inf], [-np.inf], [np.nan, np.inf, -np.inf], [None], [pd.NA]])
def test_no_finite_observation_keeps_an_empty_aggregate(values):
    frame = pd.DataFrame([_row(index + 1, value) for index, value in enumerate(values)])

    result = aggregate_metric(frame, "latency_app_s")

    assert result.empty
    assert result.columns.tolist() == ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "latency_app_s"]


@pytest.fixture
def figures(monkeypatch):
    captured = []
    close = plt.close

    def record(figure=None):
        candidate = plt.gcf() if figure is None and plt.get_fignums() else figure
        if isinstance(candidate, Figure):
            captured.append(candidate)
        close(figure)

    monkeypatch.setattr(plt, "close", record)
    return captured


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_single_metric_line_keeps_a_missing_scale_between_valid_markers(dtype, figures):
    frame = pd.DataFrame([_row(3, 0.), _row(2, np.nan), _row(1, 2.)])
    frame["latency_app_s"] = frame["latency_app_s"].astype(dtype)

    plot_metric(frame, "latency_app_s", "Latency", "s", "input_scale", None)

    line, = figures[-1].axes[0].lines
    np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3])
    np.testing.assert_allclose(line.get_ydata(), [2., np.nan, 0.], equal_nan=True)
    assert line.get_marker() == "o"


def test_overview_shows_nonfinite_panel_as_no_data_and_excludes_empty_config(figures):
    frame = pd.DataFrame([
        _row(scale, value, throughput_samples_per_s=np.inf)
        for scale, value in [(1, 2.), (2, np.nan), (3, 4.)]
    ] + [_row(2, np.nan, cpu_cores=2, throughput_samples_per_s=-np.inf)])

    plot_metric_overview(frame, panels=(("latency_app_s", "Latency", "s"),
                                       ("throughput_samples_per_s", "Throughput", "samples/s")),
                         rows=1, columns=2, shared_y_groups=(), title="Overview", xlabel="input_scale", out_png=None)

    figure = figures[-1]
    latency, throughput = figure.axes
    line, = latency.lines
    np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3])
    np.testing.assert_allclose(line.get_ydata(), [2., np.nan, 4.], equal_nan=True)
    assert not throughput.lines
    assert [text.get_text() for text in throughput.texts] == ["No data"]
    assert "CPU2" not in [text.get_text() for text in figure.legends[0].get_texts()]


def test_process_curves_keep_inherited_labels_gaps_and_unavailable_panels(tmp_path, figures):
    frame = pd.DataFrame([
        _row(scale, value, cpu_cores=np.nan, mem_cap_gb=np.nan,
             config=label, process_cpu_util_avg_pct=np.inf)
        for label in ["z / CPU+CPUinherited+Meminherited", "a / CPU+CPUinherited+Meminherited"]
        for scale, value in [(3, 4.), (2, np.nan), (1, 2.)]
    ])
    before = frame.copy(deep=True)

    target = plot_process_results(frame, tmp_path)

    figure = figures[-1]
    assert target.is_file()
    assert [line.get_label() for line in figure.axes[0].lines] == list(frame["config"].unique())
    for line in figure.axes[0].lines:
        np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3])
        np.testing.assert_allclose(line.get_ydata(), [2., np.nan, 4.], equal_nan=True)
    assert not figure.axes[2].lines
    assert [text.get_text() for text in figure.axes[2].texts] == ["Unavailable / not requested"]
    pd.testing.assert_frame_equal(frame, before)


def test_plot_refuses_mixed_run_windows_even_after_caching_the_same_values(tmp_path):
    from acprof.plotting.uncertainty import WindowStatistics, aggregate_plot_metric

    frame = pd.DataFrame([
        _row(1, latency, repeat_idx=i, warmup=0, status="ok", run_id="run-A")
        for i, latency in enumerate((1, 1, 10))
    ])
    cache = WindowStatistics()
    good = aggregate_plot_metric(frame, "latency_app_s", statistics=cache)
    assert good["n_windows"].tolist() == [3]
    assert good["latency_app_s"].tolist() == [4]

    frame.loc[2, "run_id"] = "run-B"
    output = tmp_path / "mixed.png"
    with pytest.raises(ValueError, match="inconsistent run_id"):
        plot_metric(frame, "latency_app_s", "Latency", "s", "Scale", str(output), statistics=cache)
    assert not output.exists()


@pytest.mark.parametrize("column", ["run_id", " run_id "])
def test_csv_plot_keeps_text_run_ids_and_historical_missing_markers(tmp_path, column):
    from acprof.plotting.data import prepare_df
    from acprof.plotting.uncertainty import aggregate_plot_metric

    source = tmp_path / "result_all.csv"
    frame = pd.DataFrame([
        _row(1, 2, repeat_idx=i, warmup=0, status="ok", run_id=run_id)
        for i, run_id in enumerate(("001", "", "N/A"))
    ])
    frame = frame.rename(columns={"run_id": column})
    frame.insert(0, "extension\nlabel", 7)
    source.write_text("\n" + frame.to_csv(index=False))
    prepared = prepare_df(str(source))
    assert prepared["extension\nlabel"].tolist() == [7, 7, 7]
    results = aggregate_plot_metric(prepared, "latency_app_s")
    assert results["n_windows"].tolist() == [3]

    frame.loc[2, column] = "1"
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="inconsistent run_id"):
        aggregate_plot_metric(prepare_df(str(source)), "latency_app_s")


def test_plot_rejects_ambiguous_run_id_columns_before_pandas_renames_labels(tmp_path):
    from acprof.plotting.data import prepare_df

    source = tmp_path / "ambiguous.csv"
    source.write_text("cpu_cores,mem_cap_gb,gpu_mode,input_scale,run_id,run_id\n1,2,off,32,A,B\n")
    with pytest.raises(ValueError, match="run_id"):
        prepare_df(str(source))
