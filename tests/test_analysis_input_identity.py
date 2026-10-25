import csv
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from acprof.analysis.latency_report import write_latency_model_report
from acprof.plotting.data import build_plot_groups, prepare_df
from acprof.plotting.latency import plot_latency_model_fit_curves
from acprof.plotting.metrics import plot_metric_overview


def fractional_rows():
    return [
        dict(cpu_cores=cpu, mem_cap_gb=mem, gpu_mode="off", input_scale=scale,
             latency_s=(scale / 1e-8) / cpu, status="ok", warmup=0, repeat_idx=0)
        for cpu in (1.5, 2.5, 4.5, 8.5)
        for mem in (2.25, 4.25, 8.25)
        for scale in (1e-8, 2e-8, 4e-8, 8e-8)
    ]


def test_residual_identity_roundtrips_fractional_resources_and_tiny_scales(tmp_path):
    rows = fractional_rows()
    write_latency_model_report(pd.DataFrame(rows), {}, str(tmp_path))
    with (tmp_path / "latency_model/latency_model_residuals.csv").open() as stream:
        actual = list(csv.DictReader(stream))
    fields = ("cpu_cores", "mem_cap_gb", "input_scale")
    assert {tuple(float(row[k]) for k in fields) for row in actual} == {
        tuple(row[k] for k in fields) for row in rows
    }
    try:
        with patch.object(plt, "close"):
            assert plot_latency_model_fit_curves(
                str(tmp_path / "latency_model/latency_model_residuals.csv"),
                str(tmp_path / "latency_model/latency_model_report.json"), None,
            )
        assert len(plt.gcf().axes[0].lines) == 12
        assert all(min(line.get_xdata()) == 1e-8 for line in plt.gcf().axes[0].lines)
    finally:
        plt.close("all")


@pytest.mark.parametrize("invalid", ["mystery", "", None, float("nan")])
def test_plot_groups_reject_unknown_gpu_before_filtering(invalid):
    with pytest.raises(ValueError, match="gpu_mode"):
        build_plot_groups(pd.DataFrame({"gpu_mode": ["off", "on", invalid]}))


def test_csv_rejects_unknown_gpu_even_on_filtered_error_row(tmp_path):
    path = tmp_path / "external.csv"
    pd.DataFrame([*fractional_rows(), dict(cpu_cores=2, mem_cap_gb=4, input_scale=1,
                                         gpu_mode="mystery", status="error")]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="gpu_mode"):
        prepare_df(str(path))


def test_fractional_configurations_keep_distinct_curves_and_labels():
    frame = pd.DataFrame(fractional_rows())
    try:
        with patch.object(plt, "close"):
            plot_metric_overview(frame, panels=(("latency_s", "Latency", "s"),),
                                 rows=1, columns=1, shared_y_groups=(), title="Identity",
                                 xlabel="scale", out_png=None)
        figure = plt.gcf()
        assert len(figure.axes[0].lines) == 12
        assert all(len(line.get_xdata()) == 4 for line in figure.axes[0].lines)
        labels = [label.get_text() for label in figure.legends[0].get_texts()]
        assert "CPU1.5" in labels
        assert "Mem2.25" in labels
    finally:
        plt.close("all")
