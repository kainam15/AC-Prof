"""Resource feasibility plots preserve empty and sparse matrix semantics."""

import numpy as np
import pandas as pd
import pytest

from acprof.plotting import diagnostics
from acprof.plotting.config import FEASIBILITY_STATE_INDEX

DIMENSIONS = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale"]


@pytest.fixture
def figures(monkeypatch):
    captured = []
    close = diagnostics.plt.close

    def record(figure=None):
        if hasattr(figure, "axes"):
            captured.append(figure)
        close(figure)

    monkeypatch.setattr(diagnostics.plt, "close", record)
    return captured


@pytest.mark.parametrize("outcome_columns", [["status"], ["error"], ["status", "error"]])
def test_empty_feasibility_summary_keeps_its_columns(outcome_columns):
    frame = pd.DataFrame(columns=[*DIMENSIONS, *outcome_columns])

    summary = diagnostics.summarize_resource_feasibility(frame)

    assert summary.empty
    assert set(DIMENSIONS + ["state"]).issubset(summary.columns)


@pytest.mark.parametrize("outcome_columns", [["status"], ["error"], ["status", "error"]])
def test_empty_feasibility_plot_skips_without_creating_a_figure(outcome_columns, monkeypatch, capsys):
    frame = pd.DataFrame(columns=[*DIMENSIONS, *outcome_columns])
    before = frame.copy(deep=True)
    monkeypatch.setattr(diagnostics.plt, "subplots", lambda *args, **kwargs: pytest.fail("empty figure"))

    assert not diagnostics.plot_resource_feasibility_heatmap(frame, xlabel="scale", out_png=None)

    assert "No resource feasibility data" in capsys.readouterr().out
    pd.testing.assert_frame_equal(frame, before)


def test_sparse_feasibility_panels_preserve_failure_states_and_axis_order(figures):
    frame = pd.DataFrame([
        (.5, .25, "off", 1, "ok", ""),
        (.5, .25, "off", 2, "warn", ""),
        (.5, .75, "off", 1, "ok", ""),
        (.5, .75, "off", 1, "error", "runtime failure"),
        (.5, .75, "off", 2, "error", "startup out of memory"),
        (.5, .25, "on", 1, "error", "runtime OOM"),
        (.5, .75, "on", 4, "error", "timeout"),
        (2.5, .25, "off", 2, "error", "not_measured_after_timeout"),
        (2.5, .75, "off", 4, "error", "not_measured_after_startup_oom_pruning"),
        (2.5, .25, "on", 1, "error", "runtime failure"),
        (2.5, .75, "on", 2, "", ""),
    ], columns=[*DIMENSIONS, "status", "error"]).sample(frac=1, random_state=37)
    before = frame.copy(deep=True)

    assert diagnostics.plot_resource_feasibility_heatmap(frame, xlabel="scale", out_png=None)

    expected = [
        [["ok", "warn", "unknown"], ["mixed", "startup_oom", "unknown"]],
        [["runtime_oom", "unknown", "unknown"], ["unknown", "unknown", "timeout"]],
        [["unknown", "skipped", "unknown"], ["unknown", "unknown", "pruned_startup_oom"]],
        [["error", "unknown", "unknown"], ["unknown", "unknown", "unknown"]],
    ]
    axes = figures[-1].axes
    assert [axis.get_title() for axis in axes] == [
        "CPU-only, 0.5 CPU cores", "GPU, 0.5 CPU cores",
        "CPU-only, 2.5 CPU cores", "GPU, 2.5 CPU cores",
    ]
    for axis, states in zip(axes, expected, strict=True):
        assert axis.images[0].get_array().tolist() == [
            [FEASIBILITY_STATE_INDEX[state] for state in row] for row in states
        ]
        assert [label.get_text() for label in axis.get_xticklabels()] == ["1", "2", "4"]
        assert [label.get_text() for label in axis.get_yticklabels()] == ["0.25", "0.75"]
        assert len(axis.texts) == 6
    assert len(figures[-1].legends[0].get_texts()) == 10
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(("memory_count", "scale_count"), [(4, 8), (8, 16)])
def test_feasibility_coordinate_comparison_work_is_linear(memory_count, scale_count, monkeypatch, figures):
    frame = pd.DataFrame([
        (1, memory, "off", scale, "ok", "")
        for memory in range(1, memory_count + 1)
        for scale in range(1, scale_count + 1)
    ], columns=[*DIMENSIONS, "status", "error"])
    compared_values = 0
    series_eq = pd.Series.__eq__

    def count_coordinates(series, other):
        nonlocal compared_values
        if series.name in {"mem_cap_gb", "input_scale"}:
            compared_values += len(series)
        return series_eq(series, other)

    monkeypatch.setattr(pd.Series, "__eq__", count_coordinates)

    assert diagnostics.plot_resource_feasibility_heatmap(frame, xlabel="scale", out_png=None)

    assert compared_values <= 8 * len(frame)
    np.testing.assert_array_equal(
        figures[-1].axes[0].images[0].get_array(),
        np.full((memory_count, scale_count), FEASIBILITY_STATE_INDEX["ok"]),
    )


@pytest.mark.parametrize("scale", [np.int64(2**53 + 1), np.uint64(2**64 - 1)])
def test_feasibility_integer_scale_uses_plotted_float_coordinate(scale, figures):
    frame = pd.DataFrame([(1, 4, "off", scale, "ok")], columns=[*DIMENSIONS, "status"])

    assert diagnostics.plot_resource_feasibility_heatmap(frame, xlabel="scale", out_png=None)

    axis = figures[-1].axes[0]
    assert axis.images[0].get_array().tolist() == [[FEASIBILITY_STATE_INDEX["ok"]]]
    assert float(axis.get_xticklabels()[0].get_text()) == float(scale)


@pytest.mark.parametrize("column", ["cpu_cores", "mem_cap_gb", "input_scale"])
def test_feasibility_nullable_coordinates_leave_unmeasured_cells_unknown(column, figures):
    frame = pd.DataFrame([(1, 4, "off", 1, "ok"), (1, 8, "off", 2, "error")],
                         columns=[*DIMENSIONS, "status"])
    frame[column] = frame[column].astype("Float64")
    frame.loc[1, column] = pd.NA
    before = frame.copy(deep=True)

    assert diagnostics.plot_resource_feasibility_heatmap(frame, xlabel="scale", out_png=None)

    matrix = figures[-1].axes[0].images[0].get_array()
    assert matrix[0, 0] == FEASIBILITY_STATE_INDEX["ok"]
    assert np.count_nonzero(matrix == FEASIBILITY_STATE_INDEX["ok"]) == 1
    assert np.count_nonzero(matrix == FEASIBILITY_STATE_INDEX["unknown"]) == matrix.size - 1
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("column", ["cpu_cores", "mem_cap_gb", "input_scale"])
def test_feasibility_plot_skips_missing_dimensions(column, monkeypatch, capsys):
    frame = pd.DataFrame([(1, 4, "off", 2, "ok")], columns=[*DIMENSIONS, "status"])
    frame[column] = np.nan
    monkeypatch.setattr(diagnostics.plt, "subplots", lambda *args, **kwargs: pytest.fail("empty figure"))

    assert not diagnostics.plot_resource_feasibility_heatmap(frame, xlabel="scale", out_png=None)

    assert "Incomplete resource feasibility dimensions" in capsys.readouterr().out


@pytest.mark.parametrize("mode", [None, np.nan, "unknown"])
def test_feasibility_plot_rejects_unknown_gpu_modes(mode, monkeypatch):
    frame = pd.DataFrame([(1, 4, mode, 2, "ok")], columns=[*DIMENSIONS, "status"])
    before = frame.copy(deep=True)
    monkeypatch.setattr(diagnostics.plt, "subplots", lambda *args, **kwargs: pytest.fail("invalid figure"))

    with pytest.raises(ValueError, match="unsupported gpu_mode"):
        diagnostics.plot_resource_feasibility_heatmap(frame, xlabel="scale", out_png=None)

    pd.testing.assert_frame_equal(frame, before)
