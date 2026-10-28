"""Resource feasibility plots preserve empty and sparse matrix semantics."""

import pandas as pd
import pytest

from acprof.plotting import diagnostics

DIMENSIONS = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale"]


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
