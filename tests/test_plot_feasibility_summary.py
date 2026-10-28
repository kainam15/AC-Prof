"""Feasibility classification preserves repeated outcomes and normalization."""

import numpy as np
import pandas as pd
import pytest

from acprof.plotting.diagnostics import summarize_resource_feasibility

DIMENSIONS = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale"]


@pytest.mark.parametrize(("outcomes", "expected"), [
    ([(" OK ", " \t")], "ok"),
    ([("ok", ""), (" WaRN\n", None)], "warn"),
    ([("ok", ""), ("error", "")], "mixed"),
    ([("ok", "timeout")], "mixed"),
    ([("warn", "")], "warn"),
    ([("warn", "timeout")], "timeout"),
    ([(None, pd.NA), (np.nan, None)], "unknown"),
    ([("unexpected status", "")], "unknown"),
    ([("error", "")], "error"),
    ([("", "other failure")], "error"),
    ([("", 0)], "error"),
    ([("error", "TIMEOUT")], "timeout"),
    ([("error", "Runtime OOM")], "runtime_oom"),
    ([("error", "startup"), ("error", "out of memory")], "runtime_oom"),
    ([("error", "container_start_failed"), ("error", "oom")], "runtime_oom"),
    ([("error", "out of"), ("error", "memory")], "error"),
    ([("error", "memory"), ("error", "out of")], "error"),
    ([("error", "out"), ("error", "of"), ("error", "memory")], "error"),
    ([("error", "timeout"), ("error", "oom")], "runtime_oom"),
    ([("error", "not_measured_after_timeout"), ("error", "")], "skipped"),
    ([("error", "not_measured_after_startup_oom_pruning"), ("error", "")], "pruned_startup_oom"),
    ([("error", "not_measured_after_timeout not_measured_after_startup_oom_pruning")], "skipped"),
    ([("error", "not_measured_after_timeout"), ("error", "other failure")], "error"),
    ([("error", "not_measured_after_startup_oom_pruning"), ("error", "runtime failure")], "error"),
])
def test_feasibility_summary_classifies_repeated_outcomes(outcomes, expected):
    frame = pd.DataFrame([(1, 4, "off", 2, status, error) for status, error in outcomes],
                         columns=[*DIMENSIONS, "status", "error"])
    before = frame.copy(deep=True)

    summary = summarize_resource_feasibility(frame)

    assert summary["state"].tolist() == [expected]
    assert summary["gpu_on"].tolist() == [False]
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(("columns", "expected"), [
    (["status", "error"], "mixed"), (["status"], "ok"), (["error"], "timeout"), ([], "unknown"),
])
def test_feasibility_summary_handles_missing_outcome_columns(columns, expected):
    frame = pd.DataFrame([(1, 4, "on", 2, "ok", "timeout")], columns=[*DIMENSIONS, "status", "error"])
    frame = frame[[*DIMENSIONS, *columns]]

    summary = summarize_resource_feasibility(frame)

    assert summary["state"].tolist() == [expected]


@pytest.mark.parametrize("dtype", ["string", "category", "Int64"])
def test_empty_feasibility_summary_does_not_normalize_unused_values(dtype):
    frame = pd.DataFrame(columns=DIMENSIONS)
    frame["status"] = pd.Series([], dtype=dtype)
    frame["error"] = pd.Series([], dtype=dtype)

    summary = summarize_resource_feasibility(frame)

    assert summary.empty
    assert summary.columns.tolist() == [*DIMENSIONS, "state"]


@pytest.mark.parametrize("group_count", [1, 16, 128])
def test_feasibility_text_preparation_is_bounded_by_columns(group_count, monkeypatch):
    frame = pd.DataFrame([(1, 4, "off", scale, " OK ", "") for scale in range(1, group_count + 1)],
                         columns=[*DIMENSIONS, "status", "error"])
    casts = []
    astype = pd.Series.astype

    def count_text_casts(series, dtype, *args, **kwargs):
        if series.name in {"status", "error"} and dtype is str:
            casts.append(series.name)
        return astype(series, dtype, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "astype", count_text_casts)

    summary = summarize_resource_feasibility(frame)

    assert summary["state"].tolist() == ["ok"] * group_count
    assert len(casts) <= 2


def test_feasibility_summary_preserves_group_order_with_duplicate_row_indexes():
    scale = np.nextafter(1., 2.)
    frame = pd.DataFrame([
        (2.5, .75, "on", 1., "error", "timeout"),
        (.5, .25, "off", scale, "ok", ""),
        (.5, .25, "off", 1., "warn", ""),
        (2.5, .75, "on", 1., "ok", ""),
        (.5, .25, "off", np.nan, "error", "out of memory"),
    ], columns=[*DIMENSIONS, "status", "error"], index=[7, 7, 2, 7, 2])
    frame = pd.concat([frame, pd.DataFrame({f"metric_{i}": i for i in range(200)}, index=frame.index)], axis=1)
    before = frame.copy(deep=True)

    summary = summarize_resource_feasibility(frame)

    expected = pd.DataFrame([
        (.5, .25, "off", 1., "warn", False),
        (.5, .25, "off", scale, "ok", False),
        (.5, .25, "off", np.nan, "runtime_oom", False),
        (2.5, .75, "on", 1., "mixed", True),
    ], columns=[*DIMENSIONS, "state", "gpu_on"])
    pd.testing.assert_frame_equal(summary, expected)
    pd.testing.assert_frame_equal(frame, before)
