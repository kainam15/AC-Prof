"""Startup statistics count startup identities, not repeated workload rows."""

import numpy as np
import pandas as pd
import pytest

from acprof.plotting import diagnostics
from acprof.plotting.data import aggregate_cold_start

START_A = "2026-10-29T10:00:00.123+08:00"
START_B = "2026-10-29T10:01:00.123+08:00"
START_C = "2026-10-29T10:02:00.123+08:00"


def _row(start, duration, **extra):
    return {"cpu_cores": 1, "mem_cap_gb": 4, "gpu_mode": "off", "input_scale": 1,
            "cold_start_started_at": start, "cold_start_s": duration, **extra}


@pytest.mark.parametrize("agg_func", ["mean", "median"])
@pytest.mark.parametrize("copies", [1, 2, 7])
def test_each_startup_has_one_vote_independent_of_window_count(agg_func, copies):
    frame = pd.DataFrame([_row(START_A, 2., input_scale=scale) for scale in range(copies)] + [_row(START_B, 8.)])
    frame.index = [7] * len(frame)
    before = frame.copy(deep=True)

    result = aggregate_cold_start(frame, agg_func=agg_func)

    assert result["cold_start_s"].tolist() == [5.]
    pd.testing.assert_frame_equal(frame, before)


def test_distinct_startups_with_identical_durations_keep_separate_votes():
    frame = pd.DataFrame([_row(START_A, 2.)] * 7 + [_row(START_B, 2.), _row(START_C, 8.)])

    assert aggregate_cold_start(frame)["cold_start_s"].tolist() == [4.]


@pytest.mark.parametrize("identity", [START_A, None, "", " NaN ", "unknown", "n/a", pd.NA])
@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_values_for_one_or_unidentified_startup_are_rejected(identity, reverse):
    frame = pd.DataFrame([_row(identity, 2.), _row(identity, 8.)])
    if reverse:
        frame = frame.iloc[::-1]
    before = frame.copy(deep=True)

    with pytest.raises(ValueError, match="cold_start_s"):
        aggregate_cold_start(frame)

    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("identity", [None, "", "nan", "UNKNOWN", "n/a", pd.NA])
def test_missing_and_known_startup_identities_do_not_invent_an_extra_start(identity):
    frame = pd.DataFrame([_row(identity, 2.), _row(START_A, 2.)])

    with pytest.raises(ValueError, match="identity"):
        aggregate_cold_start(frame)


@pytest.mark.parametrize("with_column", [False, True])
def test_legacy_repeated_consistent_value_remains_usable(with_column):
    frame = pd.DataFrame([_row(value, 2.) for value in [None, "", "nan", "UNKNOWN", "n/a", pd.NA]])
    if not with_column:
        frame = frame.drop(columns="cold_start_started_at")

    assert aggregate_cold_start(frame)["cold_start_s"].tolist() == [2.]


def test_startup_identity_is_scoped_to_resource_and_environment_groups():
    frame = pd.DataFrame([
        _row(START_A, 2., environment_class="native_linux"),
        _row(START_A, 2., environment_class="native_linux"),
        _row(START_A, 8., environment_class="wsl2"),
        _row(START_A, 3., cpu_cores=2, environment_class="native_linux"),
    ])

    result = aggregate_cold_start(frame)

    assert result["cold_start_s"].tolist() == [2., 3., 8.]
    assert result["config"].tolist() == [
        "native_linux / CPU+CPU1+Mem4", "native_linux / CPU+CPU2+Mem4", "wsl2 / CPU+CPU1+Mem4",
    ]


@pytest.mark.parametrize("invalid", [np.inf, -np.inf, -1.])
def test_invalid_cold_start_duration_does_not_enter_a_bar(invalid):
    frame = pd.DataFrame([_row(START_A, invalid)])

    assert aggregate_cold_start(frame).empty


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


def _phases(start, factor=1.):
    return _row(start, 8. * factor,
                cold_start_container_launch_s=1. * factor, cold_start_server_setup_s=2. * factor,
                cold_start_cuda_init_s=0., cold_start_model_load_s=4. * factor,
                cold_start_ready_wait_s=1. * factor, cold_start_first_predict_app_s=1. * factor)


@pytest.mark.parametrize("copies", [1, 2, 7])
@pytest.mark.parametrize("with_environment", [False, True])
def test_breakdown_and_first_prediction_use_unique_startups(copies, with_environment, figures):
    frame = pd.DataFrame([_phases(START_A)] * copies + [_phases(START_B, 3.)])
    if with_environment:
        frame["environment_class"] = "native_linux"
    before = frame.copy(deep=True)

    assert diagnostics.plot_cold_start_breakdown(frame, out_png=None)

    phases, first = figures[-1].axes
    assert [bar.get_height() for bar in phases.patches] == [2., 4., 0., 8., 2.]
    np.testing.assert_array_equal(phases.collections[0].get_offsets(), [[0., 16.]])
    np.testing.assert_array_equal(first.collections[0].get_offsets(), [[0., 2.]])
    assert [label.get_text() for label in first.get_xticklabels()] == ["CPU+CPU1+Mem4"]
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("metric", [
    "cold_start_model_load_s", "cold_start_s", "cold_start_first_predict_app_s",
])
def test_breakdown_rejects_contradictory_startup_evidence(metric):
    one, two = _phases(START_A), _phases(START_A)
    two[metric] += 1.
    frame = pd.DataFrame([one, two])

    with pytest.raises(ValueError, match=metric):
        diagnostics.plot_cold_start_breakdown(frame, out_png=None)


def test_breakdown_does_not_combine_complementary_incomplete_phase_rows():
    one, two = _phases(START_A), _phases(START_A)
    one["cold_start_model_load_s"] = np.nan
    two["cold_start_ready_wait_s"] = np.nan

    assert not diagnostics.plot_cold_start_breakdown(pd.DataFrame([one, two]), out_png=None)
