"""Tail plots preserve paired window evidence and visible missing-data gaps."""

import numpy as np
import pandas as pd
import pytest

from acprof.plotting import tail_latency

FAMILIES = ("latency", "latency_app")


def _row(*, scale=1, memory=4, **metrics):
    return {"cpu_cores": 1, "mem_cap_gb": memory, "gpu_mode": "off", "input_scale": scale, **metrics}


@pytest.fixture
def figures(monkeypatch):
    captured = []
    close = tail_latency.plt.close

    def record(figure=None):
        if hasattr(figure, "axes"):
            captured.append(figure)
        close(figure)

    monkeypatch.setattr(tail_latency.plt, "close", record)
    return captured


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(("p50", "p95"), [
    (np.nan, np.nan), (np.nan, 3.), (1., np.nan), (np.inf, np.inf),
    (1., np.inf), (0., 3.), (-1., 3.), (3., 1.),
])
def test_representative_memory_requires_a_valid_window_pair(family, p50, p95):
    frame = pd.DataFrame([
        _row(**{f"{family}_p50_s": 1., f"{family}_p90_s": 2., f"{family}_p95_s": 3.}),
        _row(memory=8, **{f"{family}_p50_s": p50, f"{family}_p90_s": 2., f"{family}_p95_s": p95}),
    ])
    before = frame.copy(deep=True)

    result = tail_latency._aggregate_tail_latency(frame)

    assert result["mem_cap_gb"].tolist() == [4]
    assert result[f"{family}_p50_s"].tolist() == [1.]
    assert result[f"{family}_p95_s"].tolist() == [3.]
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("family", FAMILIES)
def test_complementary_missing_repeats_do_not_create_a_percentile_pair(family):
    frame = pd.DataFrame([
        _row(**{f"{family}_p50_s": 1., f"{family}_p95_s": np.nan}),
        _row(**{f"{family}_p50_s": np.nan, f"{family}_p95_s": 3.}),
    ])

    assert tail_latency._aggregate_tail_latency(frame).empty


@pytest.mark.parametrize("family", FAMILIES)
def test_incomplete_repeats_do_not_bias_valid_window_medians(family):
    frame = pd.DataFrame([
        _row(**{f"{family}_p50_s": 1., f"{family}_p90_s": 2., f"{family}_p95_s": 3.}),
        _row(**{f"{family}_p50_s": 100., f"{family}_p90_s": 200., f"{family}_p95_s": np.nan}),
        _row(**{f"{family}_p50_s": np.nan, f"{family}_p90_s": 600., f"{family}_p95_s": 900.}),
    ], index=[7, 7, 2])
    for ordered in (frame, frame.iloc[::-1]):
        before = ordered.copy(deep=True)
        result = tail_latency._aggregate_tail_latency(ordered)
        assert result[[f"{family}_p{p}_s" for p in (50, 90, 95)]].values.tolist() == [[1., 2., 3.]]
        pd.testing.assert_frame_equal(ordered, before)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("p90", [np.nan, np.inf, -np.inf, 0., 4.])
def test_invalid_optional_p90_does_not_remove_the_valid_pair(family, p90):
    frame = pd.DataFrame([_row(**{
        f"{family}_p50_s": 1., f"{family}_p90_s": p90, f"{family}_p95_s": 3.,
    })])

    result = tail_latency._aggregate_tail_latency(frame)

    assert result[[f"{family}_p50_s", f"{family}_p95_s"]].values.tolist() == [[1., 3.]]
    assert pd.isna(result.iloc[0][f"{family}_p90_s"])


def test_sparse_p90_median_cannot_extend_outside_the_aggregated_band():
    frame = pd.DataFrame([
        _row(latency_p50_s=1., latency_p90_s=np.nan, latency_p95_s=3.),
        _row(latency_p50_s=1., latency_p90_s=np.nan, latency_p95_s=3.),
        _row(latency_p50_s=100., latency_p90_s=150., latency_p95_s=200.),
    ])

    result = tail_latency._aggregate_tail_latency(frame)

    assert result[["latency_p50_s", "latency_p95_s"]].values.tolist() == [[1., 3.]]
    assert pd.isna(result.iloc[0]["latency_p90_s"])


def test_families_keep_their_own_window_pairs_with_nullable_values():
    frame = pd.DataFrame([
        _row(latency_p50_s=1., latency_p95_s=3., latency_app_p50_s=100., latency_app_p95_s=pd.NA),
        _row(latency_p50_s=pd.NA, latency_p95_s=900., latency_app_p50_s=2., latency_app_p95_s=4.),
    ], index=[5, 5])
    columns = [f"{family}_p{percentile}_s" for family in FAMILIES for percentile in (50, 95)]
    frame[columns] = frame[columns].astype("Float64")
    before = frame.copy(deep=True)

    result = tail_latency._aggregate_tail_latency(frame)

    assert result[columns].values.tolist() == [[1., 3., 2., 4.]]
    pd.testing.assert_frame_equal(frame, before)


def test_one_representative_configuration_is_shared_by_the_two_families():
    frame = pd.DataFrame([
        _row(latency_p50_s=1., latency_p95_s=3.),
        _row(memory=8, latency_app_p50_s=2., latency_app_p95_s=4.),
    ])

    result = tail_latency._aggregate_tail_latency(frame)

    assert result["mem_cap_gb"].tolist() == [8]
    assert result[["latency_p50_s", "latency_p95_s"]].isna().all().all()
    assert result[["latency_app_p50_s", "latency_app_p95_s"]].values.tolist() == [[2., 4.]]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("missing_columns", [False, True])
def test_empty_family_panels_say_no_data(family, missing_columns, figures):
    other = next(value for value in FAMILIES if value != family)
    metrics = {f"{family}_p50_s": 1., f"{family}_p95_s": 3.}
    if not missing_columns:
        metrics.update({f"{other}_p{p}_s": np.nan for p in (50, 90, 95)})
    frame = pd.DataFrame([_row(**metrics)])

    assert tail_latency.plot_tail_latency_overview(frame, xlabel="scale", out_png=None)

    column = FAMILIES.index(other)
    for axis in (figures[-1].axes[column], figures[-1].axes[column + 2]):
        assert [text.get_text() for text in axis.texts] == ["No data"]
        assert not axis.lines
    assert "P90" not in [text.get_text() for text in figures[-1].legends[0].texts]


@pytest.mark.parametrize("family", FAMILIES)
def test_missing_scale_breaks_percentile_lines_bands_and_ratios(family, figures):
    frame = pd.DataFrame([
        _row(scale=scale, **{f"{family}_p{p}_s": np.nan if scale == 3 else p / 25. for p in (50, 90, 95)})
        for scale in range(1, 6)
    ])

    assert tail_latency.plot_tail_latency_overview(frame, xlabel="scale", out_png=None)

    column = FAMILIES.index(family)
    percentile_axis, ratio_axis = figures[-1].axes[column], figures[-1].axes[column + 2]
    for line in [*percentile_axis.lines, *ratio_axis.lines]:
        np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3, 4, 5])
        assert np.isnan(line.get_ydata()[2])
    regions = percentile_axis.collections[0].get_paths()
    assert [(path.vertices[:, 0].min(), path.vertices[:, 0].max()) for path in regions] == [(1, 2), (4, 5)]


def test_missing_p90_keeps_a_gap_without_breaking_other_percentiles(figures):
    frame = pd.DataFrame([
        _row(scale=scale, latency_p50_s=1., latency_p90_s=np.nan if scale == 3 else 2., latency_p95_s=3.)
        for scale in range(1, 6)
    ])

    assert tail_latency.plot_tail_latency_overview(frame, xlabel="scale", out_png=None)

    p50, p90, p95 = figures[-1].axes[0].lines
    assert np.isfinite(p50.get_ydata()).all()
    assert np.isfinite(p95.get_ydata()).all()
    np.testing.assert_array_equal(p90.get_xdata(), [1, 2, 3, 4, 5])
    assert np.isnan(p90.get_ydata()[2])


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(("valid", "isolated"), [
    ([True], [0]), ([True, False, True], [0, 2]), ([True, True, False, True], [3]),
])
def test_isolated_percentile_points_remain_visible(family, valid, isolated, figures):
    frame = pd.DataFrame([
        _row(scale=scale, **{f"{family}_p{p}_s": p / 25. if present else np.nan for p in (50, 90, 95)})
        for scale, present in enumerate(valid, 1)
    ])

    assert tail_latency.plot_tail_latency_overview(frame, xlabel="scale", out_png=None)

    for line in figures[-1].axes[FAMILIES.index(family)].lines:
        assert line.get_marker() == "o"
        assert line.get_markevery() == isolated


def test_sparse_p90_point_is_visible_without_marking_continuous_bands(figures):
    frame = pd.DataFrame([
        _row(scale=scale, latency_p50_s=1., latency_p90_s=2. if scale == 3 else np.nan, latency_p95_s=3.)
        for scale in range(1, 6)
    ])

    assert tail_latency.plot_tail_latency_overview(frame, xlabel="scale", out_png=None)

    p50, p90, p95 = figures[-1].axes[0].lines
    assert p90.get_marker() == "o"
    assert p90.get_markevery() == [2]
    assert p50.get_marker() == p95.get_marker() == "None"


@pytest.mark.parametrize("metrics", [
    {}, {"latency_p90_s": 2.}, {"latency_p50_s": 1.}, {"latency_p95_s": 3.},
    {"latency_p50_s": np.nan, "latency_p95_s": np.nan},
])
def test_unplottable_tail_data_skips_before_allocating_a_figure(metrics, monkeypatch):
    frame = pd.DataFrame([_row(**metrics)])
    monkeypatch.setattr(tail_latency.plt, "subplots", lambda *a, **kw: pytest.fail("unplottable figure"))

    assert not tail_latency.plot_tail_latency_overview(frame, xlabel="scale", out_png=None)
