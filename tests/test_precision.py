"""Precision criteria preserve observed statistics and do not invent evidence."""
import math

import pytest

from acprof.analysis.precision import assess_mean_precision
from acprof.analysis.uncertainty import summarize_windows


def _rows(values, metric="latency_app_s"):
    return [
        {"cpu_cores": "1", "mem_cap_gb": "4", "gpu_mode": "off", "input_scale": "64",
         "warmup": "0", "repeat_idx": str(index), "status": "ok", metric: value}
        for index, value in enumerate(values)
    ]


@pytest.mark.parametrize(("target", "status"), [(0.25, "met"), (0.24, "not_met")])
def test_asymmetric_interval_uses_half_width_and_inclusive_target(target, status):
    # [8, 13] has width 5 and half-width 2.5; dividing by mean 10 gives 0.25.
    result = assess_mean_precision(10, 8, 13, target=target)
    assert result == {"precision_status": status, "relative_ci_half_width": 0.25,
                      "precision_reason": ""}


def test_precision_is_a_ratio_independent_of_metric_unit():
    result = assess_mean_precision(0.01, 0.008, 0.013, target=0.3)
    assert result["relative_ci_half_width"] == pytest.approx(0.25)
    assert result["precision_status"] == "met"


@pytest.mark.parametrize("target", [0, -0.05, math.inf, -math.inf, math.nan, True])
def test_invalid_targets_are_rejected_even_without_windows(target):
    with pytest.raises(ValueError, match="finite positive ratio"):
        summarize_windows([], ["latency_app_s"], precision_target=target)


@pytest.mark.parametrize(
    ("mean", "low", "high", "reason"),
    [(0, 0, 0, "nonpositive_mean"), (-2, -3, -1, "nonpositive_mean"),
     (None, None, None, "mean_unavailable"), (2, None, None, "interval_unavailable"),
     (2, None, 3, "interval_unavailable"), (2, 3, 1, "invalid_interval"),
     (2, 1, math.inf, "invalid_interval"), (math.nan, 1, 3, "mean_unavailable")],
)
def test_unavailable_mean_or_interval_never_reports_met(mean, low, high, reason):
    result = assess_mean_precision(mean, low, high, target=0.05)
    assert result == {"precision_status": "not_assessable", "relative_ci_half_width": None,
                      "precision_reason": reason}


@pytest.mark.parametrize(
    ("values", "options", "reason"),
    [([3], {}, "insufficient_windows"),
     ([3, 3, 3, "nan"], {}, "missing_windows"),
     (["nan", "nan", "nan"], {}, "insufficient_windows"),
     ([0, 0, 0], {}, "nonpositive_mean"),
     ([3, 3, 3], {"include_intervals": False}, "interval_unavailable"),
     ([3, 3, 3], {"block_size": 2}, "insufficient_windows")],
)
def test_unassessable_windows_keep_original_mean_and_interval(values, options, reason):
    rows = _rows(values)
    baseline = summarize_windows(rows, ["latency_app_s"], resamples=50, **options)
    report = summarize_windows(rows, ["latency_app_s"], resamples=50,
                               precision_target=0.05, **options)
    group = report["groups"][0]
    assert group["precision_status"] == "not_assessable"
    assert group["precision_reason"] == reason
    assert group["relative_ci_half_width"] is None
    for field, value in baseline["groups"][0].items():
        assert group[field] == value


def test_negative_attributed_energy_is_preserved_but_precision_is_not_assessable():
    metric = "container_attributed_energy_eff_j"
    report = summarize_windows(_rows([4, -1, 3], metric), [metric],
                               resamples=50, precision_target=1)
    group = report["groups"][0]
    assert group["mean"] == 2
    assert group["n_windows"] == 3
    assert group["unit"] == "J/request"
    assert group["ci_low"] is not None
    assert group["precision_status"] == "not_assessable"
    assert group["precision_reason"] == "negative_values"
    assert group["relative_ci_half_width"] is None


def test_nonconsecutive_windows_cannot_pass_precision_even_for_individual_bootstrap():
    rows = _rows([3, 3, 3])
    rows[2]["repeat_idx"] = "3"
    group = summarize_windows(rows, ["latency_app_s"], resamples=50,
                              precision_target=0.05)["groups"][0]
    assert (group["ci_low"], group["ci_high"]) == (3, 3)
    assert group["reason"] == ""
    assert group["precision_reason"] == "nonconsecutive_windows"
    assert group["precision_status"] == "not_assessable"


@pytest.mark.parametrize("status", ["warn", "error"])
def test_excluded_formal_windows_prevent_successful_subset_from_passing(status):
    rows = _rows([3, 3, 3, 100])
    rows[3]["status"] = status
    group = summarize_windows(rows, ["latency_app_s"], resamples=50,
                              precision_target=0.05)["groups"][0]
    assert group["mean"] == 3
    assert group["missing_windows"] == 0
    assert group["precision_excluded_windows"] == 1
    assert group["precision_status"] == "not_assessable"
    assert group["precision_reason"] == "excluded_windows"


def test_warmup_is_not_a_missing_precision_window_and_groups_remain_separate():
    rows = _rows([3, 3, 3, 100])
    rows[-1].update(warmup="1", status="error")
    rows.extend({**row, "cpu_cores": "2"} for row in _rows([1, 4, 7]))
    report = summarize_windows(rows, ["latency_app_s"], resamples=100,
                               precision_target=0.05)
    first, second = report["groups"]
    assert first["precision_excluded_windows"] == 0
    assert first["relative_ci_half_width"] == 0
    assert first["precision_status"] == "met"
    assert second["precision_status"] == "not_met"
    assert report["precision"]["target_relative_half_width"] == 0.05
    assert report["precision"]["unit"] == "ratio"
    assert report["precision"]["scope"] == "within_run_observed_windows"


def test_one_bootstrap_resample_cannot_certify_precision():
    rows = _rows([1, 100, 10000])
    baseline = summarize_windows(rows, ["latency_app_s"], resamples=1)
    group = summarize_windows(rows, ["latency_app_s"], resamples=1,
                              precision_target=0.000001)["groups"][0]
    assert group["mean"] == 3367
    assert group["ci_low"] == group["ci_high"]
    assert group["ci_low"] is not None
    assert group["precision_status"] == "not_assessable"
    assert group["precision_reason"] == "insufficient_resamples"
    assert group["relative_ci_half_width"] is None
    assert "precision" not in baseline
    for field, value in baseline["groups"][0].items():
        assert group[field] == value


def test_constant_windows_still_require_multiple_bootstrap_resamples():
    one = summarize_windows(_rows([3, 3, 3]), ["latency_app_s"], resamples=1,
                            precision_target=0.05)["groups"][0]
    two = summarize_windows(_rows([3, 3, 3]), ["latency_app_s"], resamples=2,
                            precision_target=0.05)["groups"][0]
    assert one["precision_reason"] == "insufficient_resamples"
    assert one["precision_status"] == "not_assessable"
    assert two["relative_ci_half_width"] == 0
    assert two["precision_status"] == "met"


def test_nonconstant_windows_with_degenerate_block_interval_do_not_certify_precision():
    # Each circular length-two block sums to 3, so every six-value resample
    # has mean 1.5 even though the observed windows alternate between 1 and 2.
    group = summarize_windows(_rows([1, 2, 1, 2, 1, 2]), ["latency_app_s"],
                              block_size=2, resamples=50, precision_target=0.05)["groups"][0]
    assert group["n_windows"] == 6
    assert group["mean"] == 1.5
    assert (group["ci_low"], group["ci_high"]) == (1.5, 1.5)
    assert group["reason"] == ""
    assert group["precision_status"] == "not_assessable"
    assert group["precision_reason"] == "degenerate_interval"
    assert group["relative_ci_half_width"] is None


def test_one_observed_block_is_not_enough_for_precision():
    group = summarize_windows(_rows([1, 100, 10000]), ["latency_app_s"],
                              block_size=3, resamples=50, precision_target=0.05)["groups"][0]
    assert group["mean"] == 3367
    assert group["n_windows"] == 3
    assert group["ci_low"] is None
    assert group["ci_high"] is None
    assert group["precision_status"] == "not_assessable"
    assert group["precision_reason"] == "insufficient_windows"


def test_default_report_retains_its_historical_shape():
    report = summarize_windows(_rows([3, 3, 3]), ["latency_app_s"], resamples=50)
    assert "precision" not in report
    assert not any(field.startswith("precision") or field == "relative_ci_half_width"
                   for group in report["groups"] for field in group)
    assert report == summarize_windows(_rows([3, 3, 3]), ["latency_app_s"],
                                       resamples=50, precision_target=None)
