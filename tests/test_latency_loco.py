"""LOCO may skip tail calibration only when none of its predictions use it."""
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.analysis import latency_model as model


def _rows(scales, hardware="gpu"):
    rows = [SimpleNamespace(hardware_model=hardware, cpu_cores=2 ** index, mem_cap_gb=4 + index,
                            input_scale=scale,
                            latency_s=0.02 + scale * 0.001 * (1 + 1 / 2 ** index))
            for index, group in enumerate(scales) for scale in group]
    # Interleave configurations so regrouping cannot silently reorder training.
    return rows[::2] + rows[1::2]


def _validate(rows, hardware, *, force_calibration=False):
    fit = model._fit_log_latency_model
    depth = 0
    calls = []

    def spy(train, hardware_model, *, calibrate_gpu_upper_tail=True):
        nonlocal depth
        if depth == 0:
            calls.append((tuple(map(id, train)), calibrate_gpu_upper_tail))
            if force_calibration:
                calibrate_gpu_upper_tail = True
        depth += 1
        try:
            return fit(train, hardware_model, calibrate_gpu_upper_tail=calibrate_gpu_upper_tail)
        finally:
            depth -= 1

    with patch.object(model, "_fit_log_latency_model", side_effect=spy):
        result = model._resource_configuration_validation(rows, hardware)
    return result, calls


@pytest.mark.parametrize(("scales", "expected"), [
    ([(1, 4, 16)] * 3, [False, False, False]),
    ([(1, 3, 16), (2, 7, 16), (1, 5, 16)], [False, False, False]),
    ([(1, 4, 8), (1, 4, 16), (2, 7, 16)], [False, False, False]),
    ([(1, 4, 32), (1, 4, 16), (2, 7, 16)], [True, False, False]),
    ([(1,), (2,), (4,)], [False, False, True]),
], ids=["shared", "irregular", "equal-training-limit", "extrapolation", "failed-folds"])
@pytest.mark.parametrize("hardware", ["cpu", "gpu"])
def test_loco_skips_only_unused_tail_and_preserves_every_result(scales, expected, hardware):
    rows = _rows(scales, hardware)
    legacy, legacy_calls = _validate(rows, hardware, force_calibration=True)
    actual, calls = _validate(rows, hardware)

    assert [flag for _, flag in calls] == (expected if hardware == "gpu" else [True] * len(scales))
    assert [ids for ids, _ in calls] == [ids for ids, _ in legacy_calls]
    # Includes predictions, per-fold status/failure reasons and every summary metric.
    assert actual == legacy
    assert all(math.isfinite(value) and value > 0 for value in actual[1].values())


def test_formal_fit_and_scale_holdout_still_calibrate_gpu_tail():
    rows = _rows([(1, 4, 16, 64)] * 4)
    with patch.object(model, "_fit_gpu_upper_tail_model", wraps=model._fit_gpu_upper_tail_model) as fit_tail:
        fitted = model._fit_log_latency_model(rows, "gpu")
        assert "gpu_upper_tail" in fitted
        first_calls = fit_tail.call_count
        assert first_calls > 0
        model._input_scale_validation(rows, "gpu")
        assert fit_tail.call_count > first_calls
