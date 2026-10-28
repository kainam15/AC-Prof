"""Exact two-objective dominance without quadratic observation comparisons."""
import math

import numpy as np
import pytest

from acprof.plotting.diagnostics import _pareto_frontier_mask


def oracle(latencies, energies):
    points = list(zip(latencies, energies, strict=True))
    return [not any(a <= x and b <= y and (a < x or b < y) for a, b in points) for x, y in points]


@pytest.mark.parametrize("points", [
    [], [(1., 2.)], [(1., 2.)] * 8,
    [(2., 2.), (1., 3.), (2., 2.), (1., 2.), (3., 1.), (3., 1.)],
    [(0., -0.), (-0., 0.), (1., 0.), (0., 1.)],
    [(1., 2.), (np.nextafter(1., 2.), 2.), (1., np.nextafter(2., 3.))],
    [(math.nan, 1.), (1., math.nan), (math.nan, math.nan), (2., 2.), (1., 1.)],
    [(math.inf, math.inf), (1., math.inf), (math.inf, 1.), (-math.inf, 2.), (2., -math.inf)],
    [(math.nan, math.inf)] * 4,
])
def test_frontier_preserves_strict_dominance_duplicates_and_nonfinite_comparisons(points):
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])
    actual = _pareto_frontier_mask(x, y)
    assert actual.dtype == np.bool_
    assert actual.tolist() == oracle(x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
@pytest.mark.parametrize("seed", [7, 19, 29])
def test_frontier_matches_oracle_for_unordered_readonly_strided_arrays(dtype, seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 15, 161).astype(dtype)[::-2]
    y = rng.integers(0, 15, 161).astype(dtype)[::2]
    x.flags.writeable = False
    y.flags.writeable = False
    before_x, before_y = x.copy(), y.copy()
    assert _pareto_frontier_mask(x, y).tolist() == oracle(x, y)
    np.testing.assert_array_equal(x, before_x)
    np.testing.assert_array_equal(y, before_y)


def test_integer_coordinates_are_not_rounded_through_float():
    maximum = np.iinfo(np.uint64).max
    x = np.array([maximum, maximum - 1, maximum - 2], dtype=np.uint64)
    y = np.array([1, 1, 2], dtype=np.uint64)
    assert _pareto_frontier_mask(x, y).tolist() == [False, True, True]


@pytest.mark.parametrize("count", [256, 1024])
def test_large_frontier_does_not_compare_every_observation_pair(count):
    comparisons = []

    class ObservedArray(np.ndarray):
        def __array_ufunc__(self, function, method, *inputs, **kwargs):
            if function in (np.less, np.less_equal, np.greater, np.greater_equal, np.equal, np.not_equal):
                comparisons.append(max(np.size(item) for item in inputs))
            arrays = tuple(np.asarray(item) if isinstance(item, ObservedArray) else item for item in inputs)
            return getattr(function, method)(*arrays, **kwargs)

    x = np.arange(count, dtype=float).view(ObservedArray)
    y = x[::-1]
    assert _pareto_frontier_mask(x, y).all()
    assert sum(comparisons) <= 16 * count * math.ceil(math.log2(count))
