"""Streaming samples retain circular blocks, draw order and bounded workspace."""
import random
import tracemalloc

import pytest

from acprof.analysis.uncertainty import bootstrap_mean_interval


@pytest.mark.parametrize("block_size", [1, 7])
def test_large_resample_does_not_allocate_a_window_sized_list(block_size):
    values = tuple(float(index % 17 + 1) for index in range(50000))
    tracemalloc.start()
    try:
        low, high = bootstrap_mean_interval(values, resamples=1, seed=3, block_size=block_size)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert low == high
    assert 1 <= low <= 17
    assert peak < 128 * 1024, f"resampling workspace retained {peak:,} bytes for one estimate"


@pytest.mark.parametrize(("values", "block_size", "starts", "expected"), [
    (tuple(range(1, 8)), 1, [0] * 7 + [3] * 7 + [6] * 7, (2.5, 5.5)),
    # Nine observations: wrap the first block and keep one item of the final block.
    (tuple(range(1, 10)), 2, [8, 1, 3, 5, 7] + [0] * 5 + [7] * 5, (29 / 9, 121 / 18)),
    (tuple(range(1, 10)), 3, [8, 2, 5] + [0] * 3 + [6] * 3, (3.5, 6.5)),
])
def test_exact_draws_preserve_percentile_interpolation_and_circular_blocks(
    monkeypatch, values, block_size, starts, expected,
):
    positions = iter(starts)
    consumed = []

    class RecordedDraws:
        def __init__(self, seed):
            assert seed == "recorded-draws"

        def randrange(self, count):
            assert count == len(values)
            value = next(positions)
            consumed.append(value)
            return value

    monkeypatch.setattr("acprof.analysis.uncertainty.random.Random", RecordedDraws)
    interval = bootstrap_mean_interval(values, confidence=0.5, resamples=3,
                                       seed="recorded-draws", block_size=block_size)
    assert interval == pytest.approx(expected)
    assert consumed == starts


def test_seeded_sampling_does_not_change_global_random_state():
    before = random.getstate()
    first = bootstrap_mean_interval([1, 3, 5, 7, 9, 11], seed=29, resamples=31, block_size=2)
    assert bootstrap_mean_interval([1, 3, 5, 7, 9, 11], seed=29, resamples=31, block_size=2) == first
    assert random.getstate() == before
