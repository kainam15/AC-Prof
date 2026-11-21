"""Text-scale selection uses the probe's effective lengths and original payloads."""

import math

import pytest

from acprof.host.input_plan.text_scales import select_text_scale_probes


def test_ordinary_targets_keep_the_larger_equidistant_scale_and_its_payload():
    def probe(words):
        return {"payload": {"words": words}, "effective_input_scale": 2 * words + 1, "truncated_by_limit": False}

    selected = select_text_scale_probes(25, probe)
    assert [item["effective_input_scale"] for item in selected] == [5, 9, 13, 17, 21, 25]
    assert [item["payload"]["words"] for item in selected] == [2, 4, 6, 8, 10, 12]


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -1])
def test_invalid_observations_stop_selection(invalid):
    with pytest.raises(RuntimeError, match="invalid effective text input scale"):
        select_text_scale_probes(12, lambda _: {"effective_input_scale": invalid, "truncated_by_limit": False})


@pytest.mark.parametrize("distinct", [4, 6])
def test_narrow_band_between_long_plateaus_is_found_with_bounded_probes(distinct):
    maximum = 4096
    probed = {}

    def probe(words):
        assert words not in probed
        first = maximum - distinct + 1
        scale = min(maximum, first + max(0, words - 2344))
        result = {"payload": {"words": words}, "effective_input_scale": scale, "truncated_by_limit": False}
        probed[words] = result
        return result

    selected = select_text_scale_probes(maximum, probe)
    assert [item["effective_input_scale"] for item in selected] == list(range(maximum - distinct + 1, maximum + 1))
    assert all(item is probed[item["payload"]["words"]] for item in selected)
    assert len(probed) <= 12 * (math.ceil(math.log2(maximum)) + 2)


@pytest.mark.parametrize("flag", ["limit_exceeded", "truncated_by_limit"])
def test_rejected_candidates_never_substitute_for_missing_lengths(flag):
    def probe(words):
        return {"payload": {"words": words}, "effective_input_scale": 7 + min(3, words),
                "truncated_by_limit": False, flag: words >= 3}

    selected = select_text_scale_probes(12, probe)
    assert [item["payload"]["words"] for item in selected] == [1, 2]


def test_probe_failure_while_filling_a_collision_propagates():
    def probe(words):
        if words == 2345:
            raise TimeoutError("tokenizer probe disconnected")
        return {"effective_input_scale": min(4096, 4091 + max(0, words - 2344)), "truncated_by_limit": False}

    with pytest.raises(TimeoutError, match="tokenizer probe disconnected"):
        select_text_scale_probes(4096, probe)
