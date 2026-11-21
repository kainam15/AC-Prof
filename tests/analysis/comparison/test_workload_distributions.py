"""Detailed comparison preserves sparse request evidence at bounded traversal cost."""
from collections import Counter

import pytest

from acprof.analysis import comparison, conditions


def _cases(*weighted_facts):
    return {"sample": {
        "request_count": sum(count for count, _ in weighted_facts),
        "variants": [{"count": count, "contract": {"schema_version": 1, **facts}}
                     for count, facts in weighted_facts],
    }}


@pytest.mark.parametrize("width", [4, 64])
def test_flattening_cost_per_variant_is_bounded_as_contracts_widen(monkeypatch, width):
    root_calls: Counter[int] = Counter()
    dimensions = conditions.dimensions

    def counted(contract, prefix=""):
        if not prefix:
            root_calls[id(contract)] += 1
        return dimensions(contract, prefix)

    monkeypatch.setattr(conditions, "dimensions", counted)
    monkeypatch.setattr(comparison, "dimensions", counted, raising=False)
    fields = {f"field_{index}": index for index in range(width)}
    left = _cases((2, {"input": {**fields, "value": 1}}), (1, {"input": {**fields, "value": 2}}))
    right = _cases((1, {"input": {**fields, "value": 1}}), (2, {"input": {**fields, "value": 2}}))
    condition = comparison._workload_condition(left, right)
    assert condition["changed_dimensions"] == {"input.value": {"sample": {
        "left": {"1": "2/3", "2": "1/3"}, "right": {"1": "1/3", "2": "2/3"},
    }}}
    # Allow separate profile and marginal passes, independent of field count.
    assert max(root_calls.values(), default=0) <= 2


def test_sparse_marginals_use_all_requests_and_preserve_json_types():
    left = _cases((1, {"typed": False, "only_left": 7}), (2, {"typed": 0, "nullable": None}),
                  (3, {"typed": 0.0}), (4, {"nullable": None}))
    right = _cases((1, {"typed": False}), (3, {"typed": 0}), (2, {"typed": 0.0}),
                   (4, {"typed": None, "nullable": 1}))
    condition = comparison._workload_condition(left, right)
    assert condition["status"] == "unknown"
    assert condition["reason"] == "actual_workload_evidence_incomplete"
    assert condition["request_counts_changed"] is False
    assert condition["changed_dimensions"] == {
        "nullable": {"sample": {"left": {"null": "1"}, "right": {"null": "3/5", "1": "2/5"}}},
        "only_left": {"sample": {"left": {"7": "1/10", "null": "9/10"}, "right": {"null": "1"}}},
        "typed": {"sample": {
            "left": {"false": "1/10", "0": "1/5", "0.0": "3/10", "null": "2/5"},
            "right": {"false": "1/10", "0": "3/10", "0.0": "1/5", "null": "2/5"},
        }},
    }


def test_joint_change_with_equal_marginals_reports_exact_mixtures():
    left = _cases((1, {"x": 1, "y": 1}), (1, {"x": 2, "y": 2}))
    right = _cases((1, {"x": 1, "y": 2}), (1, {"x": 2, "y": 1}))
    condition = comparison._workload_condition(left, right)
    assert condition["status"] == "incompatible"
    assert condition["reason"] == "per_request_distribution_changed"
    assert condition["request_counts_changed"] is False
    assert condition["changed_dimensions"] == {"joint_distribution": {"sample": {
        "left": {'{"schema_version":1,"x":1,"y":1}': "1/2", '{"schema_version":1,"x":2,"y":2}': "1/2"},
        "right": {'{"schema_version":1,"x":1,"y":2}': "1/2", '{"schema_version":1,"x":2,"y":1}': "1/2"},
    }}}
