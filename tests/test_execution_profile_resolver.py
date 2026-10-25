"""Indexed plan resolution preserves persisted-plan and flat-result semantics."""
from __future__ import annotations

import copy
import gc
import math
import weakref

import pytest

from acprof.host.profiling import execution_plan as plans


def _tool(value=123.0, **overrides):
    tool = {"entries": [{"input_scale": 1.0, plans.MASSIF_HEAP_PEAK_FIELD: value}]}
    tool.update(overrides)
    return tool


def _case(tools, *, cpu=2, memory=4, mode="off"):
    return {"cpu_cores": cpu, "mem_cap_gb": memory, "gpu_mode": mode, "tools": tools}


def _plan():
    return {"profiles": [_case({"massif": _tool()})]}


def _assert_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for field, value in expected.items():
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(actual[field]), field
        else:
            assert actual[field] == value, field


def test_resource_lookup_prefers_first_case_containing_requested_tool():
    plan = {"profiles": [
        _case({"nsys": {"entries": []}}),
        _case({"massif": _tool(123)}),
        _case({"massif": _tool(456)}),
    ]}
    result = plans.ExecutionProfileResolver(plan).resolve(2, 4, "off", 1.0)
    assert result[plans.MASSIF_HEAP_PEAK_FIELD] == 123
    assert result[plans.MASSIF_ERROR_FIELD] == ""
    assert math.isnan(result[plans.NSYS_HOST_WALL_TIME_FIELD])


@pytest.mark.parametrize(("first", "expected_error"), [
    (None, "execution_profile_missing_case:massif:cpu_cores=2,mem_cap_gb=4,gpu_mode=off"),
    ({"enabled": False}, ""),
    ({"status": "disabled"}, ""),
    ({"enabled": "off"}, ""),
])
def test_first_tool_presence_wins_over_later_valid_profile(first, expected_error):
    plan = {"profiles": [_case({"massif": first}), _case({"massif": _tool()})]}
    result = plans.ExecutionProfileResolver(plan).resolve(2, 4, "off", 1.0)
    assert result[plans.MASSIF_ERROR_FIELD] == expected_error
    assert math.isnan(result[plans.MASSIF_HEAP_PEAK_FIELD])


def test_scale_tolerance_keeps_first_matching_entry_and_error_override():
    epsilon = plans.INPUT_SCALE_ABS_TOLERANCE
    tool = _tool(error="tool failed")
    tool["entries"][0][plans.MASSIF_ERROR_FIELD] = ""
    tool["entries"].append({
        "input_scale": 1.0 + epsilon * 0.75,
        plans.MASSIF_HEAP_PEAK_FIELD: 456,
        "error": "later entry",
    })
    resolver = plans.ExecutionProfileResolver({"profiles": [_case({"massif": tool})]})
    result = resolver.resolve("2.0", "4", " OFF ", 1.0 + epsilon * 0.5)
    assert result[plans.MASSIF_HEAP_PEAK_FIELD] == 123
    assert result[plans.MASSIF_ERROR_FIELD] == ""
    missing = resolver.resolve(2, 4, "off", 1.1)
    assert missing[plans.MASSIF_ERROR_FIELD] == "execution_profile_missing_scale:massif:1.1"
    assert math.isnan(missing[plans.MASSIF_HEAP_PEAK_FIELD])


def test_snapshot_and_each_returned_row_are_independent_of_mutation():
    plan = _plan()
    resolver = plans.ExecutionProfileResolver(plan)
    plan["profiles"][0]["tools"]["massif"]["entries"][0][plans.MASSIF_HEAP_PEAK_FIELD] = 456
    assert resolver.resolve(2, 4, "off", 1.0)[plans.MASSIF_HEAP_PEAK_FIELD] == 123
    assert plans.ExecutionProfileResolver(plan).resolve(2, 4, "off", 1.0)[plans.MASSIF_HEAP_PEAK_FIELD] == 456
    plan["profiles"].clear()
    row = resolver.resolve(2, 4, "off", 1.0)
    row[plans.MASSIF_HEAP_PEAK_FIELD] = 789
    row[plans.MASSIF_ERROR_FIELD] = "caller changed result"
    another = resolver.resolve(2, 4, "off", 1.0)
    assert another[plans.MASSIF_HEAP_PEAK_FIELD] == 123
    assert another[plans.MASSIF_ERROR_FIELD] == ""


def test_snapshot_does_not_retain_unrelated_report_payload():
    class ReportPayload:
        pass

    payload = ReportPayload()
    reference = weakref.ref(payload)
    plan = _plan()
    plan["profiles"][0]["tools"]["massif"]["entries"][0]["report"] = payload
    resolver = plans.ExecutionProfileResolver(plan)
    del payload, plan
    gc.collect()
    assert reference() is None
    assert resolver.resolve(2, 4, "off", 1.0)[plans.MASSIF_HEAP_PEAK_FIELD] == 123


@pytest.mark.parametrize(("plan", "query"), [
    (None, (2, 4, "off", 1.0)),
    ([], (2, 4, "on", 1.0)),
    ({"_disabled": True, "_load_error": "bad"}, (2, 4, "on", 1.0)),
    ({"_load_error": "bad", "enabled_tools": []}, (2, 4, "off", 1.0)),
    ({"profiles": []}, (2, 4, "off", 1.0)),
    ({"profiles": [], "enabled_tools": "massif;nsys"}, (2, 4, "on", 1.0)),
    ({"profiles": [], "enabled_tools": {"massif": {"enabled": True}}}, (2, 4, "off", 1.0)),
    ({"profiles": [None, _case({"massif": _tool()}, cpu=True)]}, (2, 4, "off", 1.0)),
    ({"profiles": [_case({"massif": _tool()})]}, (True, 4, "off", 1.0)),
    ({"profiles": [_case({"massif": _tool()})]}, (2, "bad", "off", 1.0)),
    ({"profiles": [_case({"massif": _tool()})]}, (2, 4, "off", float("nan"))),
    ({"profiles": [_case({"massif": _tool()})]}, (2, 4, "off", float("inf"))),
    ({"profiles": [_case({"massif": {"error": "tool error"}})]}, (2, 4, "off", 1.0)),
    ({"profiles": [_case({"massif": {"entries": None}})]}, (2, 4, "off", 1.0)),
    ({"profiles": [_case({"massif": {"entries": [None, {"input_scale": "nan"}, {"input_scale": "1.0", "error": "entry error", plans.MASSIF_HEAP_PEAK_FIELD: "inf"}]}})]}, (2, 4, "off", 1.0)),
    ({"profiles": [_case({"massif": _tool()}, mode="unexpected")]}, (2, 4, "unexpected", 1.0)),
    ({"profiles": [_case({"nsys": {"entries": [{"input_scale": 1.0, plans.NSYS_HOST_WALL_TIME_FIELD: "12.5", "error": "partial"}]}}, mode=" ON ")]}, (2, 4, "on", 1.0)),
])
def test_indexed_lookup_matches_existing_reader_for_persisted_plan_edge_cases(plan, query):
    expected = plans.find_execution_profile_entry(copy.deepcopy(plan), *query)
    resolver = plans.ExecutionProfileResolver(plan)
    _assert_equal(resolver.resolve(*query), expected)
    _assert_equal(plans.find_execution_profile_entry(resolver, *query), expected)
