"""Host-only measurement branch coverage gate contracts."""
from copy import deepcopy

import pytest

from scripts.check_critical_coverage import BRANCH_FLOORS, check_report


def baseline():
    return {"files": {name: {"summary": {"covered_branches": 90, "num_branches": 100}}
                      for name in BRANCH_FLOORS}}


def test_all_critical_paths_are_present_and_pass():
    assert {"acprof/host/client.py", "acprof/monitors/energy_cpu.py",
            "acprof/host/window_execution.py", "acprof/host/client_metrics.py",
            "acprof/result_layers.py"} <= set(BRANCH_FLOORS)
    assert check_report(baseline()) == []


@pytest.mark.parametrize("failure", ("missing_file", "missing_branch_count", "zero_branches",
                                      "low_branch_coverage", "invalid_counter"))
def test_critical_path_regressions_fail_closed(failure):
    report = deepcopy(baseline())
    target = "acprof/host/client.py"
    item = report["files"][target]["summary"]
    if failure == "missing_file":
        del report["files"][target]
    elif failure == "missing_branch_count":
        del item["num_branches"]
    elif failure == "zero_branches":
        item["num_branches"] = 0
    elif failure == "low_branch_coverage":
        item["covered_branches"] = 1
    elif failure == "invalid_counter":
        item["covered_branches"] = 101
    errors = check_report(report)
    assert len(errors) == 1 and target in errors[0]


def test_empty_report_cannot_pass():
    assert check_report({}) == ["coverage report is missing file data"]
