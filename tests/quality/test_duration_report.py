"""Slow-test diagnosis must use existing evidence without rerunning the suite."""

import pytest

from scripts.report_test_durations import rank_durations


def evidence(*rows, successful=True):
    return {
        "schema_version": 1, "successful": successful, "duration_s": 28.0,
        "tests": [{"id": name, "duration_s": duration} for name, duration in rows],
    }


def test_group_parameterized_cases_and_rank_by_cumulative_time():
    result = rank_durations(evidence(
        ("tests/tui/test_view.py::test_layout[en]", 2.0),
        ("tests/tui/test_view.py::test_layout[zh]", 3.5),
        ("tests/analysis/test_stat.py::test_stat", 4.0),
        ("tests/tui/test_view.py::test_other", 0.5),
    ))
    assert result["recorded_cases"] == 4
    assert result["recorded_test_seconds"] == 10.0
    assert result["files"] == [
        {"name": "tests/tui/test_view.py", "seconds": 6.0, "cases": 3},
        {"name": "tests/analysis/test_stat.py", "seconds": 4.0, "cases": 1},
    ]
    assert result["groups"][0] == {
        "name": "tests/tui/test_view.py::test_layout", "seconds": 5.5, "cases": 2,
    }


def test_partial_report_is_not_misrepresented_as_passed():
    result = rank_durations(evidence(("tests/test_a.py::test_case", 1.0), successful=False))
    assert not result["successful"]
    assert result["recorded_cases"] == 1


@pytest.mark.parametrize("duration", [-0.01, float("nan"), float("inf"), None, "slow"])
def test_invalid_durations_are_not_hidden(duration):
    with pytest.raises(ValueError, match="invalid test duration"):
        rank_durations(evidence(("tests/test_bad.py::test_case", duration)))


def test_reject_unknown_schema_empty_id_and_non_positive_top():
    with pytest.raises(ValueError, match="schema"):
        rank_durations({"schema_version": 2, "tests": []})
    with pytest.raises(ValueError, match="ID"):
        rank_durations(evidence(("", 1.0)))
    with pytest.raises(ValueError, match="positive"):
        rank_durations(evidence(), top=0)
