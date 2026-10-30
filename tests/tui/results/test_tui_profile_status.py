"""Read-only posthoc status shares the collector's exact completion rules."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host.posthoc.context import TOOL_ERROR_FIELD, TOOL_METRIC_FIELDS
from acprof.tui.results.profile_status import inspect_profile_result


def _complete_row(mode: str, *tools: str) -> dict[str, str]:
    row = {"gpu_mode": mode}
    for tool in tools:
        row.update({field: "1" for field in TOOL_METRIC_FIELDS[tool]})
        row[TOOL_ERROR_FIELD[tool]] = ""
    return row


def test_gpu_result_shows_complete_partial_failed_and_inapplicable(tmp_path):
    first = _complete_row("on", "torch", "ncu", "nsys")
    second = _complete_row("on", "torch")
    second[TOOL_ERROR_FIELD["nsys"]] = "permission denied"
    context = SimpleNamespace(result_dir=tmp_path, rows=[first, second])
    with patch("acprof.tui.results.profile_status.load_result_context", return_value=context) as load:
        overview = inspect_profile_result(tmp_path / "result_layers.json")
    load.assert_called_once_with(tmp_path, result_snapshot=None, cancelled=None)
    statuses = {status.tool: status for status in overview.tools}
    assert [status.state for status in overview.tools] == [
        "complete", "partial", "failed", "inapplicable"]
    assert (statuses["ncu"].completed, statuses["ncu"].total) == (1, 2)
    assert (statuses["nsys"].errors, statuses["nsys"].total) == (1, 2)
    assert overview.complete == 1
    assert overview.pending == 2
    assert overview.applicable == 3
    assert overview.suggested_tools == ("ncu",)


def test_cpu_only_result_cannot_offer_gpu_tools(tmp_path):
    context = SimpleNamespace(result_dir=tmp_path, rows=[_complete_row("off", "massif")])
    with patch("acprof.tui.results.profile_status.load_result_context", return_value=context):
        overview = inspect_profile_result(tmp_path)
    assert [status.state for status in overview.tools] == [
        "missing", "inapplicable", "inapplicable", "complete"]
    assert overview.suggested_tools == ("torch",)


def test_standalone_csv_rejected_without_loading_posthoc(tmp_path):
    with patch("acprof.tui.results.profile_status.load_result_context") as load:
        with pytest.raises(ValueError, match="独立 CSV"):
            inspect_profile_result(tmp_path / "old.csv")
    load.assert_not_called()
