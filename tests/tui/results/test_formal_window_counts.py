"""The Results overview and preview agree on exact CSV window phases."""
from unittest.mock import patch

import pytest
from textual.widgets import Button

from acprof.results.csv import ResultValidationError
from acprof.tui.results.data_status import inspect_experiment_data
from acprof.tui.run.diagnostics import summarize_result_csv
from tests.analysis.reporting.test_report_statistics import row, write_rows
from tests.support.tui_actions import open_page
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition


async def test_decimal_formal_window_enables_plot_action(tmp_path):
    source = tmp_path / "result_all.csv"
    write_rows(source, [row(0, 2, warmup="0.0"),
                        row(1, 4, warmup="-0E5", result_origin="inferred_not_measured")])
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(120, 30)) as pilot:
        await open_page(pilot, "results-tab")
        app.results.select_result(str(source))
        await wait_for_condition(
            pilot, lambda: app.results.data_status is not None and not app.read_jobs,
            description="both Results readers to finish for the selected CSV", timeout=5)
        plot = app.query_one("#plot-results", Button)
        assert not plot.disabled
        assert not app.query_one("#report-window-stats", Button).disabled
        assert app.results.formal_ok_rows == 1
        with patch.object(app, "launch") as launch:
            assert await pilot.click(plot)
            await wait_for_condition(pilot, lambda: launch.called, description="the selected PNG command", timeout=3)
        pending, = launch.call_args.args
        assert pending.kind == "plot"
        assert str(source) in pending.command


@pytest.mark.parametrize(("phase", "formal", "warmup"), [
    ("0", 1, 0), ("0.0", 1, 0), ("-0E5", 1, 0), ("+0e-1000", 1, 0),
    ("1", 0, 1), ("1.00", 0, 1), ("1e-324", 0, 0), ("1.0000000000000001", 0, 0),
    (None, 1, 0), ("", 0, 0), ("False", 0, 0), ("nan", 0, 0), ("2", 0, 0),
])
def test_preview_counts_on_legacy_rows_without_window_identity(tmp_path, phase, formal, warmup):
    source = tmp_path / "result_all.csv"
    record = row(0, 2, warmup=phase)
    del record["repeat_idx"]  # The TUI can show counts even when statistics cannot group windows.
    if phase is None:
        del record["warmup"]
    write_rows(source, [record])
    assert inspect_experiment_data(source).formal_ok_rows == formal
    summary = summarize_result_csv(source)
    assert summary.formal_ok_rows == formal
    assert summary.warmup_rows == warmup
    assert summary.ok_rows == 1


def test_strict_statistics_keeps_rejecting_a_nonnumeric_phase(tmp_path):
    source = tmp_path / "result_all.csv"
    write_rows(source, [row(0, 2, warmup="false")])
    assert inspect_experiment_data(source).formal_ok_rows == 0
    with pytest.raises(ResultValidationError, match="invalid warmup"):
        summarize_result_csv(source)
