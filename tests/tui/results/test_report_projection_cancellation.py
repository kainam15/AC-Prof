"""Cancelling an imported report stops projection while preserving worker ownership."""
import asyncio
import json
from concurrent.futures import CancelledError
from threading import Event

import pytest

from acprof.host import run_state
from acprof.tui.results import reports
from tests.support.tui_actions import open_page
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition


async def test_cancel_statistics_import_releases_run_after_current_group(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "MEASUREMENT_LOCK_ROOT", tmp_path)
    source = tmp_path / "stats.json"
    group = dict(cpu_cores=2, mem_cap_gb=8, gpu_mode="off", metric="latency_app_s", unit="s",
                 n_windows=1, missing_windows=0, mean=0.02)
    source.write_text(json.dumps(dict(schema_version=1, resampling_unit="csv_request_window", confidence=0.95,
                                     filter="status=ok and warmup=0", groups=[dict(group, input_scale=i) for i in (1, 2, 3)])))
    entered, release = Event(), Event()
    rendered = []
    original = reports.ReportRow

    def observe(*args, **kwargs):
        row = original(*args, **kwargs)
        rendered.append(row)
        if len(rendered) == 1:
            entered.set()
            assert release.wait(10)
        return row

    monkeypatch.setattr(reports, "ReportRow", observe)
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        await open_page(pilot, "results-tab")
        app.results.open_report(str(source))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            app.slash.execute_command("/cancel")
            assert not app.operation_state().allows("run")
            with pytest.raises(run_state.RunStateError), run_state.MeasurementLock():
                pass
        finally:
            release.set()
        await wait_for_condition(pilot, lambda: not app.read_jobs, description="cancelled report worker exits", timeout=4)
        assert len(rendered) == 1
        assert app.results.report_view is None
        assert app.operation_state().allows("run")
        with run_state.MeasurementLock():
            pass
        app.results.open_report(str(source))
        await wait_for_condition(pilot, lambda: app.results.report_view is not None, description="same report reopened", timeout=4)
        assert len(app.results.report_view.rows) == 3


@pytest.mark.parametrize("kind", ["monitor_overhead_diagnostic", "ui_overhead", "independent_experiment_comparison",
                                  "coverage"])
def test_cancel_after_first_report_row_never_returns_partial_view(tmp_path, monkeypatch, kind):
    overhead = dict(paired_mean_change_pct=1, ci_low_pct=0, ci_high_pct=2)
    if kind == "coverage":
        data = dict(scope="selected_sample_only; no_formal_measurement", rows=[{"model_id": "model"}] * 3)
    elif kind == "monitor_overhead_diagnostic":
        data = dict(kind=kind, successful=True, gpu_mode="off", comparisons=[dict(overhead, paired_rounds=3)] * 3)
    elif kind == "ui_overhead":
        data = dict(kind=kind, successful=True, ui="headless", pairs=[{}] * 3, **overhead)
    else:
        group = dict(cpu_cores=1, mem_cap_gb=4, gpu_mode="off", input_scale=2, metric="latency_app_s", unit="s",
                     left={"n_runs": 3, "mean": 0.2}, right={"n_runs": 3, "mean": 0.3})
        data = dict(kind=kind, resampling_unit="independent_run_mean", direction="right_minus_left_and_right_divided_by_left",
                    confidence=0.95, status="compatible", groups=[group] * 3)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(dict(schema_version=1, **data)))
    before = reports.read_report(path)
    seen = []
    original = reports.ReportRow

    def observe(*args, **kwargs):
        row = original(*args, **kwargs)
        seen.append(row)
        return row

    monkeypatch.setattr(reports, "ReportRow", observe)
    with pytest.raises(CancelledError):
        reports.read_report(path, cancelled=lambda: bool(seen))
    assert seen == [before.rows[0]]


@pytest.mark.parametrize("field", ["pairs", "attempts"])
def test_cancel_during_pair_validation_or_nested_coverage_detail(tmp_path, monkeypatch, field):
    if field == "pairs":
        data = dict(kind="ui_overhead", successful=True, ui="headless", paired_mean_change_pct=1,
                    ci_low_pct=0, ci_high_pct=2)
    else:
        data = dict(scope="recorded_results; no_reexecution", rows=[{"model_id": "model"}])
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"schema_version": 1, **data, field: [{}] * 1024}))
    entries = []

    class ObservedList(list):
        def __iter__(self):
            for item in super().__iter__():
                entries.append(item)
                yield item

    decode = json.loads

    def observe(data):
        payload = decode(data)
        payload[field] = ObservedList(payload[field])
        return payload

    monkeypatch.setattr(reports.json, "loads", observe)
    with pytest.raises(CancelledError):
        reports.read_report(path, cancelled=lambda: bool(entries))
    assert 0 < len(entries) < 1024
