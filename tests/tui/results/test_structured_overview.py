"""Structured result summaries preserve evidence and existing recovery gates."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, Collapsible, ProgressBar, Static

from acprof.tui.experiments.catalog import ExperimentRecord
from acprof.tui.i18n import translate
from acprof.tui.results.data_status import ExperimentDataStatus
from acprof.tui.results.overview import ExperimentOverview, overview_view
from acprof.tui.results.overview_panel import ExperimentOverviewPanel
from acprof.tui.ui.structured import ActionButtons, FactGrid, StatusBanner
from tests.support.tui_fixtures import AcprofTui


@pytest.fixture
def recorded_overview(tmp_path):
    directory = tmp_path / "研究模型-2026-11-08"
    record = ExperimentRecord(
        directory=directory,
        aliases=(directory,),
        run_id="fixture-run-2026-11-08-abcdefghijklmnop",
        model_id="fixture/中文-model",
        revision="1234567890abcdef1234567890abcdef12345678",
        runtime_profile="transformers-cpu",
        created_at="2026-11-08T10:20:30.456+08:00",
        status="interrupted",
        device="cpu",
        options={"model": "fixture/中文-model", "cpus": "1,2", "mems": "4", "gpus": "off"},
        state={
            "runtime": {"kind": "process", "task": {"pipeline_tag": "text-classification"}},
            "attempts": [
                {"started_at": "2026-11-08T10:21:00+08:00", "ended_at": "2026-11-08T10:22:01+08:00"},
            ],
            "cases": {
                "first": {"status": "complete"},
                "second": {"status": "interrupted"},
                "third": {"status": "pending"},
                "fourth": {"status": "pending"},
            },
        },
        metadata={"environment_class": "Native Linux", "profiling_mode": "basic"},
        validation={},
        failures=(),
    )
    return ExperimentOverview(
        directory, directory / "result_layers.json", record, "exists", "exists",
    )


def checked_data(overview, rows):
    return ExperimentDataStatus(
        path=overview.result_file,
        sha256="a" * 64,
        formal_ok_rows=rows,
        audit={"valid": True, "issues": [], "counts": {"ok": rows, "error": 0, "warn": 0, "warmup": 0}},
        profiler_overview=None,
        profiler_error="",
    )


def fact(view, key, *, details=False):
    return next(
        row
        for section in view.sections
        for row in (section.detail_rows if details else section.rows)
        if getattr(row.label, "msgid", None) == key
    )


def test_metadata_projection_performs_no_file_reads_or_csv_aggregation(recorded_overview):
    with (
        patch.object(Path, "open", side_effect=AssertionError("view must not open files")),
        patch("acprof.results.csv.read_result_csv_snapshot", side_effect=AssertionError("view must not read CSV")),
    ):
        view = overview_view(recorded_overview)
    assert view.completed_cases == 1
    assert view.total_cases == 4
    assert "1/4" in view.summary.headline
    assert translate(fact(view, "results.view.windows").value, "en") == "Checking windows"
    assert fact(view, "results.view.model").value == "fixture/中文-model"


@pytest.mark.parametrize("cases", [None, {}, []])
def test_unknown_case_totals_do_not_manufacture_zero_percent(recorded_overview, cases):
    record = replace(recorded_overview.record, state={**recorded_overview.record.state, "cases": cases})
    view = overview_view(replace(recorded_overview, record=record))
    assert view.total_cases is None
    assert "0/0" not in view.summary.headline
    assert "Case progress not recorded" in translate(view.summary.headline, "en")


def test_confirmed_zero_windows_differ_from_pending_and_unverified_data(recorded_overview):
    pending = overview_view(recorded_overview)
    zero = overview_view(recorded_overview, checked_data(recorded_overview, 0))
    error = "  PermissionError: /tmp/中文结果/result_layers.json\n"
    unverified = overview_view(recorded_overview, data_error=error)

    assert translate(fact(pending, "results.view.windows").value, "en") == "Checking windows"
    assert fact(zero, "results.view.windows").value == "0"
    assert "No valid formal windows" in translate(zero.summary.headline, "en")
    assert translate(fact(unverified, "results.view.windows").value, "en") == "Windows unverified"
    assert "No valid formal windows" not in translate(unverified.summary.headline, "en")
    assert unverified.summary.level == "error"
    assert unverified.callout is not None
    assert unverified.callout.detail == error
    assert translate(fact(zero, "results.view.result_files").value, "en") == "Created; see window checks"


def test_completed_state_with_zero_valid_windows_is_not_presented_as_success(recorded_overview):
    overview = replace(recorded_overview, record=replace(recorded_overview.record, status="complete"))
    view = overview_view(overview, checked_data(overview, 0))
    assert view.summary.level == "warning"
    assert view.callout is not None
    assert translate(view.callout.problem, "en") == "No valid formal windows"
    assert not any(action.id in {"overview-statistics", "overview-resume"} for action in view.summary.actions)


def test_four_fact_groups_keep_identifiers_times_and_paths_in_details(recorded_overview):
    view = overview_view(recorded_overview)
    assert [translate(section.title, "en") for section in view.sections] == [
        "Identity", "Run", "Resources", "Artifacts",
    ]
    visible_keys = {getattr(row.label, "msgid", None) for section in view.sections for row in section.rows}
    detail_keys = {getattr(row.label, "msgid", None) for section in view.sections for row in section.detail_rows}
    required_details = {
        "results.view.revision", "results.view.run_id", "results.view.created",
        "results.view.started", "results.view.ended", "results.view.directory", "results.view.result_path",
    }
    assert required_details <= detail_keys
    assert required_details.isdisjoint(visible_keys)
    assert fact(view, "results.view.revision", details=True).value == recorded_overview.record.revision
    assert fact(view, "results.view.run_id", details=True).value == recorded_overview.record.run_id
    assert fact(view, "results.view.created", details=True).detail == "2026-11-08T10:20:30.456+08:00"
    assert fact(view, "results.view.directory", details=True).value == str(recorded_overview.source)
    assert fact(view, "results.view.result_path", details=True).value == str(recorded_overview.result_file)
    assert fact(view, "results.view.directory", details=True).copyable


@pytest.mark.parametrize("attempts", [None, 42, {"invalid": "not a list"}])
def test_incomplete_attempt_metadata_keeps_unknown_times_instead_of_crashing(recorded_overview, attempts):
    record = replace(recorded_overview.record, state={**recorded_overview.record.state, "attempts": attempts})
    view = overview_view(replace(recorded_overview, record=record))
    assert translate(fact(view, "results.view.started", details=True).value, "en") == "Unknown"
    assert translate(fact(view, "results.view.ended", details=True).value, "en") == "Unknown"


def test_all_recorded_failures_and_audit_evidence_remain_verbatim(recorded_overview):
    failures = (
        {"reason_code": "access_denied", "detail": "  PermissionError: /home/用户/model\n"},
        {"reason_code": "runtime_dependency_missing", "detail": "ModuleNotFoundError('模型')\nline 2"},
        {"reason_code": "request_timeout", "detail": "timeout at /tmp/source.py:30"},
        {"reason_code": "future_failure", "detail": "[red] raw exception [/red]\n  trailing  "},
        {"reason_code": "inference_failed", "detail": "fifth failure must not be dropped\n"},
    )
    record = replace(recorded_overview.record, failures=failures)
    data = replace(
        checked_data(recorded_overview, 2),
        audit={"valid": False, "issues": [{"message": "CSV issue: /tmp/保留.csv\n  source"}]},
        profiler_error="raw profiler exception\n",
    )
    view = overview_view(replace(recorded_overview, record=record), data)
    assert view.callout is not None
    for failure in failures:
        assert failure["reason_code"] + "\n" + failure["detail"] in view.callout.detail
    assert "CSV issue: /tmp/保留.csv\n  source" in view.callout.detail
    assert "raw profiler exception\n" in view.callout.detail
    assert "access_denied" not in view.summary.headline
    assert "future_failure" not in view.summary.headline


@pytest.mark.parametrize(("language", "expected"), [("zh", "未知"), ("en", "Unknown")])
def test_unknown_raw_states_and_codes_stay_out_of_headlines(recorded_overview, language, expected):
    record = replace(recorded_overview.record, status="future_backend_state", failures=(
        {"reason_code": "future_reason_code", "detail": "/raw/fixture/error.log"},
    ))
    view = overview_view(replace(recorded_overview, record=record))
    text = translate(view.summary.headline, language)
    assert text.startswith(expected + " · ")
    assert "future_backend_state" not in text
    assert "future_reason_code" not in text
    assert view.callout is not None
    assert view.callout.detail == "future_reason_code\n/raw/fixture/error.log"


@pytest.mark.parametrize(("language", "size"), [("zh", (80, 24)), ("en", (120, 30))])
async def test_overview_progress_has_no_animation_and_freezes_during_measurement(
        recorded_overview, tmp_path, language, size):
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size) as pilot:
        app.activate_tab("results-tab")
        app.results.overview = recorded_overview
        app.results.data_status = checked_data(recorded_overview, 2)
        app.results.render_overview()
        await pilot.pause()
        panel = app.query_one(ExperimentOverviewPanel)
        progress = panel.query_one(ProgressBar)
        assert (progress.progress, progress.total) == (1, 4)
        assert not progress.show_eta
        assert not progress.query("#eta")
        assert all(widget.auto_refresh is None for widget in [progress, *progress.query("*")])
        assert len(panel.query(FactGrid)) == 4
        assert all(group.collapsed for group in panel.query(".fact-more").results(Collapsible))
        assert "1/4" in str(app.query_one("#result-summary", Static).content)

        # Unknown totals remain textual; never switch the native Bar to animated None.
        unknown = replace(recorded_overview, record=replace(recorded_overview.record, state={}))
        app.results.overview = unknown
        app.results.render_overview()
        await pilot.pause()
        assert not panel.query_one("#overview-case-progress").display
        assert all(widget.auto_refresh is None for widget in [progress, *progress.query("*")])

        retained = panel.view
        original = app.execution.session.snapshot
        app.execution.session.snapshot = replace(original, measurement_active=True)
        try:
            app.results.overview = recorded_overview
            with patch.object(panel, "update_view") as update:
                app.results.render_overview()
            update.assert_not_called()
            assert panel.view is retained
        finally:
            app.execution.session.snapshot = original


@pytest.mark.parametrize(("language", "size"), [("zh", (80, 24)), ("en", (120, 30))])
async def test_resume_button_uses_existing_recovery_and_is_disabled_while_busy(
        recorded_overview, tmp_path, language, size):
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size) as pilot:
        app.activate_tab("results-tab")
        app.results.overview = recorded_overview
        app.results.data_status = checked_data(recorded_overview, 0)
        app.results.render_overview()
        app.results.update_report_actions()
        await pilot.pause()
        banner = app.query_one("#overview-banner", StatusBanner)
        actions = banner.query_one(ActionButtons)
        resume = list(actions.query(Button))[0]
        assert actions.actions[0].id == "overview-resume"
        assert not resume.disabled
        assert resume.region.width > 0
        with patch.object(app.catalog, "resume_experiment") as recover, patch.object(app, "launch") as launch:
            assert await pilot.click(resume)
            await pilot.pause()
            recover.assert_called_once_with(recorded_overview.record)
            launch.assert_not_called()

            token = object()
            app.read_jobs.add(token)
            try:
                app.set_busy(True)
                await pilot.pause()
                assert resume.disabled
                await pilot.click(resume)
                await pilot.pause()
                recover.assert_called_once_with(recorded_overview.record)
                launch.assert_not_called()
            finally:
                app.read_jobs.discard(token)
                app.set_busy(False)
            await pilot.pause()
            assert not resume.disabled


async def test_cancelled_data_read_leaves_identity_visible_and_windows_unverified(
        recorded_overview, tmp_path):
    from threading import Event

    app = AcprofTui(settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language="en")
    async with app.run_test(size=(80, 24)) as pilot:
        app.activate_tab("results-tab")
        app.results.source_path = recorded_overview.source
        app.results.overview = recorded_overview
        token = app.results.data_request = Event()
        app.read_jobs.add(token)
        app.results.render_overview()
        await pilot.pause()
        panel = app.query_one(ExperimentOverviewPanel)
        assert "Checking windows" in str(app.query_one("#result-summary", Static).content)
        original_level = panel.view.summary.level

        app.results.cancel_result_reads()
        await pilot.pause()
        assert token.is_set()
        assert panel.view is not None
        assert "Windows unverified" in str(app.query_one("#result-summary", Static).content)
        assert "Checking windows" not in str(app.query_one("#result-summary", Static).content)
        assert fact(panel.view, "results.view.model").value == recorded_overview.record.model_id
        assert fact(panel.view, "results.view.windows").value.msgid == "results.view.unverified"
        assert panel.query_one("#overview-facts").display
        assert panel.view.callout is None
        assert panel.view.summary.level == original_level
        assert not app.results.data_error

        # A reader already finishing when cancellation happens cannot restore stale results.
        app.results.experiment_read_finished(
            recorded_overview.source, token, "data", checked_data(recorded_overview, 2), "")
        await pilot.pause()
        assert "Windows unverified" in str(app.query_one("#result-summary", Static).content)
        assert app.results.data_status is None

        # An explicit new read restores pending status, without running a fixture reader.
        with patch.object(app, "run_worker"):
            app.results.refresh_data_status()
        await pilot.pause()
        assert "Checking windows" in str(app.query_one("#result-summary", Static).content)
        pending = app.results.data_request
        original = app.execution.session.snapshot
        app.execution.session.snapshot = replace(original, measurement_active=True)
        try:
            with patch.object(panel, "update_view") as update:
                app.results.cancel_result_reads()
            update.assert_not_called()
        finally:
            app.execution.session.snapshot = original
            app.read_jobs.discard(pending)
