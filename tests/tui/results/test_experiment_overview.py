"""Selection loads metadata and read-only status without computing statistics."""
import asyncio
import json
from threading import Event
from unittest.mock import patch

import pytest
from textual.widgets import Button, Input, Static

from tests.support.tui_fixtures import AcprofTui
from tests.tui.results.test_profile_overview_snapshot import profile_store


async def test_selection_preserves_directory_and_automatically_checks_profiles(tmp_path):
    root = tmp_path / "experiment"
    profile_store(root, 3)
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        with patch("acprof.tui.run.diagnostics.summarize_windows") as statistics, patch.object(app, "launch") as launch:
            app.results.select_result(str(root))
            await app.workers.wait_for_complete()
            await pilot.pause()
        statistics.assert_not_called()
        launch.assert_not_called()
        assert app.query_one("#experiment-source", Input).value == str(root)
        assert app.profiles.overview.result_dir == root
        assert not app.query_one("#open-result-profiling", Button).disabled
        assert not app.query("#summarize-results, #summarize-backfill")
        assert app.results.formal_ok_rows == 2


async def test_metadata_is_visible_while_full_result_read_is_waiting(tmp_path):
    from acprof.results.csv import read_result_csv_snapshot

    root = tmp_path / "experiment"
    profile_store(root, 3)
    meta = root / "static_meta.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "model_name": "fixture/fast-overview"}))
    entered, release = Event(), Event()

    def slow_read(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return read_result_csv_snapshot(*args, **kwargs)

    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        # Patch the public reader before the selection imports its consumer.
        with patch("acprof.results.csv.read_result_csv_snapshot", side_effect=slow_read):
            app.results.select_result(str(root))
            try:
                await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), 3)
                await pilot.pause()
                assert "fixture/fast-overview" in str(app.query_one("#result-summary", Static).content)
                assert "正在检查" in str(app.query_one("#profile-status", Static).content)
            finally:
                release.set()
            await app.workers.wait_for_complete()


def test_light_overview_uses_metadata_without_opening_csv(tmp_path, monkeypatch):
    from pathlib import Path

    from acprof.tui.results.overview import overview_text, read_experiment_overview

    root = tmp_path / "experiment"
    profile_store(root, 3)
    original = Path.open

    def metadata_only(path, *args, **kwargs):
        assert path.suffix != ".csv", "light overview must not read measured rows"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", metadata_only)
    overview = read_experiment_overview(root)
    assert overview.source == root
    assert overview.data_state == "exists"
    assert "未记录" in overview_text(overview)


def test_background_check_parses_each_layer_once_without_statistics(tmp_path, monkeypatch):
    from collections import Counter
    from pathlib import Path

    from acprof.result_store import protocol
    from acprof.results import csv as result_csv
    from acprof.tui.results.data_status import inspect_experiment_data

    root = tmp_path / "experiment"
    manifest = profile_store(root, 3)
    parses = Counter()
    original = result_csv.parse_result_stream

    def observe(stream, path, expected, **kwargs):
        parses[Path(path)] += 1
        return original(stream, path, expected, **kwargs)

    monkeypatch.setattr(result_csv, "parse_result_stream", observe)
    monkeypatch.setattr(protocol, "parse_result_stream", observe)
    with patch("acprof.analysis.uncertainty.summarize_windows") as stats:
        status = inspect_experiment_data(root)
    stats.assert_not_called()
    assert status.formal_ok_rows == 2
    assert status.profiler_overview.pending == 3
    assert parses == Counter({root / item["path"]: 1 for item in manifest["layers"].values()})


async def test_fast_switch_discards_old_check_and_keeps_measurement_excluded(tmp_path):
    from acprof.tui.results.data_status import inspect_experiment_data

    first, second = tmp_path / "first", tmp_path / "second"
    profile_store(first, 3)
    profile_store(second, 3)
    entered, release = Event(), Event()

    def delayed(source, **kwargs):
        if source == first:
            entered.set()
            assert release.wait(10)
        return inspect_experiment_data(source, **kwargs)

    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        with patch("acprof.tui.results.controller.inspect_experiment_data", side_effect=delayed):
            app.results.select_result(str(first))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                app.results.select_result(str(second))
                for _ in range(30):
                    await pilot.pause()
                    if app.profiles.overview is not None:
                        break
                assert app.profiles.overview.result_dir == second
                assert app.query_one("#start-run", Button).disabled
            finally:
                release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
        assert app.profiles.overview.result_dir == second
        assert app.query_one("#experiment-source", Input).value == str(second)
        assert not app.read_jobs


async def test_profile_confirmation_revalidates_data_before_launch(tmp_path):
    root = tmp_path / "experiment"
    profile_store(root, 3)
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(120, 30)) as pilot:
        app.results.select_result(str(root))
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.profiles.open_selection_dialog()
        await pilot.pause()
        with patch.object(app, "launch") as launch:
            assert await pilot.click("#profile-choice-run")
            await pilot.pause()
            launch.assert_not_called()
            manifest = root / "result_layers.json"
            manifest.write_text("{broken}")
            assert await pilot.click("#confirm-yes")
            await app.workers.wait_for_complete()
            await pilot.pause()
            launch.assert_not_called()
        assert app.query_one("#open-result-profiling", Button).disabled
        assert not app.read_jobs


def test_report_freshness_uses_data_and_metadata_provenance(tmp_path):
    import hashlib

    from acprof.tui.results.report_files import html_report_state

    source = tmp_path / "result.csv"
    source.write_text("status\nok\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = tmp_path / "static_meta.json"
    metadata.write_text("{}")
    meta_digest = hashlib.sha256(metadata.read_bytes()).hexdigest()
    report = tmp_path / "report.html"
    report.write_text('<script id="report-data" type="application/json">' + json.dumps({
        "sources": [{"path": str(source), "sha256": digest, "metadata_sha256": {str(metadata): meta_digest}}]
    }) + '</script>')
    assert html_report_state(report, source, digest) == "current"
    assert html_report_state(report, source, "0" * 64) == "stale"
    metadata.write_text('{"profile": "updated"}')
    assert html_report_state(report, source, digest) == "stale"
    report.write_text("<html>historical report</html>")
    assert html_report_state(report, source, digest) == "unverified"
    report.write_text('<script id="report-data">{broken}</script>')
    assert html_report_state(report, source, digest) == "damaged"


def test_unreadable_report_discovery_preserves_metadata_and_data_status(tmp_path):
    from acprof.tui.results.data_status import inspect_experiment_data
    from acprof.tui.results.overview import read_experiment_overview

    profile_store(tmp_path, 3)
    with patch("acprof.tui.results.report_files.existing_html_report", side_effect=PermissionError("report permission denied")):
        overview = read_experiment_overview(tmp_path)
        status = inspect_experiment_data(tmp_path)
    assert overview.data_state == "exists"
    assert status.formal_ok_rows == 2
    assert status.profiler_overview is not None
    assert status.html_state == "damaged"


def test_new_format_missing_or_damaged_never_uses_legacy_export(tmp_path):
    import pytest

    from acprof.results.layout import ArtifactLayout, resolve_result_file
    from acprof.tui.results.data_status import inspect_experiment_data

    layout = ArtifactLayout.for_new_run(tmp_path / "new")
    layout.initialize()
    legacy = layout.root / "result_all.csv"
    legacy.write_text("status,warmup,latency_app_s\nok,0,0.02\n")
    layout.result_csv.unlink()
    assert resolve_result_file(layout.root) == layout.result_csv
    with pytest.raises((ValueError, FileNotFoundError), match="result_layers"):
        inspect_experiment_data(layout.root)
    layout.result_csv.write_text("{broken}")
    with pytest.raises(ValueError, match="result|JSON|json"):
        inspect_experiment_data(layout.root)
    assert legacy.read_text() == "status,warmup,latency_app_s\nok,0,0.02\n"


def test_all_tui_commands_require_explicit_task_in_multi_task_directory(tmp_path):
    import sys
    from pathlib import Path

    import pytest

    from acprof.results.layout import ArtifactLayout
    from acprof.tui.commands import (
        prepare_comparison,
        prepare_html_report,
        prepare_plot,
        prepare_profile,
    )

    collection = tmp_path / "collection"
    run_id = "a" * 32
    tasks = []
    for name in ("first", "second"):
        task_id = name + "-" + "b" * 12
        ArtifactLayout.for_new_run(collection / "tasks" / task_id / run_id).initialize()
        tasks.append({"task_id": task_id, "status": "complete"})
    manifests = collection / "metadata" / "task_runs"
    manifests.mkdir(parents=True)
    (manifests / f"{run_id}.json").write_text(json.dumps({"schema_version": 1, "run_id": run_id, "tasks": tasks}))
    common = dict(project_dir=tmp_path, python_executable=Path(sys.executable))
    operations = [
        lambda: prepare_plot(str(collection), **common),
        lambda: prepare_html_report(str(collection), **common),
        lambda: prepare_profile(str(collection), tools="ncu", dry_run=True, **common),
        lambda: prepare_comparison(str(collection), str(collection), baseline="left", purpose="same-hardware", **common),
    ]
    for operation in operations:
        with pytest.raises(ValueError, match="specify one task run directory"):
            operation()


async def test_profile_completion_failure_and_cancel_refresh_only_selected_experiment(tmp_path):
    root = tmp_path / "experiment"
    profile_store(root, 3)
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(120, 30)) as pilot:
        app.results.select_result(str(root))
        await app.workers.wait_for_complete()
        await pilot.pause()
        for returncode, error, cancelled in ((0, "", False), (1, "fixture failure", False), (130, "", True)):
            app.execution.session.kind = "profile"
            app.execution.session.stop_requested = cancelled
            with patch.object(app, "launch") as launch:
                app.process_finished("profile", returncode, None, error)
                await app.workers.wait_for_complete()
                await pilot.pause()
                launch.assert_not_called()
            assert app.profiles.overview.result_dir == root
            assert app.query_one("#experiment-source", Input).value == str(root)


async def test_no_automatic_reads_or_backfill_inside_measurement(tmp_path):
    from acprof.tui.run.progress import ProgressSnapshot

    root = tmp_path / "experiment"
    profile_store(root, 3)
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        app.execution.session.snapshot = ProgressSnapshot(measurement_active=True)
        with patch("acprof.tui.results.controller.inspect_experiment_data") as inspect, patch.object(app, "launch") as launch:
            app.results.select_result(str(root))
            app.profiles.request_profile_run(str(root), "ncu")
            await pilot.pause()
            inspect.assert_not_called()
            launch.assert_not_called()
        assert app.query_one("#open-result-profiling", Button).disabled
        assert not app.read_jobs


async def test_backfill_validation_can_be_cancelled_without_launching(tmp_path):
    from acprof.results.csv import read_result_csv_snapshot

    root = tmp_path / "experiment"
    profile_store(root, 3)
    entered, release = Event(), Event()

    def slow_read(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return read_result_csv_snapshot(*args, **kwargs)

    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        app.results.select_result(str(root))
        await app.workers.wait_for_complete()
        app.activate_tab("backfill-tab")
        await pilot.pause()
        with patch("acprof.results.csv.read_result_csv_snapshot", side_effect=slow_read), patch.object(app, "launch") as launch:
            app.profiles.launch_profile(dry_run=True, tools="ncu")
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                await pilot.pause()
                assert await pilot.click("#backfill-cancel")
            finally:
                release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            launch.assert_not_called()
        assert not app.is_busy()


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_restored_legacy_path_loads_overview_and_narrow_controls_remain_usable(tmp_path, language):
    import os
    from pathlib import Path

    from acprof.tui.settings.preferences import TuiSettings, UiPreferences, save_settings

    root = tmp_path / ("模型-" + "long-experiment-path-" * 5)
    profile_store(root, 3)
    settings = tmp_path / "settings.json"
    app = AcprofTui(settings_path=settings)
    save_settings(settings, TuiSettings(ui=UiPreferences(language=language), last_result_csv=str(root / "result_layers.json")), app.project_dir)
    app = AcprofTui(settings_path=settings)
    async with app.run_test(size=(80, 24)) as pilot:
        await app.workers.wait_for_complete()
        app.activate_tab("results-tab")
        await pilot.pause()
        assert app.query_one("#experiment-source", Input).value == str(root)
        assert app.configuration.saved_settings.last_result_csv == str(root)
        assert app.profiles.overview is not None
        assert not app.query_one("#report-calculate", Button).disabled, (app.operation_state(), app.results.source_display, app.read_jobs)
        for selector in ("#experiment-source", "#choose-experiment-source", "#copy-experiment-path",
                         "#refresh-experiment-state", "#report-calculate"):
            widget = app.query_one(selector)
            region = widget.region
            assert 0 <= region.x < region.right <= 80
            assert 0 <= region.y < region.bottom <= 24
            assert app.get_widget_at(region.x + 1, region.y + 1)[0] is widget
        with patch.object(app, "copy_to_clipboard") as copy:
            assert await pilot.click("#copy-experiment-path")
            copy.assert_called_once_with(str(root))
        app.clear_notifications()
        app.logs.flush()
        await pilot.pause()
        if output := os.environ.get("ACPROF_SNAPSHOT_OUTPUT_DIR"):
            Path(output).mkdir(parents=True, exist_ok=True)
            (Path(output) / f"experiment-overview-{language}-80x24.svg").write_text(app.export_screenshot())
        assert await pilot.click("#refresh-experiment-state")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.results.formal_ok_rows == 2


async def test_refresh_marks_report_stale_and_failed_generation_keeps_overview(tmp_path):
    from acprof.tui.results.data_status import inspect_experiment_data

    root = tmp_path / "experiment"
    profile_store(root, 3)
    status = inspect_experiment_data(root)
    report = root / "report.html"
    report.write_text('<script id="report-data">' + json.dumps({"sources": [{
        "path": str(status.path), "sha256": status.sha256, "metadata_sha256": dict(status.metadata_identity),
    }]}) + '</script>')
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(120, 30)) as pilot:
        app.results.select_result(str(root))
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not app.query_one("#html-report-open", Button).disabled
        metadata = root / "static_meta.json"
        metadata.write_text(json.dumps({**json.loads(metadata.read_text()), "model_name": "updated/model"}))
        app.results.refresh_experiment()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "已过期" in str(app.query_one("#result-html-status", Static).content)
        assert app.query_one("#html-report-open", Button).disabled
        before = str(app.query_one("#result-summary", Static).content)
        app.results.html.finished(error="fixture report failure", cancelled=False)
        assert str(app.query_one("#result-summary", Static).content) == before
        assert "fixture report failure" in str(app.query_one("#html-report-status", Static).content)
        app.results.html.launch()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not app.query_one("#html-report-open", Button).disabled
        assert "与当前数据一致" in str(app.query_one("#result-html-status", Static).content)
        assert report.read_text().startswith('<script id="report-data">')
