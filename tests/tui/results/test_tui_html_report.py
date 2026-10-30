"""HTML reporting uses the public CLI and the existing TUI process owner."""
import asyncio
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, Collapsible, Input, Static, TabbedContent
from tests.support.tui_fixtures import AcprofTui

from acprof.artifact_layout import ArtifactLayout
from acprof.experiments.configuration import RunConfigError
from acprof.tui import commands
from acprof.tui.process_controller import ProcessController
from acprof.tui.settings import TuiSettings, UiPreferences, save_settings


@pytest.fixture
def csv_result(tmp_path):
    source = tmp_path / "experiment with spaces" / "result_all.csv"
    source.parent.mkdir()
    source.write_text(
        "cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_p95_s\n"
        "2,4,off,32,0,0,ok,0.04\n", encoding="utf-8")
    return source


def prepare(source, root):
    return commands.prepare_html_report(source, project_dir=root, python_executable=Path("/python"))


def test_command_normalizes_manifest_and_multiple_inputs_without_writing(tmp_path, csv_result):
    root = tmp_path / "layered"
    layout = ArtifactLayout.for_new_run(root)
    layout.initialize()
    manifest = root / "result_layers.json"
    manifest.write_text("{}")
    before = sorted(root.rglob("*"))
    pending = prepare(f'"{manifest}";"{csv_result}"', tmp_path)
    assert pending.kind == "html-report"
    assert pending.command[:5] == ("/python", "-u", "-m", "acprof", "report")
    assert pending.command[5:7] == (str(root), str(csv_result))
    assert Path(pending.report_path).parent == root / "plots" / "analysis"
    assert pending.command[-2:] == ("--output", pending.report_path)
    assert sorted(root.rglob("*")) == before
    assert prepare(str(root), tmp_path).report_path != pending.report_path


def test_relative_csv_and_existing_output_get_new_versions(tmp_path, csv_result):
    pending = prepare(str(csv_result.relative_to(tmp_path)), tmp_path)
    output = Path(pending.report_path)
    assert output.parent == csv_result.parent / "analysis"
    output.parent.mkdir()
    output.write_text("previous report")
    assert prepare(str(csv_result), tmp_path).report_path != str(output)
    assert output.read_text() == "previous report"


@pytest.mark.parametrize("source", ["", "missing.csv", "missing", '"unterminated'])
def test_missing_or_malformed_inputs_are_rejected(tmp_path, source):
    with pytest.raises((RunConfigError, ValueError)):
        prepare(source, tmp_path)


async def test_generate_button_is_reachable_and_measurement_blocks_it(tmp_path, csv_result):
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        app.results.show_result_section("exports-view")
        app.query_one("#result-csv", Input).value = str(csv_result)
        await pilot.pause()
        button = app.query_one("#html-report-generate", Button)
        button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        with patch.object(ProcessController, "execute_command") as execute:
            assert await pilot.click(button)
            await pilot.pause()
            assert execute.call_count == 1
            assert app.process_kind == "html-report"
            assert button.disabled
            app.results.launch_html_report()
            assert execute.call_count == 1
            app.process_finished("html-report", 1, None, "fixture failure")
            assert "fixture failure" in str(app.query_one("#html-report-status", Static).content)
            assert not app.is_busy()
            app.latest_snapshot = replace(app.latest_snapshot, measurement_active=True)
            app.results.launch_html_report()
            assert execute.call_count == 1


async def test_real_cli_success_persists_path_and_preserves_input(tmp_path, csv_result):
    settings = tmp_path / "settings.json"
    app = AcprofTui(settings_path=settings)
    original = csv_result.read_bytes()
    async with app.run_test(size=(120, 30)) as pilot:
        app.results.show_result_section("exports-view")
        app.query_one("#result-csv", Input).value = str(csv_result)
        await pilot.pause()
        app.results.launch_html_report()
        await app.workers.wait_for_complete()
        await pilot.pause()
        output = Path(app.query_one("#html-report-path", Input).value)
        html = output.read_text()
        assert "Plotly" in html
        assert '<script src=' not in html
        for view in ("Comparison Matrix", "Pareto", "Scaling"):
            assert view in html
        assert '"configs":' in html
        assert csv_result.read_bytes() == original
        assert app.configuration.saved_settings.last_html_report == str(output)
        assert app.query_one("#main-tabs", TabbedContent).active == "results-tab"
        assert not app.is_busy()
        assert "quality" in str(app.query_one("#html-report-status", Static).content)
    restored = AcprofTui(settings_path=settings)
    async with restored.run_test():
        assert restored.query_one("#html-report-path", Input).value == str(output)


async def test_real_layer_manifest_and_csv_comparison_keep_three_views(tmp_path, csv_result):
    from acprof.result_layers import publish_result_layers
    layered = tmp_path / "layered"
    ArtifactLayout.for_new_run(layered).initialize()
    publish_result_layers(csv_result, output_dir=layered)
    second = tmp_path / "second.csv"
    second.write_text(csv_result.read_text().replace("2,4,off", "4,4,off"))
    original = {path: path.read_bytes() for path in layered.rglob("*") if path.is_file()}
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test() as pilot:
        app.query_one("#result-csv", Input).value = str(layered / "result_layers.json")
        await pilot.pause()
        app.query_one("#html-extra-sources", Input).value = str(second)
        app.results.launch_html_report()
        await app.workers.wait_for_complete()
        await pilot.pause()
        output = Path(app.query_one("#html-report-path", Input).value)
        assert output.parent == layered / "plots" / "analysis"
        html = output.read_text()
        data = json.loads(html.split('<script id="report-data" type="application/json">', 1)[1].split('</script>', 1)[0])
        assert len(data["sources"]) == 2
        assert len(data["configs"]) == 2
        for view in ("matrix", "pareto", "scaling"):
            assert f'data-view="{view}"' in html
        assert "__DATA__" not in html
        assert "__VIEW_MODEL__" not in html
        assert '<script src=' not in html
        for path, content in original.items():
            assert path.read_bytes() == content


async def test_no_valid_windows_retains_evidence_and_displays_limit(tmp_path, csv_result):
    csv_result.write_text(csv_result.read_text().replace(",ok,", ",failed,"))
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test() as pilot:
        app.query_one("#result-csv", Input).value = str(csv_result)
        await pilot.pause()
        app.results.launch_html_report()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert Path(app.query_one("#html-report-path", Input).value).is_file()
        assert "没有有效正式窗口" in str(app.query_one("#html-report-status", Static).content)


@pytest.mark.parametrize(("language", "failed", "exported"), [
    ("zh", "失败", "HTML 已导出"), ("en", "failed", "HTML exported"),
])
@pytest.mark.parametrize("failure_kind", ["existing", "copy", "directory_sync"])
async def test_export_button_and_failure_keep_generated_artifact(
    tmp_path, monkeypatch, language, failed, exported, failure_kind,
):
    from acprof import artifacts
    from acprof.tui import report_files

    source = tmp_path / "report.html"
    source.write_bytes(b"<html>complete offline report</html>")
    destination = tmp_path / "export.html"
    settings = tmp_path / "settings.json"
    save_settings(settings, TuiSettings(ui=UiPreferences(language=language)), tmp_path)
    app = AcprofTui(settings_path=settings)
    async with app.run_test(size=(80, 24)) as pilot:
        app.results.show_result_section("exports-view")
        app.query_one("#html-report-panel", Collapsible).collapsed = False
        app.query_one("#html-export-options", Collapsible).collapsed = False
        app.query_one("#html-report-path", Input).value = str(source)
        app.query_one("#html-export-path", Input).value = str(destination)
        await pilot.pause()
        # Expanding the nested panels schedules scrolling after refresh.
        await pilot.wait_for_scheduled_animations()
        button = app.query_one("#html-report-export", Button)
        button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(button)
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert destination.read_bytes() == source.read_bytes()
        assert exported in str(app.query_one("#html-report-status", Static).content)

        def fail_copy(_reader, writer, **_kwargs):
            writer.write(b"partial")
            raise OSError("injected copy failure")

        def fail_sync(_directory):
            raise OSError("injected directory sync failure")

        if failure_kind != "existing":
            destination = tmp_path / "retry.html"
            app.query_one("#html-export-path", Input).value = str(destination)
            if failure_kind == "copy":
                monkeypatch.setattr(report_files.shutil, "copyfileobj", fail_copy)
            else:
                monkeypatch.setattr(artifacts, "sync_directory", fail_sync)
        button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(button)
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert failed in str(app.query_one("#html-report-status", Static).content)
        if failure_kind == "copy":
            assert not destination.exists()
        else:
            assert destination.read_bytes() == source.read_bytes()
        assert source.read_bytes() == b"<html>complete offline report</html>"
        assert not list(tmp_path.glob(".retry.html.*.tmp"))
        assert app.query_one("#html-report-path", Input).value == str(source)
        assert not app.is_busy()
        assert not button.disabled


def test_measurement_lock_blocks_html_export(tmp_path):
    from acprof.host.run_state import MeasurementLock
    from acprof.tui.report_files import export_html_file
    source, target = tmp_path / "report.html", tmp_path / "export.html"
    source.write_text("report")
    with MeasurementLock(), pytest.raises(RuntimeError):
        export_html_file(source, target)
    assert not target.exists()


def test_cli_receipt_reports_excluded_windows_and_quality(tmp_path, csv_result, capsys):
    from acprof.cli.report import main
    with csv_result.open("a") as stream:
        stream.write("2,4,off,32,1,0,failed,\n")
    output = tmp_path / "report.html"
    assert main([str(csv_result), "--output", str(output)]) == 0
    lines = capsys.readouterr().out.splitlines()
    receipts = [json.loads(line.removeprefix("ACPROF_REPORT ")) for line in lines if line.startswith("ACPROF_REPORT ")]
    assert len(receipts) == 1
    assert receipts[0]["valid_windows"] == 1
    assert receipts[0]["excluded_rows"] == 1
    assert receipts[0]["quality_status"] == ["unknown"]


@pytest.mark.parametrize(("size", "language"), [((80, 24), "en"), ((120, 30), "zh"), ((150, 45), "en")])
async def test_html_actions_fit_visible_report_page(tmp_path, size, language):
    settings = tmp_path / "settings.json"
    save_settings(settings, TuiSettings(ui=UiPreferences(language=language)), tmp_path)
    app = AcprofTui(settings_path=settings)
    async with app.run_test(size=size) as pilot:
        app.results.show_result_section("exports-view")
        app.query_one("#html-report-panel", Collapsible).collapsed = False
        await pilot.pause()
        assert not app.query_one("#html-report-open", Button).display
        assert not app.query_one("#html-report-cancel", Button).display
        app.query_one("#html-report-path", Input).value = str(tmp_path / "existing.html")
        await pilot.pause()
        for name in ("generate", "open"):
            button = app.query_one(f"#html-report-{name}", Button)
            button.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert button.region.width >= button.get_content_width(button.size, button.size)
            assert button.region.x >= 0
            assert button.region.right <= size[0]
            assert button.region.y >= 0
            assert button.region.bottom <= size[1]
        with patch.object(ProcessController, "execute_command") as execute:
            app.results.launch_html_report()
            execute.assert_not_called()
        status = str(app.query_one("#html-report-status", Static).content)
        assert ("Select an experiment" if language == "en" else "请先选择实验") in status


@pytest.mark.parametrize(("data", "error"), [("", "missing or duplicate CSV columns"),
    ("cpu_cores,status,warmup\n", "no measurement rows"),
    ('cpu_cores,status,warmup\n"broken\n', "unexpected end of data")])
async def test_real_cli_failure_shows_error_and_keeps_old_report(tmp_path, csv_result, data, error):
    csv_result.write_text(data)
    old = csv_result.parent / "report.html"
    old.write_text("old report")
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test() as pilot:
        app.query_one("#result-csv", Input).value = str(csv_result)
        await pilot.pause()
        assert app.query_one("#html-report-path", Input).value == str(old)
        app.results.launch_html_report()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert error in str(app.query_one("#html-report-status", Static).content)
        assert not app.is_busy()
        assert old.read_text() == "old report"
        assert app.query_one("#html-report-path", Input).value == str(old)
        assert not app.configuration.saved_settings.last_html_report


async def test_cancel_reaps_process_and_keeps_previous_html(tmp_path, csv_result):
    old = csv_result.parent / "report.html"
    old.write_text("old")
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test() as pilot:
        app.query_one("#result-csv", Input).value = str(csv_result)
        await pilot.pause()
        command = (sys.executable, "-u", "-c", "import time; print('ready', flush=True); time.sleep(30)")
        pending = commands.PendingLaunch(command, "html-report", report_path=str(tmp_path / "new.html"))
        app.launch(pending)
        async with asyncio.timeout(5):
            while app.query_one("#stop-run", Button).disabled:
                await pilot.pause()
        app.results.show_result_section("exports-view")
        cancel = app.query_one("#html-report-cancel", Button)
        cancel.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(cancel)
        await pilot.pause()
        # Escape cancels the stop dialog without terminating the task.
        await pilot.press("escape")
        await pilot.pause()
        assert app.is_busy()
        assert not app.stop_requested
        assert await pilot.click(cancel)
        await pilot.pause()
        assert await pilot.click("#confirm-yes")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not app.is_busy()
        assert app.lifecycle.process is None
        assert "已取消" in str(app.query_one("#html-report-status", Static).content)
        assert old.read_text() == "old"
        assert app.query_one("#html-report-path", Input).value == str(old)


@pytest.mark.parametrize("environment", [{}, {"DISPLAY": ":0", "SSH_CONNECTION": "remote"}, {"WAYLAND_DISPLAY": "wayland-0", "SSH_TTY": "/dev/pts/1"}])
async def test_remote_or_headless_never_launches_browser(tmp_path, environment):
    report = tmp_path / "report.html"
    report.write_text("report")
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test():
        app.query_one("#html-report-path", Input).value = str(report)
        with patch.dict(os.environ, environment, clear=True), patch("acprof.platform.platform.system", return_value="Linux"), patch("webbrowser.open") as browser:
            app.results.open_html_report()
            browser.assert_not_called()
        status = str(app.query_one("#html-report-status", Static).content)
        assert str(report) in status
        assert "SFTP" in status


@pytest.mark.parametrize("outcome", [True, False, OSError("missing browser")])
async def test_local_browser_action_does_not_change_report(tmp_path, outcome):
    report = tmp_path / "report with spaces.html"
    report.write_text("report")
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test() as pilot:
        app.query_one("#html-report-path", Input).value = str(report)
        with patch("acprof.platform.can_open_local_browser", return_value=True), patch("webbrowser.open", side_effect=outcome if isinstance(outcome, Exception) else None, return_value=outcome) as browser:
            app.results.open_html_report()
            await app.workers.wait_for_complete()
            await pilot.pause()
            browser.assert_called_once_with(report.as_uri(), new=2)
        assert report.read_text() == "report"
        assert not app.is_busy()
        if outcome is not True:
            assert "失败" in str(app.query_one("#html-report-status", Static).content)


def test_export_is_identical_and_refuses_existing_files_and_links(tmp_path):
    from acprof.tui.report_files import export_html_file
    source, target = tmp_path / "source.html", tmp_path / "target.html"
    source.write_bytes(b"<html>offline report</html>")
    export_html_file(source, target)
    assert target.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        export_html_file(source, target)
    with pytest.raises(FileExistsError):
        export_html_file(source, source)
    link = tmp_path / "link.html"
    link.symlink_to(tmp_path / "absent.html")
    with pytest.raises(FileExistsError):
        export_html_file(source, link)
    assert not (tmp_path / "absent.html").exists()
