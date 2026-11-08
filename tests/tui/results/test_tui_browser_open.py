"""Exercise the visible report actions without launching a real browser."""
import asyncio
import subprocess
from dataclasses import replace
from threading import Event
from unittest.mock import Mock

import pytest
from textual.widgets import Button, Input, Static

from acprof import platform as host_platform
from acprof.tui.results import browser
from tests.support.tui_fixtures import AcprofTui


@pytest.fixture
def wsl_bridge(monkeypatch):
    for key in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(host_platform.platform, "system", lambda: "Linux")
    monkeypatch.setattr(host_platform, "is_wsl", lambda **_: True)
    monkeypatch.setattr(browser.shutil, "which", lambda name: "/tools/wslview" if name == "wslview" else None)
    native = Mock(side_effect=AssertionError("WSL must not use the generic Linux browser"))
    monkeypatch.setattr(browser.webbrowser, "open", native)
    return native


@pytest.mark.parametrize(("language", "size"), [("zh", (80, 24)), ("en", (120, 30))])
async def test_open_and_copy_are_visible_with_pending_and_requested_status(
    tmp_path, monkeypatch, wsl_bridge, language, size,
):
    report = tmp_path / ("long-directory-" * 6) / "中文 report.html"
    report.parent.mkdir()
    report.write_text("offline report")
    started, finish = Event(), Event()

    def launch(*args, **kwargs):
        started.set()
        if not finish.wait(5):
            raise AssertionError("test did not release the browser worker")
        return subprocess.CompletedProcess(args[0], 0, "", "")

    run = Mock(side_effect=launch)
    monkeypatch.setattr(browser, "run_command", run)
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    clipboard = Mock()
    monkeypatch.setattr(app, "copy_to_clipboard", clipboard)
    try:
        async with app.run_test(size=size) as pilot:
            await app.workers.wait_for_complete()
            app.results.show_result_section("statistics-view")
            app.query_one("#html-report-path", Input).value = str(report)
            await pilot.pause()
            location = app.query_one("#html-report-location", Static)
            assert str(location.content) == str(report)
            assert location.region.height > 1
            for selector in ("#html-report-copy", "#html-report-open"):
                button = app.query_one(selector, Button)
                button.scroll_visible(animate=False, immediate=True)
                await pilot.pause()
                assert not button.disabled and button in app.screen.focus_chain
                assert button.region.width > 0
                assert app.get_widget_at(button.region.x + 1, button.region.y + 1)[0] is button, "\n".join(
                    str((node.id, str(node.region), str(node.virtual_size), node.scroll_y, node.max_scroll_y))
                    for node in (button, *button.ancestors) if hasattr(node, "virtual_size")
                )
                assert await pilot.click(selector)
                if selector == "#html-report-copy":
                    clipboard.assert_called_once_with(str(report))
            async with asyncio.timeout(5):
                while not started.is_set():
                    await pilot.pause()
            status = str(app.query_one("#html-report-status", Static).content)
            assert ("正在请求" if language == "zh" else "Requesting browser") in status
            assert app.query_one("#html-report-open", Button).disabled
            app.results.html.open()  # A duplicate request must be rejected while busy.
            assert run.call_count == 1
            finish.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            status = str(app.query_one("#html-report-status", Static).content)
            assert ("已提交打开请求" if language == "zh" else "Open request submitted") in status
            assert ("无法确认浏览器" if language == "zh" else "Page visibility cannot be confirmed") in status
            assert "wslview" in status and str(report) in status
            assert not app.is_busy()
            assert not app.query_one("#html-report-copy", Button).disabled
            assert report.read_text() == "offline report"
            assert app.query_one("#html-report-path", Input).value == str(report)
            await pilot.resize_terminal(80, 24)
            copy = app.query_one("#html-report-copy", Button)
            copy.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert await pilot.click("#html-report-copy")
            assert clipboard.call_count == 2
    finally:
        finish.set()
    wsl_bridge.assert_not_called()


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_failed_wsl_open_preserves_file_and_enables_copy(tmp_path, monkeypatch, wsl_bridge, language):
    report = tmp_path / "report.html"
    report.write_bytes(b"offline")
    monkeypatch.setattr(browser, "run_command", Mock(return_value=subprocess.CompletedProcess([], 7, "", "interop disabled")))
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(100, 32)) as pilot:
        await app.workers.wait_for_complete()
        app.results.show_result_section("statistics-view")
        app.query_one("#html-report-path", Input).value = str(report)
        await pilot.pause()
        button = app.query_one("#html-report-open", Button)
        button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click("#html-report-open")
        await app.workers.wait_for_complete()
        await pilot.pause()
        status = str(app.query_one("#html-report-status", Static).content)
        assert "interop disabled" in status and "powershell.exe: not found" in status
        assert ("报告仍保留" if language == "zh" else "Report preserved") in status
        assert "已提交打开请求" not in status and "Open request submitted" not in status
        assert not app.is_busy()
        assert not app.query_one("#html-report-copy", Button).disabled
        assert report.read_bytes() == b"offline"
    wsl_bridge.assert_not_called()
