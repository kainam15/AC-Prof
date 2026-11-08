"""Windows startup exercises real workers without Unix APIs or host probes."""

import asyncio
import os
import sys
from dataclasses import replace

import httpx
import pytest
from textual.widgets import Button, Input, Static, TabbedContent

from acprof import __version__
from acprof.capabilities import Capability
from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from acprof.tui.settings.environment import EnvironmentSettingsScreen


@pytest.mark.parametrize("language", ["zh", "en"])
@pytest.mark.parametrize("http_status", [200, 503])
async def test_windows_startup_and_permission_checks_without_unix_apis(
    tmp_path, monkeypatch, language, http_status,
):
    monkeypatch.setattr("acprof.platform.platform.system", lambda: "Windows")
    monkeypatch.delattr(os, "getuid", raising=False)
    monkeypatch.setitem(sys.modules, "fcntl", None)
    monkeypatch.setattr("acprof.tui.app.PROJECT_DIR", tmp_path)
    # Keep the real platform policy and diagnostics; isolate external probes.
    monkeypatch.setattr("acprof.tui.run.diagnostics.shutil.which", lambda _name: None)
    monkeypatch.setattr("acprof.tui.run.diagnostics.probe_cpu_energy", lambda: Capability(
        "unsupported", "Windows fixture", "fixture",
    ))
    monkeypatch.setattr("acprof.tui.run.diagnostics.probe_perf_instructions", lambda **_kwargs: Capability(
        "unsupported", "Windows fixture", "fixture",
    ))
    requests = []

    async def respond(_transport, request):
        requests.append(request)
        return httpx.Response(http_status, json={"info": {"version": "9.0.0", "yanked": False}})

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    config = replace(RunConfig.smoke(""), runtime="docker")
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await asyncio.wait_for(app.workers.wait_for_complete(), 10)
        await pilot.pause()

        assert app.preflight.check_completed
        assert app.preflight.error == ""
        assert app.preflight.environment_error == ""
        platform_check = next(check for check in app.preflight.checks if check.check_id == "environment")
        assert platform_check.status == "fail"
        assert "Unsupported collection environment" in str(platform_check.detail)
        assert "Unsupported collection environment" in str(app.query_one("#preflight-run-reason", Static).content)
        assert app.query_one("#start-run", Button).disabled

        version_status = str(app.query_one("#version-status", Static).content)
        assert __version__ in version_status
        assert len(requests) == 1
        assert str(requests[0].url) == "https://pypi.org/pypi/acprof/json"
        if http_status == 200:
            assert ("最新版本：9.0.0" if language == "zh" else "Latest version: 9.0.0") in version_status
            assert "uv tool upgrade acprof" in version_status
        else:
            assert "HTTPStatusError" in version_status
            assert "503" in version_status
        await pilot.press("d", "e", "m", "o")
        assert app.query_one("#model", Input).value == "demo"

        app.activate_tab("settings-tab")
        await pilot.pause()
        connections = app.query_one("#open-environment-settings", Button)
        connections.focus()
        await pilot.pause()
        assert await pilot.click(connections)
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, EnvironmentSettingsScreen)
        screen.query_one("#environment-tabs", TabbedContent).active = "permissions-tab"
        await pilot.pause()
        check = screen.query_one("#check-environment-permissions", Button)
        check.focus()
        await pilot.pause()
        assert await pilot.click(check)
        await asyncio.wait_for(app.workers.wait_for_complete(), 10)
        await pilot.pause()
        report = str(screen.query_one("#permission-report", Static).content)
        assert "Unsupported collection environment" in report
        assert "AttributeError" not in report
        assert not screen.query_one("#check-environment-permissions", Button).disabled
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is not screen
