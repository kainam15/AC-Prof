"""Reviewed version-management states at narrow and standard terminal sizes."""
from dataclasses import replace
from pathlib import Path

import pytest
from packaging.version import Version
from textual.containers import VerticalScroll
from textual.widgets import Button, Static, TabbedContent

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from acprof.tui.i18n import message_id
from acprof.updates.installation import Installation
from acprof.updates.releases import select_releases
from acprof.updates.worker import UpdatePlan

pytestmark = pytest.mark.visual


@pytest.mark.parametrize(("language", "size", "scene"), [
    ("zh", (80, 24), "versions"), ("en", (120, 30), "versions"),
    ("zh", (80, 24), "downgrade"),
])
def test_version_manager_snapshot(snap_compare, svg_operations, tmp_path, monkeypatch, language, size, scene):
    app = AcprofTui(RunConfig.smoke(""), settings_path=tmp_path / "tui.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    controller = app.updates
    controller.current = Version("1.1.0rc1" if scene == "downgrade" else "1.0.0")
    controller.channel = "stable" if scene == "downgrade" else "preview"
    controller.effective_channel = "preview" if scene == "downgrade" else "stable"
    controller.installation = Installation("uv tool", Path("/home/user/.local/share/uv/tools/acprof"),
                                           Path("/home/user/.local/share/uv/tools/acprof/bin/python"),
                                           Path("/home/user/.local/bin/acprof"), Path("/home/user/.local/bin/uv"))
    controller.release = select_releases({"releases": {
        "1.0.0": [{"yanked": False}], "1.1.0rc1": [{"yanked": False}],
    }}, str(controller.current), controller.channel)
    controller.status = "available"
    controller.checked_at = "2026-11-16T12:00:00+08:00"
    monkeypatch.setattr("acprof.tui.updates.service.make_plan", lambda install, release, directory: UpdatePlan(
        str(install.prefix), str(install.entrypoint), str(install.manager), "/usr/bin/python3",
        str(release.current), str(release.latest), release.channel, str(directory), str(directory / "fixture")))

    async def prepare(pilot):
        # The visible header counts include the startup preflight result.
        await svg_operations.wait_for_preflight()
        await pilot.pause()
        app.title = f"AC-Prof {controller.current}"
        app.activate_tab("settings-tab")
        app.localization.set_text(app.query_one("#settings-location", Static),
                                  message_id("settings.load.file_path", "/home/user/.config/acprof/tui.json"))
        controller.render()
        await pilot.pause()
        assert app.query_one("#main-tabs", TabbedContent).active == "settings-tab"
        app.clear_notifications()
        if scene == "downgrade":
            controller.request_install()
            await pilot.pause()
            app.screen.query_one("#confirm-no", Button).focus()
            await pilot.pause()
            return
        app.set_focus(None)
        status = app.query_one("#version-status", Static)
        status.scroll_visible(animate=False, top=True, immediate=True)
        await pilot.pause()
        body = app.query_one("#settings-body", VerticalScroll)
        assert body.scrollable_content_region.contains_region(status.region)
        assert status.region.y == body.scrollable_content_region.y or body.scroll_y == body.max_scroll_y

    assert snap_compare(app, terminal_size=size, run_before=prepare)
