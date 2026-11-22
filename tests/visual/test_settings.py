"""The default settings page at a desktop terminal size."""

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Button

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from tests.support.tui_actions import open_page

pytestmark = pytest.mark.visual


def test_settings_page(snap_compare, svg_operations, tmp_path, monkeypatch):
    monkeypatch.setattr("acprof.tui.app.PROJECT_DIR", tmp_path)
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ACPROF_WECOM_WEBHOOK_URL"):
        monkeypatch.delenv(key, raising=False)
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")

    async def prepare(pilot):
        await svg_operations.wait_for_preflight()
        await open_page(pilot, "settings-tab")
        body = app.query_one("#settings-body", VerticalScroll)
        permissions = app.query_one("#open-environment-settings", Button)
        assert body.scrollable_content_region.contains_region(permissions.region)
        app.clear_notifications()
        app.set_focus(None)

    assert snap_compare(app, terminal_size=(120, 38), run_before=prepare)
