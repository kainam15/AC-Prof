"""Source configuration and frontend recovery use existing dialogs."""
import os
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest
from textual.widgets import Button, Input, Select, Static

from acprof.experiments.configuration import RunConfig
from acprof.host.env_utils import load_project_env
from acprof.host.images.errors import frontend_error
from acprof.tui.i18n import translate
from acprof.tui.run.downloads import download_summary
from acprof.tui.run.preparation import PreparationScreen
from acprof.tui.settings.environment import EnvironmentSettingsScreen, field_id
from tests.support.tui_fixtures import AcprofTui


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_source_controls_save_restore_and_allow_official_choice(tmp_path, language):
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    destination = os.environ.get("ACPROF_SOURCE_EVIDENCE")
    with patch.dict(os.environ, {}, clear=True):
        async with app.run_test(size=(80, 24)) as pilot:
            screen = EnvironmentSettingsScreen(tmp_path, tmp_path / "backups")
            await app.push_screen(screen)
            await pilot.pause()
            preset = screen.query_one("#use-domestic-sources", Button)
            preset.focus()
            preset.scroll_visible(immediate=True)
            await pilot.pause()
            assert await pilot.click(preset)
            await pilot.pause()
            pypi = screen.query_one("#" + field_id("ACPROF_PYPI_ARTIFACT_MIRROR"), Select)
            assert pypi.value == "tencent"
            assert not (tmp_path / ".env.local").exists()
            assert screen.query_one("#" + field_id("ACPROF_RUNTIME_IMAGE_SOURCE"), Select).value == "auto"
            mirror = screen.query_one("#" + field_id("ACPROF_DOCKER_HUB_MIRROR"), Input)
            mirror.focus()
            await pilot.pause()
            await pilot.press(*"mirror.example/docker.io")
            save = screen.query_one("#save-environment-settings", Button)
            assert screen.region.contains_region(save.region)
            assert await pilot.click(save)
            await pilot.pause()
            assert screen.saved
            assert os.environ["ACPROF_DOCKER_HUB_MIRROR"] == "mirror.example/docker.io"
            assert os.environ["ACPROF_PYPI_ARTIFACT_MIRROR"] == "tencent"
            loaded = {}
            load_project_env(tmp_path, environ=loaded)
            assert loaded["ACPROF_DEBIAN_ARTIFACT_MIRROR"] == "tencent"
            await pilot.click("#close-environment-settings")
            restored = EnvironmentSettingsScreen(tmp_path, tmp_path / "backups")
            await app.push_screen(restored)
            await pilot.pause()
            official = restored.query_one("#" + field_id("ACPROF_PYPI_ARTIFACT_MIRROR"), Select)
            assert official.value == "tencent"
            official.focus()
            await pilot.pause()
            await pilot.press("enter", "home", "enter")
            await pilot.pause()
            assert official.value == ""
            assert await pilot.click("#save-environment-settings")
            await pilot.pause()
            assert os.environ["ACPROF_PYPI_ARTIFACT_MIRROR"] == ""
            if destination:
                app.save_screenshot(filename=f"sources-{language}-80x24.svg", path=destination)
            await pilot.resize_terminal(120, 30)
            await pilot.pause()
            assert restored.region.contains_region(restored.query_one("#save-environment-settings").region)
            if destination:
                app.save_screenshot(filename=f"sources-{language}.svg", path=destination)


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_frontend_failure_offers_preflight_retry_without_rebuild(tmp_path, language):
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    error = frontend_error("i/o timeout", "mirror.example/docker/dockerfile:1@sha256:" + "a" * 64).to_dict()
    replies = []
    async with app.run_test(size=(80, 24)) as pilot:
        screen = PreparationScreen({"stage": "environment", "request": {
            "kind": "error", "id": 1, "docker_error": error, "detail": error["detail"],
        }}, respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        detail = str(screen.query_one("#preparation-detail", Static).content)
        assert "mirror.example" in detail
        assert ("尚未开始安装依赖" if language == "zh" else "before dependency installation") in detail
        assert "ACPROF_DOCKER_HUB_MIRROR" not in detail
        assert not list(screen.query("#preparation-rebuild"))
        if destination := os.environ.get("ACPROF_SOURCE_EVIDENCE"):
            app.save_screenshot(filename=f"frontend-failure-{language}-80x24.svg", path=destination)
        button = screen.query_one("#preparation-continue", Button)
        assert str(button.label) == ("重新预检并重试" if language == "zh" else "Recheck and retry")
        button.focus()
        await pilot.pause()
        assert await pilot.click(button)
        assert replies == [{"action": "retry"}]


def test_source_summary_distinguishes_policy_route_and_unknown_frontend_cache():
    report = {
        "runtime": {"source_mode": "build", "platform_local": True, "environment_local": False,
                    "will_attempt_ghcr_pull": False, "download_policy": {
                        "docker_hub_mirror": "mirror.example/docker.io", "runtime_registry": "ghcr.io/example/runtime",
                        "artifact_mirrors": {"ACPROF_PYPI_ARTIFACT_MIRROR": "tencent"},
                    }},
        "sources": [{"category": "oci", "stage": "buildkit-frontend", "url": "https://mirror.example/docker/dockerfile",
                     "route": "externally-managed", "cache_status": "unknown"}],
    }
    original = deepcopy(report)
    english = download_summary(report, lambda value: translate(value, "en"))
    for text in ("tencent", "Official source", "mirror.example", "Cache reuse confirmed during build",
                 "Docker manages image transfer", "platform Found", "environment Not found"):
        assert text in english
    assert report == original


def test_custom_prebuilt_registry_summary_does_not_claim_ghcr_access():
    report = {"runtime": {"source_mode": "pull", "will_attempt_ghcr_pull": True,
                          "download_policy": {"runtime_registry": "mirror.example/runtime"}}}
    english = download_summary(report, lambda value: translate(value, "en"))
    assert "Will attempt mirror.example" in english
    assert "GHCR" not in english
