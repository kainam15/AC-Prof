"""Environment overlay interactions use cached checks, never host probes."""
import asyncio
from dataclasses import replace

import pytest
from textual import events
from textual.widgets import Button, Collapsible, Select, Static, TabbedContent

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.run.progress import ProgressSnapshot
from acprof.tui.ui.dialogs import ConfirmActionScreen
from acprof.tui.ui.environment import display_checks


@pytest.fixture
def app(tmp_path, monkeypatch):
    from unittest.mock import Mock

    probe = Mock(return_value=[PreflightCheck("Docker", "warn", "fixture diagnostic")])
    monkeypatch.setattr("acprof.tui.run.preflight_controller.quick_preflight", probe)
    instance = AcprofTui(RunConfig.smoke(""), settings_path=tmp_path / "settings.json")
    return instance, probe


async def settle(app, pilot):
    await pilot.pause()
    await asyncio.wait_for(app.workers.wait_for_complete(), 10)
    await pilot.pause()


async def test_hover_pin_expand_and_dismiss_keep_page_layout_and_cached_probe(app):
    app, probe = app
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        tabs = app.query_one("#main-tabs", TabbedContent)
        before = tabs.region
        assert app.query("#environment-popover"), "the header needs an interactive cached overlay"
        panel = app.query_one("#environment-popover")
        assert not panel.display
        await pilot.hover("#current-environment")
        await pilot.pause()
        assert panel.display
        assert [button.id for button in panel.query(Button)] == ["preflight-expand", "preflight-retry"]
        assert tabs.region == before
        assert tabs.active == "run-tab"
        await pilot.hover("#preflight-expand")
        await pilot.pause(0.25)
        assert panel.display, "crossing into the panel must cancel delayed closure"
        await pilot.click("#preflight-expand")
        await pilot.pause()
        assert app.query_one("#preflight-details").display
        await pilot.click("#current-environment")
        await pilot.hover("#quit-app")
        await pilot.pause(0.25)
        assert panel.display, "click pins the panel"
        await pilot.press("escape")
        assert not panel.display
        probe.assert_called_once()


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_keyboard_overlay_and_left_aligned_title_survive_resize(app, language):
    app, probe = app
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        entry = app.query_one("#current-environment", Static)
        assert entry.can_focus, "Tab must reach the environment entry"
        for width, height in ((80, 24), (120, 30), (150, 45)):
            await pilot.resize_terminal(width, height)
            entry.focus()
            await pilot.press("enter")
            await pilot.pause()
            panel = app.query_one("#environment-popover")
            quit_button = app.query_one("#quit-app", Button)
            assert panel.display
            assert panel.region.y == entry.region.bottom == 2
            assert panel.region.width <= 70
            assert panel.region.bottom <= height
            assert app.get_widget_at(*quit_button.region.center)[0] is quit_button
            for button in panel.query(Button):
                assert panel.region.contains_region(button.region)
                assert app.get_widget_at(*button.region.center)[0] is button
            assert app.query_one("HeaderTitle").styles.content_align_horizontal == "left"
            await pilot.click("#preflight-expand")
            await pilot.pause()
            expand = app.query_one("#preflight-expand", Button)
            region = expand.content_region
            visible = "\n".join(strip.crop(region.x, region.right).text
                                for strip in app.screen._compositor.render_strips()[region.y:region.bottom])
            assert ("收起详情" if language == "zh" else "Collapse") in visible
            assert panel.region.y == 2
            assert app.get_widget_at(*expand.region.center)[0] is expand
            await pilot.click(expand)
            await pilot.press("escape")
            assert app.focused is entry
            await pilot.press("space")
            assert panel.display
            await pilot.click("HeaderTitle")
            assert not panel.display
        probe.assert_called_once()


async def test_hover_reentry_and_modal_hide_transient_panel(app):
    app, probe = app
    async with app.run_test(size=(120, 30)) as pilot:
        await settle(app, pilot)
        panel = app.preflight.panel
        await pilot.hover("#current-environment")
        # Pilot.hover waits for CPU idle before and after each move, which can
        # exceed the close delay. Queue rapid movement without those pauses.
        for selector in ("#quit-app", "#preflight-expand"):
            x, y = app.query_one(selector).region.center
            app.post_message(events.MouseMove(
                app.screen, x, y, delta_x=0, delta_y=0, button=0,
                shift=False, meta=False, ctrl=False, screen_x=x, screen_y=y,
            ))
        await pilot.pause(0.25)
        assert panel.display
        await pilot.hover("#quit-app")
        await pilot.pause(0.25)
        assert not panel.display
        await pilot.hover("#current-environment")
        app.push_screen(ConfirmActionScreen("fixture", "fixture"))
        await pilot.pause()
        assert not panel.display
        await pilot.press("escape")
        await pilot.hover("#quit-app")
        app.query_one("#quit-app").focus()
        await pilot.pause()
        await pilot.press("tab")
        assert app.focused is app.query_one("#current-environment")
        await pilot.press("enter")
        assert panel.display
        await pilot.click("#current-environment")
        assert not panel.display
        probe.assert_called_once()


async def test_environment_overlay_remains_reachable_above_maximized_log(app):
    app, probe = app
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        await pilot.press("f8")
        assert app.screen.maximized is not None
        await pilot.hover("#current-environment")
        await pilot.pause()
        panel = app.preflight.panel
        assert panel.display
        button = app.query_one("#preflight-expand", Button)
        assert app.get_widget_at(*button.region.center)[0] is button
        await pilot.press("escape")
        assert not panel.display
        assert app.screen.maximized is not None
        probe.assert_called_once()


@pytest.mark.parametrize(("runtime", "mode", "gpu"), [
    ("docker", "basic", "off"), ("docker", "basic", "on"),
    ("docker", "full", "off"), ("docker", "full", "on"),
    ("process", "basic", "off"), ("process", "basic", "on"),
])
async def test_projection_separates_probes_declarations_and_disabled_metrics(app, runtime, mode, gpu):
    app, probe = app
    app.configuration.initial_config = replace(RunConfig.smoke(""), runtime=runtime, profiling_mode=mode, gpus=gpu)
    probe.return_value = [PreflightCheck("Runtime", "ok", runtime, runtime_kind=runtime, check_id="runtime")]
    if runtime == "process":
        probe.return_value.append(PreflightCheck("process_cpu", "ok", "psutil", "available", check_id="process_cpu"))
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        rows = {row.check_id: row for row in display_checks(app.preflight)}
        assert rows["cpu_energy"].status == ("not_requested" if mode == "basic" else "declared")
        assert rows["gpu_power"].status == ("not_requested" if mode == "basic" or gpu == "off" else "declared")
        assert rows["gpu_memory"].status == ("not_requested" if gpu == "off" else "declared")
        assert rows["profiler"].status == "not_requested"
        assert rows["latency"].status == "declared"
        if runtime == "process":
            assert "docker" not in rows
            assert "container_cpu" not in rows
            assert rows["process_cpu"].status == "ok"
        probe.assert_called_once()


async def test_details_and_start_reason_are_redacted_and_scrollable(app):
    app, probe = app
    app.configuration.initial_config = replace(RunConfig.smoke(""), profiling_mode="full")
    probe.return_value = [PreflightCheck(
        "perf instructions", "fail", 'token=fixture-secret\nhttps://user:password@example.org/check?key=hidden\n' +
        "very long diagnostic " * 500, "permission_denied", check_id="cpu_instructions", source="perf_stat",
    )]
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        assert "fixture-secret" not in str(app.query_one("#preflight-run-reason", Static).content)
        await pilot.click("#current-environment")
        await pilot.click("#preflight-expand")
        await pilot.pause()
        row = app.query_one("#preflight-item-cpu_instructions", Collapsible)
        row.scroll_visible(animate=False, top=True)
        await pilot.pause()
        await pilot.click(row.query_one("CollapsibleTitle"))
        await pilot.pause()
        assert not row.collapsed
        assert app.query_one("#preflight-scroll").max_scroll_y > 0
        details = "\n".join(str(widget.content) for widget in row.query(Static))
        for secret in ("fixture-secret", "user:password", "key=hidden"):
            assert secret not in details
        assert "[REDACTED]" in details
        assert "perf_stat" in details
        probe.assert_called_once()


async def test_stale_auto_runtime_and_measurement_actions(app):
    app, probe = app
    probe.return_value = [PreflightCheck("Runtime", "ok", "Process", runtime_kind="process", check_id="runtime")]
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        assert app.preflight.selected_runtime == "process"
        app.query_one("#runtime", Select).value = "docker"
        await pilot.pause()
        assert app.preflight.selected_runtime == ""
        assert app.preflight.display_state == "stale"
        assert app.query_one("#start-run", Button).disabled
        await pilot.click("#current-environment")
        assert "Docker" in str(app.query_one("#preflight-summary", Static).content)
        assert "Process" not in str(app.query_one("#preflight-summary", Static).content)
        app.execution.session.snapshot = ProgressSnapshot(measurement_active=True)
        app.set_busy(True)
        await pilot.press("f6")
        await pilot.click("#preflight-retry")
        assert app.query_one("#preflight-retry", Button).disabled
        assert "测量中" in str(app.query_one("#preflight-action-reason", Static).content)
        await pilot.click("#preflight-expand")
        await pilot.pause()
        for button in app.preflight.panel.query(Button):
            assert button.region.bottom < app.preflight.panel.region.bottom
        probe.assert_called_once()


@pytest.mark.parametrize(("theme", "no_color"), [("acprof-graphite", False), ("acprof-light", False), ("acprof-light", True)])
async def test_palette_and_long_panel_fit(app, theme, no_color, monkeypatch):
    app, probe = app
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    app = AcprofTui(app.configuration.initial_config, settings_path=app.configuration.settings_path)
    app.configuration.preferences = replace(app.configuration.preferences, theme=theme)
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        await pilot.click("#current-environment")
        await pilot.click("#preflight-expand")
        await pilot.pause()
        panel = app.preflight.panel
        assert panel.region.bottom <= 24
        assert "受限" in str(app.query_one("#preflight-title", Static).content)
        assert "未启用" in str(app.query_one("#preflight-capabilities", Static).content)
        assert all(panel.region.contains_region(button.region) for button in panel.query(Button))
        assert app.theme == theme
        colors = {
            color.get_truecolor()
            for strip in app.screen._compositor.render_strips()
            for segment in strip if segment.style
            for color in (segment.style.color, segment.style.bgcolor) if color
        }
        assert colors
        assert all(len(set(rgb)) == 1 for rgb in colors) == no_color
        probe.assert_called_once()
