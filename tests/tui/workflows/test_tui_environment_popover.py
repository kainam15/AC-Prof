"""Environment overlay interactions use cached checks, never host probes."""
import asyncio
from dataclasses import replace

import pytest
from rich.color import Color
from textual import events
from textual.widgets import Button, Select, Static, TabbedContent

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.run.environment_checks import ENVIRONMENT_CHECK_IDS
from acprof.tui.run.progress import ProgressSnapshot
from acprof.tui.ui.dialogs import ConfirmActionScreen
from acprof.tui.ui.environment import EnvironmentIndicator, display_checks


@pytest.fixture
def app(tmp_path, monkeypatch):
    from unittest.mock import Mock

    monkeypatch.delenv("NO_COLOR", raising=False)
    probe = Mock(return_value=[PreflightCheck("Docker", "warn", "fixture diagnostic")])
    monkeypatch.setattr("acprof.tui.run.preflight_controller.quick_preflight", probe)

    def machine_checks(**kwargs):
        rows = list(probe.return_value)
        known = {row.check_id for row in rows}
        return rows + [PreflightCheck(name, "ok", "fixture environment read", check_id=name)
                       for name in ENVIRONMENT_CHECK_IDS if name not in known]

    monkeypatch.setattr("acprof.tui.run.preflight_controller.scan_environment", machine_checks)
    instance = AcprofTui(RunConfig.smoke(""), settings_path=tmp_path / "settings.json")
    return instance, probe


async def settle(app, pilot):
    await pilot.pause()
    await asyncio.wait_for(app.workers.wait_for_complete(), 10)
    await pilot.pause()


def counts(app):
    return [str(entry.content) for entry in app.query(EnvironmentIndicator) if entry.display]


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_category_pins_are_independent_with_one_visible_detail(app, language):
    app, probe = app
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        panel = app.preflight.panel
        pinned_label = "已固定" if language == "zh" else "Pinned"
        hover_label = "点击入口可固定" if language == "zh" else "Click entry to pin"
        for group in ("error", "warning", "success"):
            await pilot.hover("#environment-" + group)
            assert hover_label in str(panel.query_one("#preflight-title", Static).content)
            assert not panel.pinned, "another category must not inherit a previous pin"
            await pilot.click("#environment-" + group)
            assert panel.pinned
            assert pinned_label in str(panel.query_one("#preflight-title", Static).content)
            assert len(app.query("EnvironmentPopover")) == 1
            await pilot.hover("#quit-app")
            await pilot.pause(0.25)
            assert panel.display
        for group in ("error", "warning", "success"):
            await pilot.hover("#environment-" + group)
            assert panel.pinned, "each category must remember its own pin"
            assert panel.group == group
        await pilot.click("#environment-warning")
        assert not panel.display
        await pilot.hover("#environment-warning")
        assert not panel.pinned
        await pilot.hover("#quit-app")
        await pilot.pause(0.25)
        assert not panel.display
        await pilot.hover("#environment-error")
        assert panel.pinned
        await pilot.press("escape")
        await pilot.hover("#environment-success")
        assert panel.pinned, "Escape must only clear the displayed category"
        await pilot.click("HeaderTitle")
        await pilot.hover("#environment-success")
        assert not panel.pinned, "outside click must clear the displayed category"
        for group in ("error", "warning"):
            app.query_one("#environment-" + group).focus()
            await pilot.press("enter")
            assert panel.group == group
            assert panel.pinned
        app.query_one("#environment-error").focus()
        await pilot.press("space")
        assert not panel.display, "keyboard unpin must close the selected category"
        await pilot.hover("#environment-warning")
        assert panel.pinned, "keyboard unpin must preserve other categories"
        probe.assert_called_once()


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_environment_counts_filter_cached_checks_on_hover(app, language, monkeypatch):
    from unittest.mock import Mock

    app, probe = app
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    scan = Mock(return_value=[
        PreflightCheck("Docker", "ok", "daemon ready", check_id="docker"),
        PreflightCheck("uv", "ok", "uv ready", check_id="uv"),
        PreflightCheck("RAPL", "warn", "partial RAPL access", check_id="cpu_energy"),
        PreflightCheck("perf", "fail", "permission denied", check_id="cpu_instructions"),
        PreflightCheck("GPU power", "warn", "NVML error", "error", check_id="gpu_power"),
        PreflightCheck("Torch", "pending", "self-check budget exhausted", check_id="torch"),
    ])
    monkeypatch.setattr("acprof.tui.run.preflight_controller.scan_environment", scan)
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        assert counts(app) == ["⊘ 2", "⚠ 1", "✔ 2", "◌ 1"]
        panel = app.preflight.panel
        assert not panel.display
        tabs = app.query_one("#main-tabs").region
        for group, identifiers, label in (
            ("error", {"cpu_instructions", "gpu_power"}, "异常" if language == "zh" else "Error"),
            ("warning", {"cpu_energy"}, "警告" if language == "zh" else "Warning"),
            ("success", {"docker", "uv"}, "通过" if language == "zh" else "Passed"),
            ("pending", {"torch"}, "待验证" if language == "zh" else "Unverified"),
        ):
            entry = app.query_one("#environment-" + group)
            assert await pilot.hover(entry)
            await pilot.pause()
            assert panel.display
            assert {signal.row.check_id for signal in panel.query(".environment-signal")} == identifiers
            assert label in str(panel.query_one("#preflight-title", Static).content)
            assert app.environment.label.split(" / ")[0] in str(panel.query_one("#preflight-summary", Static).content)
            assert app.query_one("#main-tabs").region == tabs
            assert await pilot.hover(entry, offset=(2, 0)), "the count is part of the hover target"
            assert {signal.row.check_id for signal in panel.query(".environment-signal")} == identifiers

        # A selected category can still be pinned and inspected with the keyboard.
        await pilot.click(entry)
        await pilot.hover("#quit-app")
        await pilot.pause(0.25)
        assert panel.display
        assert panel.pinned
        app.query_one("#preflight-signal-torch").focus()
        await pilot.pause()
        assert "self-check budget exhausted" in str(panel.query_one("#preflight-explanation", Static).content)
        await pilot.press("escape")
        assert app.focused is entry
        await pilot.press("enter")
        await pilot.pause()
        assert {signal.row.check_id for signal in panel.query(".environment-signal")} == {"torch"}
        probe.assert_called_once()
        scan.assert_called_once()


async def test_environment_counts_keep_pending_errors_and_refresh_visible(app):
    app, probe = app
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        app.preflight.environment_checks = ()
        app.preflight.check_completed = False
        app.preflight.refresh_preflight_state()
        assert counts(app) == ["⊘ 0", "⚠ 0", "✔ 0", "◌ 24"]
        app.preflight.check_completed = True
        app.preflight.environment_error = "fixture scanner failed"
        app.preflight.refresh_preflight_state()
        assert counts(app) == ["⊘ 1", "⚠ 0", "✔ 0"]
        await pilot.hover("#environment-error")
        await pilot.pause()
        assert app.query("#preflight-signal-check_error")
        app.preflight.environment_error = ""
        app.preflight.environment_checks = (PreflightCheck("Docker", "ok", "daemon ready", check_id="docker"),)
        app.preflight.refresh_preflight_state()
        await pilot.pause()
        assert counts(app) == ["⊘ 0", "⚠ 0", "✔ 1"]
        assert not app.query(".environment-signal"), "the hovered error group must discard stale checks"
        assert "没有对应检查项" in str(app.query_one("#preflight-explanation", Static).content)
        app.preflight.check_running = True
        app.preflight.refresh_preflight_state()
        assert counts(app) == ["⊘ 0", "⚠ 0", "✔ 1", "◌ …"], "rechecking cached results must remain visible"
        await pilot.pause()
        await pilot.hover("#environment-pending")
        assert "正在检查全部机器能力" in str(app.query_one("#preflight-explanation", Static).content)
        app.preflight.environment_checks = (PreflightCheck("Torch", "pending", "unverified", check_id="torch"),)
        app.preflight.refresh_preflight_state()
        assert counts(app) == ["⊘ 0", "⚠ 0", "✔ 0", "◌ …"]
        app.preflight.check_running = False
        probe.assert_called_once()


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_environment_entry_ignores_header_whitespace_after_resize(app, language):
    app, probe = app
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        entry = app.query_one("#current-environment")
        panel = app.preflight.panel
        for width, height in ((80, 24), (120, 30), (150, 45)):
            await pilot.resize_terminal(width, height)
            await pilot.pause()
            after_label = entry.region.right + 2
            between_entries = app.query_one("#environment-error").region.right + 1
            for x in (between_entries, after_label, width - 1):
                blank = (x, entry.region.y)
                await pilot.hover(offset=blank)
                assert not panel.display, "header whitespace must not open the environment panel"
                await pilot.click(offset=blank)
                assert not panel.display, "header whitespace must not pin the environment panel"

            await pilot.hover("#environment-error")
            assert panel.display
            await pilot.hover(offset=blank)
            await pilot.pause(0.25)
            assert not panel.display, "leaving the entry for header whitespace must close the panel"

            await pilot.click("#environment-error")
            await pilot.hover(offset=blank)
            await pilot.pause(0.25)
            assert panel.display, "clicking the entry must still pin the panel"
            await pilot.click(offset=blank)
            assert not panel.display, "clicking header whitespace must dismiss the pinned panel"
        probe.assert_called_once()


async def test_hover_pin_and_dismiss_keep_page_layout_and_cached_probe(app):
    app, probe = app
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        tabs = app.query_one("#main-tabs", TabbedContent)
        before = tabs.region
        assert app.query("#environment-popover"), "the header needs an interactive cached overlay"
        panel = app.query_one("#environment-popover")
        assert not panel.display
        await pilot.hover("#environment-warning")
        await pilot.pause()
        assert panel.display
        assert [button.id for button in panel.query(Button)] == ["preflight-retry"]
        assert tabs.region == before
        assert tabs.active == "run-tab"
        await pilot.hover("#preflight-retry")
        await pilot.pause(0.25)
        assert panel.display, "crossing into the panel must cancel delayed closure"
        await pilot.hover("#preflight-signal-0")
        await pilot.pause()
        assert "fixture diagnostic" in str(app.query_one("#preflight-explanation", Static).content)
        await pilot.click("#environment-warning")
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
        entry = app.query_one("#environment-warning", Static)
        assert entry.can_focus, "Tab must reach each environment entry"
        app.query_one("#quit-app").focus()
        for group in ("error", "warning", "success"):
            await pilot.press("tab")
            assert app.focused is app.query_one("#environment-" + group)
        await pilot.press("shift+tab")
        assert app.focused is entry
        for width, height in ((80, 24), (120, 30), (150, 45)):
            await pilot.resize_terminal(width, height)
            entry.focus()
            await pilot.pause()
            region = app.query_one("#current-environment").region
            strip = app.screen._compositor.render_strips()[region.y].crop(region.x, region.right)
            assert all(not segment.style or not segment.style.underline for segment in strip)
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
            retry = app.query_one("#preflight-retry", Button)
            retry.focus()
            await pilot.pause()
            region = retry.content_region
            visible = "\n".join(strip.crop(region.x, region.right).text
                                for strip in app.screen._compositor.render_strips()[region.y:region.bottom])
            assert ("重新检查" if language == "zh" else "Check again") in visible
            assert panel.region.y == 2
            assert app.get_widget_at(*retry.region.center)[0] is retry
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
        await pilot.hover("#environment-warning")
        # Pilot.hover waits for CPU idle before and after each move, which can
        # exceed the close delay. Queue rapid movement without those pauses.
        for selector in ("#quit-app", "#preflight-retry"):
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
        await pilot.hover("#environment-warning")
        app.push_screen(ConfirmActionScreen("fixture", "fixture"))
        await pilot.pause()
        assert not panel.display
        await pilot.press("escape")
        await pilot.hover("#quit-app")
        app.query_one("#quit-app").focus()
        await pilot.pause()
        await pilot.press("tab")
        assert app.focused is app.query_one("#environment-error")
        await pilot.press("enter")
        assert panel.display
        await pilot.click("#environment-error")
        assert not panel.display
        probe.assert_called_once()


async def test_environment_overlay_remains_reachable_above_maximized_log(app):
    app, probe = app
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        await pilot.press("f8")
        assert app.screen.maximized is not None
        await pilot.hover("#environment-warning")
        await pilot.pause()
        panel = app.preflight.panel
        assert panel.display
        button = app.query_one("#preflight-retry", Button)
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
async def test_projection_always_displays_all_machine_capabilities(app, runtime, mode, gpu):
    app, probe = app
    app.configuration.initial_config = replace(RunConfig.smoke(""), runtime=runtime, profiling_mode=mode, gpus=gpu)
    probe.return_value = [PreflightCheck("Runtime", "ok", runtime, runtime_kind=runtime, check_id="runtime")]
    if runtime == "process":
        probe.return_value.append(PreflightCheck("process_cpu", "ok", "psutil", "available", check_id="process_cpu"))
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        rows = {row.check_id: row for row in display_checks(app.preflight)}
        for name in ("cpu_energy", "gpu_power", "gpu_memory", "process_cpu", "container_cpu", "ncu", "nsys", "torch", "massif"):
            assert rows[name].status == "ok"
        assert "docker" in rows
        assert "latency" not in rows
        assert "throughput" not in rows
        probe.assert_called_once()


async def test_details_and_start_reason_are_redacted_and_scrollable(app):
    app, probe = app
    app.configuration.initial_config = replace(RunConfig.smoke(""), profiling_mode="full")
    probe.return_value = [PreflightCheck(
        "perf instructions", "fail", 'token=fixture-secret\nhttps://user:password@example.org/check?key=hidden\n' +
        "very long diagnostic " * 250 + "\nEND OF CHECK OUTPUT", "permission_denied", check_id="cpu_instructions", source="perf_stat",
    )]
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        assert "fixture-secret" not in str(app.query_one("#preflight-run-reason", Static).content)
        await pilot.click("#environment-error")
        await pilot.pause()
        row = app.query_one("#preflight-signal-cpu_instructions")
        row.scroll_visible(animate=False, top=True)
        await pilot.pause()
        await pilot.hover(row)
        await pilot.pause()
        inspection = app.query_one("#preflight-inspection")
        assert inspection.max_scroll_y > 0
        details = str(app.query_one("#preflight-explanation", Static).content)
        for secret in ("fixture-secret", "user:password", "key=hidden"):
            assert secret not in details
        assert "[REDACTED]" in details
        assert "perf_stat" in details
        assert "permission_denied" in details
        assert "\nEND OF CHECK OUTPUT" in details
        inspection.focus()
        await pilot.press("end")
        assert inspection.scroll_y > 0
        probe.assert_called_once()


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_status_dots_explain_cached_checks_on_hover_and_keyboard_focus(app, language):
    app, probe = app
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    app.configuration.initial_config = replace(RunConfig.smoke(""), runtime="docker", profiling_mode="full")
    probe.return_value = [
        PreflightCheck("Docker", "ok", "daemon ready", check_id="docker"),
        PreflightCheck("RAPL", "warn", "RAPL fixture limitation", "unsupported", check_id="cpu_energy"),
        PreflightCheck("perf", "fail", "Permission denied; token=fixture-secret", "permission_denied", check_id="cpu_instructions"),
        PreflightCheck("check exception", "error", "fixture check error", check_id="check_error"),
    ]
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        await pilot.click("#environment-warning")
        await pilot.pause()
        for identifier, color in (("docker", "success"), ("cpu_energy", "warning"),
                                  ("cpu_instructions", "error"), ("check_error", "error"), ("gpu_memory", "success")):
            await pilot.hover("#environment-" + color)
            await pilot.pause()
            signal = app.query_one(f"#preflight-signal-{identifier}")
            signal.scroll_visible(animate=False, top=True)
            await pilot.pause()
            assert await pilot.hover(signal)
            region = signal.content_region
            cell = app.screen._compositor.render_strips()[region.y].crop(region.x, region.x + 1)
            assert cell.text == "●"
            assert next(iter(cell)).style.color.get_truecolor() == Color.parse(app.theme_variables[color]).get_truecolor()
            if language == "en" and identifier in {"docker", "cpu_instructions", "gpu_memory"}:
                visible = app.screen._compositor.render_strips()[region.y].crop(region.x, region.right).text
                if identifier == "cpu_instructions":
                    assert "Unavailable" in visible
                else:
                    assert " · " not in visible
            explanation = str(app.query_one("#preflight-explanation", Static).content)
            problem = identifier in {"cpu_energy", "cpu_instructions", "check_error"}
            assert (("原因：" if language == "zh" else "Reason:") in explanation) == problem
            assert (("解决方法：" if language == "zh" else "Solution:") in explanation) == problem
        for identifier in ("capture_interface", "gpu_device"):
            signal = app.query_one(f"#preflight-signal-{identifier}")
            signal.focus()
            await pilot.pause()
            explanation = str(app.query_one("#preflight-explanation", Static).content)
            assert ("原因：" if language == "zh" else "Reason:") not in explanation
            assert ("解决方法：" if language == "zh" else "Solution:") not in explanation
        device = app.query_one("#preflight-signal-gpu_device")
        assert "●" in device.render().plain
        assert " · " not in device.render().plain
        await pilot.hover("#environment-error")
        await pilot.pause()
        failure = app.query_one("#preflight-signal-cpu_instructions")
        failure.scroll_visible(animate=False, top=True)
        await pilot.pause()
        await pilot.hover(failure)
        explanation = str(app.query_one("#preflight-explanation", Static).content)
        assert ("原因：" if language == "zh" else "Reason:") in explanation
        assert ("解决方法：" if language == "zh" else "Solution:") in explanation
        assert "Permission denied" in explanation
        assert "fixture-secret" not in explanation
        assert ("采集权限" if language == "zh" else "Permissions") in explanation
        app.query_one("#preflight-signal-check_error").focus()
        await pilot.press("shift+tab")
        assert app.focused is failure
        assert "Permission denied" in str(app.query_one("#preflight-explanation", Static).content)
        app.execution.session.snapshot = ProgressSnapshot(measurement_active=True)
        app.set_busy(True)
        await pilot.hover(failure)
        probe.assert_called_once()
        app.execution.session.snapshot = ProgressSnapshot()
        app.set_busy(False)
        previous_detail = str(app.query_one("#preflight-explanation", Static).content)
        app.query_one("#gpus", Select).value = "on"
        await pilot.pause()
        assert app.preflight.display_state == "limited"
        assert str(app.query_one("#preflight-explanation", Static).content) == previous_detail
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
        assert app.preflight.display_state == "ready"
        assert app.query_one("#start-run", Button).disabled
        await pilot.click("#environment-warning")
        assert "Runtime" not in str(app.query_one("#preflight-summary", Static).content)
        assert str(app.query_one("#preflight-summary", Static).content).splitlines()[1].startswith("最近检查：")
        app.execution.session.snapshot = ProgressSnapshot(measurement_active=True)
        app.set_busy(True)
        await pilot.press("f6")
        await pilot.click("#preflight-retry")
        assert app.query_one("#preflight-retry", Button).disabled
        assert "测量中" in str(app.query_one("#preflight-action-reason", Static).content)
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
        await pilot.click("#environment-warning")
        await pilot.pause()
        panel = app.preflight.panel
        assert panel.region.bottom <= 24
        assert "警告" in str(app.query_one("#preflight-title", Static).content)
        assert not any("未启用" in signal.render().plain for signal in app.query(".environment-signal"))
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
