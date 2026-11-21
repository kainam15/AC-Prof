"""Environment overlay interactions use cached checks, never host probes."""
import asyncio
from dataclasses import replace

import pytest
from rich.color import Color
from textual.widgets import Button, Collapsible, Select, Static, TabbedContent

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.run.environment_checks import ENVIRONMENT_CHECK_IDS
from acprof.tui.run.progress import ProgressSnapshot
from acprof.tui.ui.dialogs import ConfirmActionScreen
from acprof.tui.ui.environment import EnvironmentIndicator, display_checks
from acprof.tui.ui.structured import Callout, FactGrid
from tests.support.tui_fixtures import expand_check_source


@pytest.fixture
def app(tmp_path, monkeypatch):
    from unittest.mock import Mock

    monkeypatch.delenv("NO_COLOR", raising=False)
    probe = Mock(return_value=[PreflightCheck("Docker", "warn", "fixture diagnostic", check_id="docker", runtime_kind="docker")])
    monkeypatch.setattr("acprof.tui.run.preflight_controller.quick_preflight", probe)

    def machine_checks(**kwargs):
        rows = list(probe.return_value)
        known = {row.check_id for row in rows}
        return rows + [PreflightCheck(name, "ok", "fixture environment read", check_id=name)
                       for name in ENVIRONMENT_CHECK_IDS if name not in known]

    monkeypatch.setattr("acprof.tui.run.preflight_controller.scan_environment", machine_checks)
    instance = AcprofTui(replace(RunConfig.smoke("fixture/model"), runtime="docker", profiling_mode="full",
                                 gpus="on", compute_profile_tool="both", execution_profile_tool="both", repeat=3),
                         settings_path=tmp_path / "settings.json")
    return instance, probe


async def settle(app, pilot):
    await pilot.pause()
    await asyncio.wait_for(app.workers.wait_for_complete(), 10)
    await pilot.pause()


def counts(app):
    return [str(entry.content) for entry in app.query(EnvironmentIndicator) if entry.display]


def fact_text(app, identifier):
    grid = app.query_one(identifier, FactGrid)
    return "\n".join(
        f"{app.tr(row.label)}: {app.tr(row.value)}\n{row.detail}"
        for row in (*grid.section.rows, *grid.section.detail_rows)
    )


def detail_text(app):
    """Read the current projection; disclosure accessibility is checked separately."""
    callout = app.query_one("#preflight-callout", Callout)
    if callout.display:
        return "\n".join(str(callout.query_one(".callout-" + kind, Static).content)
                         for kind in ("problem", "impact", "next", "raw"))
    if app.query_one("#preflight-facts", FactGrid).display:
        return fact_text(app, "#preflight-facts")
    return str(app.query_one("#preflight-empty", Static).content)


def context_text(app):
    return fact_text(app, "#preflight-context")


def visible_buttons(panel):
    return [button for button in panel.query(Button)
            if button.display and all(getattr(parent, "display", True) for parent in button.ancestors)
            and button.region.width and button.region.height]


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_category_clicks_open_switch_and_dismiss_without_hover_popups(app, language):
    app, probe = app
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        panel = app.preflight.panel
        # Hovering header indicators never pops the overlay open accidentally.
        for group in ("error", "warning", "success"):
            await pilot.hover("#environment-" + group)
            await pilot.pause()
            assert not panel.display
        # Clicking opens and pins the selected category; clicking another switches cleanly.
        for group in ("error", "warning", "success"):
            await pilot.click("#environment-" + group)
            await pilot.pause()
            assert panel.display
            assert panel.pinned
            assert panel.group == group
            assert len(app.query("EnvironmentPopover")) == 1
            await pilot.hover("#quit-app")
            await pilot.pause(0.25)
            assert panel.display
        # Clicking the already open category toggles it closed.
        await pilot.click("#environment-success")
        assert not panel.display
        # Keyboard Enter/Space toggle cleanly, Escape and outside click dismiss.
        for group in ("error", "warning"):
            app.query_one("#environment-" + group).focus()
            await pilot.press("enter")
            assert panel.display
            assert panel.group == group
        await pilot.press("escape")
        assert not panel.display
        app.query_one("#environment-error").focus()
        await pilot.press("space")
        assert panel.display
        await pilot.click("HeaderTitle")
        assert not panel.display
        probe.assert_called_once()


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_environment_counts_filter_cached_checks_on_click(app, language, monkeypatch):
    from unittest.mock import Mock

    app, probe = app
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    probe.return_value = []
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
        assert counts(app) == ["× 0", "! 3", "✓ 3", "… 2"]
        for width in (80, 120, 52, 80):
            await pilot.resize_terminal(width, 24)
            await pilot.pause()
            for indicator in app.query(EnvironmentIndicator):
                if indicator.display:
                    assert indicator.render().plain == str(indicator.content)
            warning = app.query_one("#environment-warning")
            region = warning.content_region
            strip = app.screen._compositor.render_strips()[region.y].crop(region.x, region.right)
            assert strip.text.strip() == "! 3"
            assert "perf" not in str(warning.tooltip)
        panel = app.preflight.panel
        assert not panel.display
        tabs = app.query_one("#main-tabs").region
        for group, identifiers, label in (
            ("error", set(), "错误" if language == "zh" else "Error"),
            ("warning", {"cpu_energy", "cpu_instructions", "gpu_power"}, "警告" if language == "zh" else "Warning"),
            ("success", {"docker", "uv", "configuration"}, "通过" if language == "zh" else "Passed"),
            ("pending", {"torch", "preflight"}, "待验证" if language == "zh" else "Unverified"),
        ):
            entry = app.query_one("#environment-" + group)
            assert await pilot.click(entry)
            await pilot.pause()
            assert panel.display
            assert {signal.row.check_id for signal in panel.query(".environment-signal")} == identifiers
            assert label in str(entry.tooltip)
            assert not panel.query("#preflight-title")
            assert not panel.query("#preflight-summary")
            assert app.title.endswith(f" · {app.environment.label.split(' / ')[0]}")
            assert ("平台:" if language == "zh" else "Platform:") not in context_text(app)
            assert app.query_one("#main-tabs").region == tabs

        # Clicking the active category keeps it open until explicitly dismissed.
        await pilot.hover("#quit-app")
        await pilot.pause(0.25)
        assert panel.display
        assert panel.pinned
        signal = app.query_one("#preflight-signal-torch")
        signal.focus()
        await pilot.pause()
        assert "self-check budget exhausted" not in detail_text(app)
        await pilot.press("enter")
        assert "self-check budget exhausted" in detail_text(app)
        await pilot.press("escape")
        assert app.focused is entry
        await pilot.press("enter")
        await pilot.pause()
        assert {signal.row.check_id for signal in panel.query(".environment-signal")} == {"torch", "preflight"}
        probe.assert_called_once()
        scan.assert_called_once()


async def test_environment_counts_keep_pending_errors_and_refresh_visible(app):
    app, probe = app
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        app.preflight.environment_checks = ()
        app.preflight.checks = ()
        app.preflight.check_completed = False
        app.preflight.refresh_preflight_state()
        assert counts(app) == ["× 0", "! 0", "✓ 1", "… 1"]
        app.preflight.check_completed = True
        app.preflight.environment_error = "fixture scanner failed"
        app.preflight.refresh_preflight_state()
        assert counts(app) == ["× 0", "! 1", "✓ 1", "… 1"]
        assert app.query_one('#start-run', Button).disabled
        await pilot.pause()
        assert await pilot.click("#environment-warning")
        await pilot.pause()
        assert app.query("#preflight-signal-inventory_error")
        app.preflight.environment_error = ""
        app.preflight.environment_checks = (PreflightCheck("Docker", "ok", "daemon ready", check_id="docker"),)
        app.preflight.refresh_preflight_state()
        await pilot.pause()
        assert counts(app) == ["× 0", "! 0", "✓ 2", "… 1"]
        assert app.query_one('#start-run', Button).disabled
        assert not app.query(".environment-signal"), "the warning group must discard resolved issues"
        assert "没有对应检查项" in detail_text(app)
        app.preflight.check_running = True
        app.preflight.refresh_preflight_state()
        assert counts(app) == ["× 0", "! 0", "✓ 1", "… 2"], "old success becomes checking"
        await pilot.pause()
        await pilot.click("#environment-pending")
        await pilot.pause()
        assert "点击检查项" in detail_text(app)
        app.preflight.environment_checks = (PreflightCheck("Torch", "pending", "unverified", check_id="torch"),)
        app.preflight.refresh_preflight_state()
        assert counts(app) == ["× 0", "! 0", "✓ 1", "… 2"]
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
            before_label = entry.region.x - 2
            blank = (before_label, entry.region.y)
            await pilot.hover(offset=blank)
            assert not panel.display, "header whitespace must not open the environment panel"
            await pilot.click(offset=blank)
            assert not panel.display, "header whitespace must not pin the environment panel"

            await pilot.hover("#environment-error")
            assert not panel.display, "hovering indicator must not open the environment panel"

            await pilot.click("#environment-error")
            await pilot.hover(offset=blank)
            await pilot.pause(0.25)
            assert panel.display, "clicking the entry must open the panel"
            await pilot.click(offset=blank)
            assert not panel.display, "clicking header whitespace must dismiss the panel"
        probe.assert_called_once()


async def test_click_pin_and_dismiss_keep_page_layout_and_cached_probe(app):
    app, probe = app
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        tabs = app.query_one("#main-tabs", TabbedContent)
        before = tabs.region
        assert app.query("#environment-popover"), "the header needs an interactive cached overlay"
        panel = app.query_one("#environment-popover")
        assert not panel.display
        await pilot.click("#environment-warning")
        await pilot.pause()
        assert panel.display
        assert visible_buttons(panel)[0] is panel.retry_button
        assert len(visible_buttons(panel)) == 1
        assert tabs.region == before
        assert tabs.active == "run-tab"
        assert await pilot.click("#preflight-signal-docker")
        await pilot.pause()
        assert "fixture diagnostic" in detail_text(app)
        assert len(visible_buttons(panel)) == 2
        await pilot.hover("#quit-app")
        await pilot.pause(0.25)
        assert panel.display, "opened panel stays open until explicitly dismissed"
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
        await pilot.press("shift+tab")
        assert app.focused is app.query_one("#show-shortcuts")
        await pilot.press("shift+tab")
        assert app.focused is app.query_one("#environment-success")
        assert not app.query_one("#environment-not_requested").display
        app.query_one("#environment-error").focus()
        for group in ("warning", "success"):
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
            assert panel.region.y == entry.region.bottom == 1
            assert panel.region.right <= quit_button.region.x
            assert panel.region.width <= 70
            assert panel.region.bottom <= height
            assert app.get_widget_at(*quit_button.region.center)[0] is quit_button
            for button in visible_buttons(panel):
                assert panel.region.contains_region(button.region)
                assert app.get_widget_at(*button.region.center)[0] is button
            assert app.query_one("HeaderTitle").styles.content_align_horizontal == "left"
            retry = app.preflight.panel.retry_button
            retry.focus()
            await pilot.pause()
            region = retry.content_region
            visible = "\n".join(strip.crop(region.x, region.right).text
                                for strip in app.screen._compositor.render_strips()[region.y:region.bottom])
            assert ("重新检查" if language == "zh" else "Check again") in visible
            assert panel.region.y == 1
            assert app.get_widget_at(*retry.region.center)[0] is retry
            await pilot.press("escape")
            assert app.focused is entry
            await pilot.press("space")
            assert panel.display
            await pilot.click("HeaderTitle")
            assert not panel.display
        probe.assert_called_once()


async def test_modal_and_keyboard_toggle_panel(app):
    app, probe = app
    async with app.run_test(size=(120, 30)) as pilot:
        await settle(app, pilot)
        panel = app.preflight.panel
        await pilot.hover("#environment-warning")
        await pilot.pause()
        assert not panel.display
        await pilot.click("#environment-warning")
        assert panel.display
        app.push_screen(ConfirmActionScreen("fixture", "fixture"))
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.click("#environment-warning")
        assert not panel.display
        app.query_one("#quit-app").focus()
        await pilot.pause()
        await pilot.press("shift+tab", "shift+tab", "shift+tab", "shift+tab")
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
        await pilot.click("#environment-warning")
        await pilot.pause()
        panel = app.preflight.panel
        assert panel.display
        button = app.preflight.panel.retry_button
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
    app.configuration.initial_config = replace(RunConfig.smoke("fixture/model"), runtime=runtime, profiling_mode=mode, gpus=gpu)
    probe.return_value = [PreflightCheck("Runtime", "ok", runtime, runtime_kind=runtime, check_id="runtime")]
    if runtime == "process":
        probe.return_value.append(PreflightCheck("process_cpu", "ok", "psutil", "available", check_id="process_cpu"))
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        rows = {row.check_id: row for row in display_checks(app.preflight)}
        for name in ("cpu_energy", "gpu_power", "gpu_memory", "process_cpu", "container_cpu", "ncu", "nsys", "torch", "massif"):
            assert rows[name].status == 'ok'
        assert "docker" in rows
        assert "latency" not in rows
        assert "throughput" not in rows
        probe.assert_called_once()


async def test_details_and_start_reason_are_redacted_and_scrollable(app, monkeypatch):
    app, probe = app
    app.configuration.initial_config = replace(RunConfig.smoke("fixture/model"), runtime="docker", extra_options={"dram_energy": "required"})
    probe.return_value = [PreflightCheck(
        "RAPL DRAM", "fail", 'token=fixture-secret\nhttps://user:password@example.org/check?key=hidden\n' +
        "very long diagnostic " * 250 + "\nEND OF CHECK OUTPUT", "permission_denied", check_id="dram_energy", source="rapl_topology",
    )]
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        assert app.query_one("#start-run", Button).disabled
        assert "fixture-secret" not in str(app.query_one("#preflight-run-reason", Button).label)
        await pilot.click("#environment-error")
        await pilot.pause()
        row = app.query_one("#preflight-signal-dram_energy")
        row.scroll_visible(animate=False, top=True)
        await pilot.pause()
        assert await pilot.click(row)
        await pilot.pause()
        inspection = app.query_one("#preflight-inspection")
        callout = app.query_one("#preflight-callout", Callout)
        disclosure = callout.query_one(".callout-details", Collapsible)
        copy = callout.query_one(".callout-copy", Button)
        assert disclosure.collapsed
        assert copy not in app.screen.focus_chain
        disclosure.query_one("CollapsibleTitle").focus()
        await pilot.press("enter")
        await pilot.pause()
        assert not disclosure.collapsed
        assert inspection.max_scroll_y > 0
        details = detail_text(app)
        for secret in ("fixture-secret", "user:password", "key=hidden"):
            assert secret not in details
        assert "[REDACTED]" in details
        assert "rapl_topology" in details
        assert "permission_denied" in details
        assert "\nEND OF CHECK OUTPUT" in details
        copied = []
        monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
        copy.scroll_visible(animate=False, immediate=True)
        copy.focus()
        await pilot.pause()
        assert app.get_widget_at(*copy.region.center)[0] is copy
        await pilot.press("enter")
        assert copied == [str(callout.query_one(".callout-raw", Static).content)]
        assert "fixture-secret" not in copied[0]
        inspection.focus()
        await pilot.press("end")
        assert inspection.scroll_y > 0
        probe.assert_called_once()


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_status_symbols_explain_cached_checks_only_after_selection(app, language):
    app, probe = app
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    app.configuration.initial_config = replace(RunConfig.smoke("fixture/model"), runtime="docker", profiling_mode="full", gpus="on")
    probe.return_value = [
        PreflightCheck("Docker", "ok", "daemon ready", check_id="docker"),
        PreflightCheck("RAPL", "warn", "cannot read energy_uj", "permission_denied", check_id="cpu_energy"),
        PreflightCheck("perf", "fail", "Permission denied; token=fixture-secret", "permission_denied", check_id="cpu_instructions"),
        PreflightCheck("check exception", "error", "fixture check error", check_id="check_error"),
    ]
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        await pilot.click("#environment-warning")
        await pilot.pause()
        for identifier, color in (("docker", "success"), ("cpu_energy", "warning"),
                                  ("cpu_instructions", "warning"), ("check_error", "error"), ("gpu_memory", "success")):
            if app.preflight.panel.group != color:
                await pilot.click("#environment-" + color)
                await pilot.pause()
            await expand_check_source(app, pilot, identifier)
            signal = app.query_one(f"#preflight-signal-{identifier}")
            signal.scroll_visible(animate=False, top=True)
            await pilot.pause()
            before_hover = detail_text(app)
            assert await pilot.hover(signal)
            assert detail_text(app) == before_hover
            assert await pilot.click(signal)
            region = signal.content_region
            cell = app.screen._compositor.render_strips()[region.y].crop(region.x, region.x + 1)
            assert cell.text == {"success": "✓", "warning": "!", "error": "×"}[color]
            assert next(iter(cell)).style.color.get_truecolor() == Color.parse(app.theme_variables[color]).get_truecolor()
            if language == "en" and identifier in {"docker", "cpu_instructions", "gpu_memory"}:
                visible = app.screen._compositor.render_strips()[region.y].crop(region.x, region.right).text
                if identifier == "cpu_instructions":
                    assert "Permission denied" in visible
                else:
                    assert " · " not in visible
            problem = identifier in {"cpu_energy", "cpu_instructions", "check_error"}
            callout = app.query_one("#preflight-callout", Callout)
            assert callout.display == problem
            assert app.query_one("#preflight-facts", FactGrid).display != problem
            if problem:
                for kind in ("problem", "impact", "next"):
                    assert str(callout.query_one(".callout-" + kind, Static).content)
            if identifier == "cpu_energy":
                label = "RAPL CPU 能耗 · 无权限" if language == "zh" else "RAPL CPU energy · Permission denied"
                assert signal.render().plain == "! " + label
                assert label in app.tr(callout.content.problem)
                assert ("CPU / 内存指标可能缺失" if language == "zh" else "CPU or memory metrics may be missing") in app.tr(callout.content.impact)
                assert "cpu_energy_total_j" in app.tr(callout.content.detail)
                assert "energy_uj" in app.tr(callout.content.next_step)
        for identifier in ("capture_interface", "gpu_device"):
            await expand_check_source(app, pilot, identifier)
            signal = app.query_one(f"#preflight-signal-{identifier}")
            before_focus = detail_text(app)
            signal.focus()
            await pilot.pause()
            assert detail_text(app) == before_focus
            await pilot.press("enter")
            assert app.query_one("#preflight-facts", FactGrid).display
            assert not app.query_one("#preflight-callout", Callout).display
        device = app.query_one("#preflight-signal-gpu_device")
        assert "✓" in device.render().plain
        assert " · " not in device.render().plain
        await pilot.click("#environment-warning")
        await pilot.pause()
        failure = app.query_one("#preflight-signal-cpu_instructions")
        failure.scroll_visible(animate=False, top=True)
        await pilot.pause()
        assert await pilot.click(failure)
        explanation = detail_text(app)
        callout = app.query_one("#preflight-callout", Callout)
        assert callout.display
        for kind in ("problem", "impact", "next"):
            assert str(callout.query_one(".callout-" + kind, Static).content)
        assert "Permission denied" in explanation
        assert "fixture-secret" not in explanation
        assert ("权限" if language == "zh" else "permission") in explanation.lower()
        failure.focus()
        await pilot.press("tab", "shift+tab")
        assert app.focused is failure
        assert "Permission denied" in detail_text(app)
        app.execution.session.snapshot = ProgressSnapshot(measurement_active=True)
        app.set_busy(True)
        await pilot.hover(failure)
        probe.assert_called_once()
        app.execution.session.snapshot = ProgressSnapshot()
        app.set_busy(False)
        previous_detail = detail_text(app)
        app.query_one("#gpus", Select).value = "on"
        await pilot.pause()
        assert app.preflight.display_state == "failed"
        assert detail_text(app) == previous_detail
        probe.assert_called_once()


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_selected_check_details_ignore_hover_focus_and_resize(app, language):
    app, probe = app
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    probe.return_value = [
        PreflightCheck("Environment", "ok", "selected environment evidence", check_id="environment"),
        PreflightCheck("Docker", "ok", "hovered Docker evidence", check_id="docker"),
        PreflightCheck("cgroup", "ok", "keyboard cgroup evidence", check_id="cgroup"),
    ]
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        await pilot.click("#environment-success")
        await pilot.pause()
        await expand_check_source(app, pilot, "environment")
        await expand_check_source(app, pilot, "docker")
        await expand_check_source(app, pilot, "cgroup")
        selected = app.query_one("#preflight-signal-environment")
        preview = app.query_one("#preflight-signal-docker")
        keyboard = app.query_one("#preflight-signal-cgroup")
        explanation = app.query_one("#preflight-inspection")
        for width, height in ((80, 24), (120, 35)):
            await pilot.resize_terminal(width, height)
            selected.scroll_visible(animate=False, top=True)
            await pilot.pause()
            assert await pilot.click(selected)
            assert app.focused is selected
            assert "selected environment evidence" in detail_text(app)
            preview.scroll_visible(immediate=True, force=True)
            await pilot.pause()
            assert await pilot.hover(preview)
            assert "selected environment evidence" in detail_text(app)
            assert app.focused is selected
            assert await pilot.hover(explanation)
            assert "selected environment evidence" in detail_text(app)

            explanation.focus()
            await pilot.pause()
            preview.scroll_visible(immediate=True, force=True)
            await pilot.pause()
            assert await pilot.hover(preview)
            assert "selected environment evidence" in detail_text(app)
            assert await pilot.hover(app.preflight.panel.retry_button)
            assert "selected environment evidence" in detail_text(app)

            selected.focus()
            await pilot.pause()
            await pilot.press("tab", "tab")
            assert app.focused is preview
            assert "selected environment evidence" in detail_text(app), "Tab only moves focus"
            await pilot.press("enter")
            assert "hovered Docker evidence" in detail_text(app)
            selected.scroll_visible(immediate=True, force=True)
            await pilot.pause()
            assert await pilot.hover(selected)
            assert "hovered Docker evidence" in detail_text(app)
            await pilot.press("tab", "tab")  # cgroup title precedes its check
            assert app.focused is keyboard
            assert "hovered Docker evidence" in detail_text(app)
            await pilot.press("space")
            assert "keyboard cgroup evidence" in detail_text(app)
            assert await pilot.hover(explanation)
            assert "keyboard cgroup evidence" in detail_text(app)
        probe.assert_called_once()


async def test_selected_check_clears_when_category_or_cached_checks_change(app):
    app, probe = app
    probe.return_value = [
        PreflightCheck("Environment", "ok", "selected old evidence", check_id="environment"),
        PreflightCheck("Docker", "ok", "hovered old evidence", check_id="docker"),
        PreflightCheck("RAPL", "warn", "warning evidence", check_id="cpu_energy"),
    ]
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        await pilot.click("#environment-success")
        await pilot.pause()
        await expand_check_source(app, pilot, "environment")
        await expand_check_source(app, pilot, "docker")
        assert app.preflight.panel._selected_row is None
        assert "selected old evidence" not in detail_text(app)
        app.query_one("#preflight-signal-environment").scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert await pilot.click("#preflight-signal-environment")
        app.query_one("#preflight-signal-docker").scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert await pilot.hover("#preflight-signal-docker")
        assert "selected old evidence" in detail_text(app)

        await pilot.click("#environment-warning")
        await pilot.pause()
        assert app.preflight.panel._selected_row is None
        hint = detail_text(app)
        assert await pilot.hover("#preflight-signal-cpu_energy")
        assert detail_text(app) == hint
        assert await pilot.click("#preflight-signal-cpu_energy")
        assert "warning evidence" in detail_text(app)
        assert "old evidence" not in detail_text(app)

        await pilot.click("#environment-success")
        await pilot.pause()
        assert app.preflight.panel._selected_row is None
        assert "selected old evidence" not in detail_text(app)
        await expand_check_source(app, pilot, "environment")
        await expand_check_source(app, pilot, "docker")
        app.query_one("#preflight-signal-environment").scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert await pilot.click("#preflight-signal-environment")
        assert "selected old evidence" in detail_text(app)
        app.query_one("#preflight-signal-docker").scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert await pilot.hover("#preflight-signal-docker")
        assert "selected old evidence" in detail_text(app)

        app.preflight.checks = app.preflight.environment_checks = (
            PreflightCheck("Docker", "ok", "refreshed Docker evidence", check_id="docker"),
        )
        app.preflight.refresh_preflight_state()
        await pilot.pause()
        assert not app.query("#preflight-signal-environment")
        assert app.preflight.panel._selected_row is None
        assert "old evidence" not in detail_text(app)
        app.query_one("#preflight-signal-docker").scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert await pilot.hover("#preflight-signal-docker")
        assert "refreshed Docker evidence" not in detail_text(app)
        assert await pilot.click("#preflight-signal-docker")
        assert "refreshed Docker evidence" in detail_text(app)
        probe.assert_called_once()


async def test_stale_explicit_runtime_and_measurement_actions(app):
    app, probe = app
    app.configuration.initial_config = replace(RunConfig.smoke("fixture/model"), runtime="process")
    app.preflight.AUTO_CHECK_DELAY = 60
    probe.return_value = [PreflightCheck("Runtime", "ok", "Process", runtime_kind="process", check_id="runtime")]
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        assert app.preflight.selected_runtime == "process"
        app.query_one("#runtime", Select).value = "docker"
        await pilot.pause()
        assert app.preflight.selected_runtime == ""
        assert app.preflight.display_state == "pending"
        assert app.query_one("#start-run", Button).disabled
        await pilot.click("#environment-warning")
        assert "Process" not in context_text(app), "the old selected runtime is not machine evidence"
        assert app.preflight.completed_at.strftime("%Y-%m-%d %H:%M:%S") in context_text(app)
        app.execution.session.snapshot = ProgressSnapshot(measurement_active=True)
        app.set_busy(True)
        await pilot.press("f6")
        await pilot.click(app.preflight.panel.retry_button)
        assert app.preflight.panel.retry_button.disabled
        assert "测量中" in str(app.query_one("#preflight-action-reason", Static).content)
        await pilot.pause()
        for button in visible_buttons(app.preflight.panel):
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
        assert not panel.query("#preflight-title")
        assert not any("未启用" in signal.render().plain for signal in app.query(".environment-signal"))
        assert all(panel.region.contains_region(button.region) for button in visible_buttons(panel))
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


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_first_issue_requires_explicit_selection_without_stealing_entry_focus(app, language):
    app, probe = app
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    probe.return_value = [
        PreflightCheck("RAPL", "warn", "first issue evidence", check_id="cpu_energy"),
        PreflightCheck("GPU power", "warn", "second issue evidence", check_id="gpu_power"),
    ]
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        entry = app.query_one("#environment-warning")
        entry.focus()
        await pilot.press("enter")
        await pilot.pause()
        panel = app.preflight.panel
        assert app.focused is entry
        assert panel._selected_row is None
        hint = detail_text(app)
        assert ("点击" if language == "zh" else "Click") in hint
        assert await pilot.hover("#preflight-signal-gpu_power")
        assert detail_text(app) == hint

        first = app.query_one("#preflight-signal-cpu_energy")
        first.scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert await pilot.click(first)
        assert "first issue evidence" in detail_text(app)
        callout = app.query_one("#preflight-callout", Callout)
        assert callout.display
        for kind in ("problem", "impact", "next"):
            value = callout.query_one(".callout-" + kind, Static)
            assert str(value.content)
            assert panel.region.contains_region(value.region)
        assert callout.query_one(".callout-details", Collapsible).collapsed
        retry = panel.retry_button
        assert panel.region.contains_region(retry.region)
        assert app.get_widget_at(*retry.region.center)[0] is retry

        second = app.query_one("#preflight-signal-gpu_power")
        second.scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert await pilot.hover(second)
        assert "first issue evidence" in detail_text(app)
        second.focus()
        await pilot.pause()
        assert "first issue evidence" in detail_text(app)
        await pilot.press("enter")
        assert "second issue evidence" in detail_text(app)
        assert app.focused is second
        probe.assert_called_once()


async def test_unknown_cached_status_uses_neutral_facts_without_another_probe(app):
    app, probe = app
    probe.return_value = [
        PreflightCheck("Torch", "future_status", "unrecognized cached evidence",
                       "future_capability", check_id="torch"),
    ]
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(app, pilot)
        await pilot.click("#environment-pending")
        await pilot.pause()
        signal = app.query_one("#preflight-signal-torch")
        assert signal.render().plain.startswith("- ")
        assert "future_status" not in signal.render().plain
        assert not app.query_one("#preflight-callout", Callout).display
        assert not app.query_one("#preflight-facts", FactGrid).display
        await expand_check_source(app, pilot, "torch")
        signal.scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert await pilot.click(signal)
        assert not app.query_one("#preflight-callout", Callout).display
        assert app.query_one("#preflight-facts", FactGrid).display
        assert "unrecognized cached evidence" in detail_text(app)
        assert app.get_widget_at(*app.preflight.panel.retry_button.region.center)[0] is app.preflight.panel.retry_button
        probe.assert_called_once()
