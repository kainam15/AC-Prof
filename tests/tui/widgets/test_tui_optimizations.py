"""Unit and Pilot tests for TUI shortcuts, summaries, caching, search, and layout."""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.geometry import Region
from textual.widgets import Button, ContentSwitcher, Input, Select, Static

from acprof.experiments.configuration import RunConfig
from acprof.host.images.inventory import DockerConnection, ImageInventory, ImageLayer, ManagedImage
from acprof.tui.busy_state import BusyControlCache, apply_busy_state
from acprof.tui.experiments.picker import PickerChoice
from acprof.tui.images.presentation import (
    ImageDetailPanel,
    ImageTree,
    ImageWorkspace,
    render_image_tree,
)
from acprof.tui.logging.widget import SelectableLog
from acprof.tui.models.candidates import ModelCandidate, model_choice
from acprof.tui.models.store_summary import model_store_facts
from acprof.tui.run.progress import ProgressSnapshot
from acprof.tui.run.stages import Stage
from acprof.tui.ui.shortcuts import ShortcutsScreen
from acprof.tui.ui.structured import StatusBanner
from tests.support.tui_fixtures import AcprofTui


def test_model_choice_uses_neutral_for_cached_unverified_and_structured_views() -> None:
    cached = ModelCandidate(
        model_id="demo/cached",
        revision="abcdef1234567890",
        status="cached_unverified",
        reason="verified cache entry; runtime validation required",
        conditions={"device": "cpu"},
        detail="{}",
    )
    choice = model_choice(cached)
    assert isinstance(choice, PickerChoice)
    assert choice.summary is not None
    assert choice.summary.level == "neutral"
    assert choice.callout is None
    assert choice.section is not None

    blocked = ModelCandidate(
        model_id="demo/gated",
        revision="1111222233334444",
        status="access_required",
        reason="401 unauthorized",
        conditions={},
        detail="{}",
    )
    blocked_choice = model_choice(blocked)
    assert blocked_choice.summary is not None
    assert blocked_choice.summary.level == "error"
    assert blocked_choice.callout is not None
    assert blocked_choice.callout.level == "error"


async def test_selectable_log_match_index_tracks_forward_reverse_and_reset() -> None:
    class _LogHarness(App[None]):
        def compose(self) -> ComposeResult:
            yield SelectableLog(id="log")

    app = _LogHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        log = app.query_one("#log", SelectableLog)
        log.write("[INFO] alpha one\n[WARN] beta\n[INFO] alpha two\n[ERROR] alpha three")
        await pilot.pause()
        log.set_view(query="alpha")
        assert log.match_count == 3
        assert log.current_match == 0
        assert log.find_match() is True
        assert log.current_match == 1
        assert log.find_match() is True
        assert log.current_match == 2
        assert log.find_match() is True
        assert log.current_match == 3
        assert log.find_match() is True
        assert log.current_match == 1
        assert log.find_match(previous=True) is True
        assert log.current_match == 3
        assert log.find_match(previous=True) is True
        assert log.current_match == 2

        log.set_view(query="")
        assert log.match_count == 0
        assert log.current_match == 0
        assert log.find_match(previous=True) is False


async def test_busy_control_cache_skips_redundant_scans_and_respects_measurement(tmp_path) -> None:
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(120, 30)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        cache = BusyControlCache()
        cache.capture(app)
        assert cache.captured

        # Warm the per-tab nav target once, then verify repeated set_busy toggles perform
        # zero queries inside BusyControlCache/apply_busy_state itself.
        start_run = app.query_one("#start-run", Button)
        job_token = object()
        app.read_jobs.add(job_token)
        app.set_busy(True)
        app.read_jobs.remove(job_token)
        app.set_busy(False)
        app.query_one("#model", Input).focus()

        with patch.object(app.images, "update_image_controls"), \
             patch.object(app.images, "sync_image_refresh_timer"), \
             patch.object(app.preflight, "refresh_preflight_state"), \
             patch.object(app.profiles, "update_action_availability"), \
             patch.object(app.results, "update_report_actions"), \
             patch.object(app.updates, "refresh_controls"), \
             patch.object(app.preflight, "pause_auto_check"), \
             patch.object(app.preflight, "resume_auto_check"), \
             patch.object(app, "query", side_effect=AssertionError("unexpected DOM scan")), \
             patch.object(app, "query_one", side_effect=AssertionError("unexpected DOM scan")):
            app.read_jobs.add(job_token)
            app.set_busy(True)
            assert start_run.disabled
            assert cache.general_controls[0].disabled
            app.read_jobs.remove(job_token)
            app.set_busy(False)
            assert not start_run.disabled
            assert not cache.general_controls[0].disabled

        # measurement_active keeps resize disabled and image controls locked even when busy=False.
        app.execution.session.snapshot = ProgressSnapshot(measurement_active=True)
        apply_busy_state(app, cache, False)
        assert cache.tables
        assert all(not table.resize_enabled for table in cache.tables if table.is_mounted)
        assert all(widget.disabled for widget in cache.image_controls if widget.is_mounted)

        cache.clear()
        assert not cache.captured


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_shortcuts_modal_and_run_tab_f3_preserve_single_plan_summary(tmp_path, language) -> None:
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        from acprof.tui.run.plan_panel import RunPlanPanel
        assert len(app.query(RunPlanPanel)) == 1
        assert not app.query("#run-config-summary")

        # Typing '?' inside #model inserts a literal '?' instead of opening shortcuts.
        model_input = app.query_one("#model", Input)
        model_input.focus()
        await pilot.pause()
        await pilot.press("question_mark")
        await pilot.pause()
        assert model_input.value.endswith("?")
        assert not isinstance(app.screen, ShortcutsScreen)

        # F1 opens ShortcutsScreen even when focus is inside an Input.
        await pilot.press("f1")
        await pilot.pause()
        assert isinstance(app.screen, ShortcutsScreen)
        dialog = app.screen.query_one("#shortcuts-dialog")
        assert dialog.region.bottom <= 24
        assert app.screen.query_one("#shortcuts-banner", StatusBanner).summary.level == "neutral"
        assert not app.screen.query("#shortcuts-global Collapsible")
        await pilot.press("question_mark")
        await pilot.pause()
        assert not isinstance(app.screen, ShortcutsScreen)

        # '?' outside an Input opens ShortcutsScreen; Esc closes it.
        app.query_one("#start-run", Button).focus()
        await pilot.pause()
        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, ShortcutsScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, ShortcutsScreen)

        # F3 switches between basic run-form and advanced-form.
        switcher = app.query_one("#experiment-pages", ContentSwitcher)
        assert switcher.current == "run-form"
        await pilot.press("f3")
        await pilot.pause()
        assert switcher.current == "advanced-form"
        app.query_one("#warmup", Input).value = "2"
        await pilot.pause()
        await pilot.press("f3")
        await pilot.pause()
        assert switcher.current == "run-form"
        assert app.configuration.collect_config().warmup == 2
        assert len(app.query(RunPlanPanel)) == 1
        assert not app.query("#run-config-summary")


async def test_monitor_log_search_match_index_and_shift_enter_reverse_jump(tmp_path) -> None:
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(120, 30)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.activate_tab("monitor-tab")
        await pilot.pause()

        log = app.query_one("#run-log", SelectableLog)
        log.write("[INFO] match alpha\n[INFO] filler\n[WARN] match beta\n[ERROR] match gamma")
        await pilot.click("#log-tools")
        await pilot.pause()
        search = app.query_one("#log-search", Input)
        search.focus()
        search.value = "match"
        await pilot.pause()

        status = app.query_one("#log-status", Static)
        assert "/3" not in str(status.content)
        await pilot.press("enter")
        await pilot.pause()
        assert "匹配 1/3" in str(status.content)
        await pilot.press("enter")
        await pilot.pause()
        assert "匹配 2/3" in str(status.content)
        await pilot.press("shift+enter")
        await pilot.pause()
        assert "匹配 1/3" in str(status.content)
        await pilot.press("shift+enter")
        await pilot.pause()
        assert "匹配 3/3" in str(status.content)

        # Tab from #log-search still goes straight to #export-log without extra buttons.
        await pilot.press("tab")
        await pilot.pause()
        assert app.focused is app.query_one("#export-log", Button)

        # Clearing query removes match fraction and English updates status cleanly.
        search.value = ""
        await pilot.pause()
        assert "/" not in str(status.content)
        search.value = "match"
        await pilot.pause()
        search.focus()
        await pilot.press("enter")
        app.query_one("#ui-language", Select).value = "en"
        await pilot.pause()
        assert "Match 1/3" in str(status.content)

        # Measurement active disables log search input.
        app.consume_process_line("", ProgressSnapshot(stage=Stage.MEASURING, measurement_active=True), True)
        await pilot.pause()
        assert search.disabled


async def test_image_tree_cursor_line_refreshes_affected_lines_and_panel_facts(tmp_path) -> None:
    root = ManagedImage("sha256:" + "1" * 64, ("acprof/runtime:1",), 300, "2026-01-01T00:00:00Z", "runtime", acprof=True)
    child = ManagedImage(
        "sha256:" + "2" * 64,
        ("acprof/weights:1",),
        400,
        "2026-01-01T00:00:00Z",
        "model",
        acprof=True,
        parent_id=root.image_id,
        ancestor_ids=(root.image_id,),
    )
    leaf = ManagedImage(
        "sha256:" + "3" * 64,
        ("acprof/final:1",),
        500,
        "2026-01-01T00:00:00Z",
        "final",
        acprof=True,
        parent_id=child.image_id,
        ancestor_ids=(root.image_id, child.image_id),
    )
    third_party = ManagedImage(
        "sha256:" + "9" * 64,
        ("ubuntu:24.04",),
        200,
        "2026-01-01T00:00:00Z",
        "other",
        acprof=False,
    )
    inventory = ImageInventory(
        DockerConnection((), "docker"),
        "daemon",
        (root, child, leaf, third_party),
        layers=(ImageLayer("sha256:" + "a" * 64, "sha256:" + "b" * 64, 128, (leaf.image_id,)),),
    )

    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    with patch.object(app.images, "refresh_images"):
        async with app.run_test(size=(120, 30)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            app.activate_tab("images-tab")
            await pilot.pause()

            tree = app.query_one("#image-tree", ImageTree)
            render_image_tree(
                tree,
                inventory,
                (root, child, leaf, third_party),
                set(),
                root.image_id,
                app.tr,
                100,
                show_external_ancestors=True,
            )
            await pilot.pause()
            tree.root.children[0].expand_all()
            await pilot.pause()

            refreshed_lines: list[int] = []
            with patch.object(tree, "refresh_line", side_effect=refreshed_lines.append) as refresh_line_mock, \
                 patch.object(tree, "refresh", wraps=tree.refresh) as refresh_mock:
                tree.cursor_line = 2
                await pilot.pause()
                assert refresh_line_mock.called
                assert set(refreshed_lines).issubset(set(range(0, 5)))
                assert all(call.args and isinstance(call.args[0], Region) and call.args[0].height == 1
                           for call in refresh_mock.call_args_list)

                refreshed_lines.clear()
                tree.watch_cursor_line(2, 2)
                assert not refreshed_lines

            panel = app.query_one(ImageDetailPanel)
            panel.show_image(third_party, inventory)
            await pilot.pause()
            assert panel.query_one("#image-detail-banner", StatusBanner).summary.level == "neutral"
            assert not panel.query("#image-detail-facts")
            assert "ubuntu:24.04" in str(panel.query_one("#image-metadata-detail", Static).content)

            panel.show_layer(inventory.layers[0], inventory)
            await pilot.pause()
            assert panel.query_one("#image-detail-banner", StatusBanner).summary.level == "neutral"
            assert not panel.query("#image-detail-facts")


def test_model_store_facts_omit_neutral_dash_prefix_for_zero_reclaimable() -> None:
    report = {"path": "/tmp/store", "total_bytes": 1024, "free_bytes": 4096, "capacity_bytes": None, "models": []}
    empty_preview = model_store_facts(report, {"entries": [], "reclaimable_bytes": 0})
    assert empty_preview.rows[-1].status is None
    assert empty_preview.rows[-1].value == "0 B"
    reclaim_preview = model_store_facts(report, {"entries": ["entry-1"], "reclaimable_bytes": 512})
    assert reclaim_preview.rows[-1].status == "warning"


async def test_environment_popover_fit_and_short_run_plan_heights(tmp_path) -> None:
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.has_class("short")
        assert app.query_one("#run-plan-body").styles.max_height is not None
        assert app.query_one("#run-plan-body").styles.max_height.value == 8

        app.query_one("#environment-success").focus()
        await pilot.press("enter")
        await pilot.pause()
        popover = app.query_one("#environment-popover")
        scroll = app.query_one("#preflight-scroll")
        caps = app.query_one("#preflight-capabilities")

        for width, height in ((60, 14), (55, 16), (80, 24), (120, 30), (120, 36), (150, 45)):
            await pilot.resize_terminal(width, height)
            await pilot.pause()
            assert popover.display
            assert popover.region.bottom <= height
            sections = [section.region for section in caps.query(".environment-section")]
            assert len(sections) >= 2
            assert all(region.width > 0 and popover.region.x <= region.x < region.right <= popover.region.right
                       for region in sections)
            assert all(first.bottom <= second.y for first, second in zip(sections, sections[1:], strict=False))
            assert 1 <= int(scroll.styles.max_height.value) <= 10

        app.configuration.preferences = replace(app.configuration.preferences, show_command_bar=True)
        app.configuration.apply_ui_preferences()
        await pilot.pause()
        for width, height in ((80, 24), (120, 30), (120, 36), (150, 45)):
            await pilot.resize_terminal(width, height)
            await pilot.pause()
            popover.fit()
            await pilot.pause()
            bottom_y = app.query_one("#bottom-panel").region.y
            assert popover.region.bottom <= bottom_y, f"overlap at {(width, height)}: {popover.region} vs y={bottom_y}"


async def test_queued_callbacks_after_app_exit_release_jobs_without_accessing_widgets(tmp_path) -> None:
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        workspace = app.query_one(ImageWorkspace)
    workspace._apply_detail_height()
    token = object()
    app.read_jobs.add(token)
    app.configuration.ui_preference_changed()
    app.images.image_filter_changed()
    app.logs.view_changed()
    app.results.result_source_changed()
    app.results.analysis_export_finished(token, "export finished after exit")
    assert token not in app.read_jobs
    html_token = object()
    app.read_jobs.add(html_token)
    app.results.html.file_finished(html_token, "HTML action finished after exit")
    assert html_token not in app.read_jobs
