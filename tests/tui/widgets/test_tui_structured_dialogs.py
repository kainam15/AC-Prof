"""Tests for structured modal views, sorting, keyboard navigation, and HTML facts."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from textual.app import App
from textual.widgets import Button, Collapsible, DataTable, Input, Static, TabbedContent

from acprof.experiments.configuration import RunConfig
from acprof.host.images.inventory import DiskSpace, DockerConnection, DockerStorage, StorageUsage
from acprof.tui.commands import PendingLaunch
from acprof.tui.experiments.picker import PickerChoice, PickerTable, SearchPickerScreen
from acprof.tui.i18n import translate
from acprof.tui.images.storage import StorageSpaceScreen
from acprof.tui.logging.export_screen import ExportDirectoryScreen, LogExportScreen
from acprof.tui.models.store import ModelStoreScreen
from acprof.tui.results.profile_screen import ProfilerChoiceScreen
from acprof.tui.results.profile_status import ProfilerOverview, ProfilerStatus
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.run.preparation import PreparationScreen
from acprof.tui.run.recovery import RecoveryReview
from acprof.tui.run.recovery_controller import RecoveryScreen
from acprof.tui.settings.environment import EnvironmentSettingsScreen
from acprof.tui.ui.dialogs import ConfirmActionScreen
from acprof.tui.ui.structured import Callout, FactDetailScreen, FactGrid, FactValue, StatusBanner
from acprof.tui.ui.table import ResizableDataTable
from acprof.tui.ui.view_models import CalloutContent, FactRow, StatusSummary, ViewSection
from tests.support.model_store_fixtures import ModelStoreFixture
from tests.support.tui_fixtures import AcprofTui


class _ModalHarness(App[None]):
    def tr(self, text: str) -> str:
        return translate(text, "zh")


async def test_search_picker_sorting_keyboard_navigation_and_structured_details() -> None:
    structured_a = PickerChoice(
        value="z-item",
        cells=("z-model", "已完成"),
        detail="raw-z",
        search_text="z-model",
        actions=frozenset({"view", "reuse"}),
        summary=StatusSummary("success", "z-model · 已完成"),
        callout=None,
        section=ViewSection("详情", (FactRow("目录", "/tmp/z"),)),
    )
    structured_b = PickerChoice(
        value="a-item",
        cells=("a-model", "失败"),
        detail="raw-a",
        search_text="a-model",
        actions=frozenset({"view", "resume"}),
        summary=StatusSummary("error", "a-model · 失败"),
        callout=CalloutContent("error", "模型加载失败", "未开始测量", "检查路径"),
        section=ViewSection("详情", (FactRow("目录", "/tmp/a"),)),
    )
    plain_choice = PickerChoice(
        value="m-item",
        cells=("m-model", "中性"),
        detail="plain-detail-only",
        search_text="m-model",
        actions=frozenset({"reuse"}),
    )

    screen = SearchPickerScreen(
        "选择实验",
        ("模型", "状态"),
        lambda cancelled: ((structured_a, structured_b, plain_choice), ()),
        actions=("view", "reuse", "resume"),
    )
    results: list[tuple[str, object] | None] = []
    app = _ModalHarness()
    async with app.run_test(size=(100, 30)) as pilot:
        app.push_screen(screen, results.append)
        await app.workers.wait_for_complete()
        await pilot.pause()

        table = screen.query_one("#picker-table", PickerTable)
        assert isinstance(table, ResizableDataTable)
        search = screen.query_one("#picker-search", Input)
        assert app.focused is search

        # Left/Right inside search Input should edit cursor position, not jump buttons.
        search.value = "model"
        search.cursor_position = 3
        await pilot.press("left")
        assert search.cursor_position == 2
        assert app.focused is search

        # Column header sort ascending then descending (preserves current selection on z-item;
        # moving to row 0 inspects a-item's error banner & callout).
        col_key = table.ordered_columns[0].key
        table.post_message(DataTable.HeaderSelected(table, col_key, 0, "模型"))
        await pilot.pause()
        assert screen.sort_state == (0, False)
        assert [c.value for c in screen.filtered] == ["a-item", "m-item", "z-item"]
        assert "↑" in str(table.ordered_columns[0].label)
        assert screen.selected() is structured_a
        table.move_cursor(row=0, animate=False)
        await pilot.pause()
        assert screen.query_one("#picker-detail-banner", StatusBanner).display
        assert screen.query_one("#picker-detail-callout", Callout).display
        assert screen.query_one("#picker-detail-facts", FactGrid).display
        assert not screen.query_one("#picker-detail", Static).display

        table.post_message(DataTable.HeaderSelected(table, col_key, 0, "模型"))
        await pilot.pause()
        assert screen.sort_state == (0, True)
        assert [c.value for c in screen.filtered] == ["z-item", "m-item", "a-item"]
        assert "↓" in str(table.ordered_columns[0].label)

        # Plain fallback when choice has no structured summary/callout/section.
        search.value = "m-model"
        await pilot.pause()
        assert len(screen.filtered) == 1
        assert not screen.query_one("#picker-detail-banner", StatusBanner).display
        assert screen.query_one("#picker-detail", Static).display
        assert "plain-detail-only" in str(screen.query_one("#picker-detail", Static).content)

        # Empty filter makes Down arrow a safe no-op.
        search.value = "no-such-match"
        await pilot.pause()
        await pilot.press("down")
        assert app.focused is search

        # Restoring search, Left/Right moves across enabled #picker-actions buttons.
        search.value = "a-model"
        await pilot.pause()
        screen.query_one("#picker-view", Button).focus()
        await pilot.press("right")
        assert app.focused is screen.query_one("#picker-resume", Button)
        await pilot.press("left")
        assert app.focused is screen.query_one("#picker-view", Button)

        # Down from #picker-search focuses #picker-table; Enter activates primary enabled action.
        search.focus()
        await pilot.press("down")
        await pilot.pause()
        assert app.focused is table
        await pilot.press("enter")
        await pilot.pause()
        assert results == [("view", "a-item")]


async def test_confirm_preparation_and_recovery_screens_structured_and_arrow_keys(tmp_path) -> None:
    app = _ModalHarness()
    async with app.run_test(size=(100, 30)) as pilot:
        confirm = ConfirmActionScreen("确认", "继续吗？", "确定")
        app.push_screen(confirm)
        await pilot.pause()
        confirm.query_one("#confirm-no", Button).focus()
        await pilot.press("right")
        assert app.focused is confirm.query_one("#confirm-yes", Button)
        await pilot.press("left")
        assert app.focused is confirm.query_one("#confirm-no", Button)
        confirm.dismiss(False)
        await pilot.pause()

        prep = PreparationScreen({
            "stage": "runtime",
            "request": {
                "id": 1,
                "kind": "error",
                "failed_stage": "load",
                "detail": "RuntimeError: missing file",
            },
        })
        app.push_screen(prep)
        await pilot.pause()
        assert prep.query_one("#preparation-banner", StatusBanner).summary.level == "error"
        callout = prep.query_one("#preparation-callout", Callout)
        assert "模型未能完成加载" in str(callout.content.problem)
        assert not list(prep.query("#preparation-detail"))
        prep.query_one("#preparation-cancel", Button).focus()
        await pilot.press("right")
        assert app.focused is prep.query_one("#preparation-continue", Button)
        await pilot.press("left")
        assert app.focused is prep.query_one("#preparation-cancel", Button)
        prep.dismiss(None)
        await pilot.pause()

        pending = PendingLaunch(["acprof", "run"], "run", RunConfig.smoke("demo/model"))
        recovery = RecoveryScreen(RecoveryReview(
            directory=tmp_path / "exp",
            phase="measurement",
            detail="partial windows exist",
            resume=pending,
            restart=pending,
        ))
        app.push_screen(recovery)
        await pilot.pause()
        assert recovery.query_one("#recovery-banner", StatusBanner).summary.level == "neutral"
        assert recovery.query_one("#recovery-callout", Callout).display
        recovery_facts = recovery.query_one("#recovery-facts", FactGrid)
        assert recovery_facts.display
        assert not list(recovery.query("#recovery-detail"))
        assert not list(recovery_facts.query(Collapsible))
        fact_values = list(recovery_facts.query(FactValue))
        assert fact_values[1].has_class("status-neutral")
        assert str(fact_values[1].render()).startswith("- ")
        assert fact_values[2].has_class("status-success")
        assert str(fact_values[2].render()).startswith("✓ ")
        recovery.query_one("#recovery-cancel", Button).focus()
        await pilot.press("right")
        assert app.focused is recovery.query_one("#recovery-resume", Button)
        await pilot.press("right")
        assert app.focused is recovery.query_one("#recovery-new", Button)
        await pilot.press("left")
        assert app.focused is recovery.query_one("#recovery-resume", Button)
        recovery.dismiss(None)
        await pilot.pause()

        completed_recovery = RecoveryScreen(RecoveryReview(
            directory=tmp_path / "done",
            phase="complete",
            detail="already completed",
            resume=None,
            restart=pending,
        ))
        app.push_screen(completed_recovery)
        await pilot.pause()
        completed_values = list(completed_recovery.query_one("#recovery-facts", FactGrid).query(FactValue))
        assert completed_values[1].has_class("status-success")
        assert str(completed_values[1].render()).startswith("✓ ")
        assert completed_values[2].has_class("status-warning")
        assert str(completed_values[2].render()).startswith("! ")
        completed_recovery.dismiss(None)
        await pilot.pause()


async def test_storage_and_model_store_screens_use_structured_views_and_arrow_nav(tmp_path) -> None:
    app = _ModalHarness()
    storage_data = DockerStorage(
        DockerConnection((), "docker"),
        "daemon",
        "/var/lib/docker",
        DiskSpace(1000, 400, 600),
        (
            StorageUsage("Images", 200, 50),
            StorageUsage("Containers", 50, 0),
            StorageUsage("Local Volumes", 0, 0),
            StorageUsage("Build Cache", 150, 150),
        ),
    )

    store_fixture = ModelStoreFixture(tmp_path / "store")
    blob = store_fixture.blob("weights.bin", 1024)
    store_fixture.entry(1, [blob])

    async with app.run_test(size=(100, 30)) as pilot:
        storage_screen = StorageSpaceScreen(lambda: None, selected_count=1)
        app.push_screen(storage_screen)
        await pilot.pause()
        storage_screen.show_storage(storage_data, 50)
        await pilot.pause()
        assert storage_screen.query_one("#storage-banner", StatusBanner).summary.level == "success"
        # Current main distinguishes environment, disk, usage and selection readings.
        assert storage_screen.query_one("#storage-root", Static).content == "/var/lib/docker"
        for key in ("environment", "disk", "usage", "selection"):
            assert "本次" in str(storage_screen.query_one(f"#storage-{key}-state", Static).content)
        assert storage_screen.query_one("#storage-disk-percent", Static).content == "40.0%"
        storage_screen.query_one("#storage-refresh", Button).focus()
        await pilot.press("right")
        assert app.focused is storage_screen.query_one("#storage-close", Button)
        await pilot.press("left")
        assert app.focused is storage_screen.query_one("#storage-refresh", Button)
        storage_screen.dismiss(None)
        await pilot.pause()

        store_screen = ModelStoreScreen(store_fixture.root)
        app.push_screen(store_screen)
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert store_screen.query_one("#store-banner", StatusBanner).summary.level == "success"
        assert store_screen.query_one("#store-facts", FactGrid).display
        assert not list(store_screen.query(Collapsible))
        assert not list(store_screen.query("#store-report"))
        assert store_screen.query_one("#store-note", Static).display
        table = store_screen.query_one("#store-table", ResizableDataTable)
        assert table.display
        assert table.row_count == 1
        store_screen.query_one("#store-close", Button).focus()
        await pilot.press("right")
        assert app.focused is store_screen.query_one("#store-refresh", Button)


async def test_results_html_panel_banner_facts_and_legacy_controls_hidden(tmp_path) -> None:
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(120, 30)) as pilot:
        await app.workers.wait_for_complete()
        app.activate_tab("results-tab")
        app.results.show_result_section("statistics-view")
        await pilot.pause()

        banner = app.query_one("#html-report-banner", StatusBanner)
        assert banner.summary.level == "neutral"
        assert app.query_one("#html-legacy-controls").disabled
        assert not app.query_one("#html-legacy-controls").display

        report = tmp_path / "report.html"
        report.write_text("<!doctype html><html></html>", encoding="utf-8")
        app.query_one("#html-report-path", Input).value = str(report)
        app.results.html.freshness = "current"
        with patch("acprof.platform.can_open_local_browser", return_value=False):
            app.results.update_report_actions()
            await pilot.pause()
            facts = app.query_one("#html-report-facts", FactGrid)
            assert len(facts.section.rows) == 2
            assert not list(facts.query(Collapsible))
            freshness_value = list(facts.query(FactValue))[0]
            assert freshness_value.has_class("status-success")
            assert str(freshness_value.render()).startswith("✓ ")
            app.results.html.open()
            await pilot.pause()
            assert banner.summary.level == "warning"


async def test_search_picker_flat_facts_and_stable_column_widths_across_sorts() -> None:
    choice_a = PickerChoice(
        value="m1",
        cells=("m1", "c1"),
        detail="m1 detail",
        search_text="m1 c1",
        actions=frozenset({"select"}),
        summary=StatusSummary("success", "ok"),
        section=ViewSection("", (FactRow("模型", "m1", status="ok"),)),
    )
    choice_b = PickerChoice(
        value="m2",
        cells=("m2", "c2"),
        detail="m2 detail",
        search_text="m2 c2",
        actions=frozenset({"select"}),
        summary=StatusSummary("neutral", "idle"),
        section=ViewSection("", (FactRow("模型", "m2"),)),
    )
    app = _ModalHarness()
    screen = SearchPickerScreen("选择模型", ("模型", "状态"), lambda _: ((choice_a, choice_b), ()), actions=("select",))
    async with app.run_test(size=(100, 30)) as pilot:
        app.push_screen(screen)
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not list(screen.query("#picker-detail-facts Collapsible"))
        table = screen.query_one("#picker-table", PickerTable)
        base_widths = [column.content_width for column in table.ordered_columns]
        headers = [column.key for column in table.ordered_columns]
        screen.sort_column(DataTable.HeaderSelected(table, headers[0], 0, "模型"))
        await pilot.pause()
        assert table.ordered_columns[0].content_width > base_widths[0]
        screen.sort_column(DataTable.HeaderSelected(table, headers[1], 1, "状态"))
        await pilot.pause()
        assert table.ordered_columns[0].content_width == base_widths[0]
        screen.sort_column(DataTable.HeaderSelected(table, headers[0], 0, "模型"))
        await pilot.pause()
        screen.sort_column(DataTable.HeaderSelected(table, headers[1], 1, "状态"))
        await pilot.pause()
        assert [column.content_width for column in table.ordered_columns] == [base_widths[0], table.ordered_columns[1].content_width]


async def test_preparation_download_confirmation_and_review_render_fact_grids(tmp_path) -> None:
    app = _ModalHarness()
    download_report = {
        "expected_download_bytes": 1024,
        "max_download_bytes": None,
        "model_store_path": str(tmp_path / "hf-store"),
        "model": {"total_bytes": 2048, "cached_bytes": 1024, "endpoint": "https://huggingface.co/demo/model"},
        "disk": {"total_bytes": 10_000, "free_bytes": 8_000, "reclaimable_bytes": 500, "remaining_bytes": 6_976},
        "docker_storage": {"docker_storage_total_bytes": 4_000, "docker_storage_available_bytes_at_start": 3_000},
        "runtime": {"kind": "docker", "source_mode": "pull", "will_attempt_ghcr_pull": False},
        "sources": [],
        "network_probe": [{
            "category": "model",
            "method": "HEAD+GET",
            "status": "passed",
            "endpoint": "https://huggingface.co/demo/model",
            "final_host": "cdn.hf.co",
            "http_status": 200,
            "received_bytes": 64,
            "elapsed_seconds": 0.1,
        }],
        "probe_payload_bytes": 64,
        "probe_byte_limit": 262_144,
    }
    async with app.run_test(size=(100, 32)) as pilot:
        dl_screen = PreparationScreen({
            "stage": "model",
            "status": "waiting",
            "request": {
                "id": 1,
                "kind": "review",
                "resolved": True,
                "questions": [],
                "download_report": download_report,
            },
        })
        app.push_screen(dl_screen)
        await pilot.pause()
        assert len(list(dl_screen.query(".download-section").results(FactGrid))) == 4
        assert not list(dl_screen.query(Collapsible))
        assert not list(dl_screen.query("#preparation-detail"))
        dl_screen.dismiss(None)
        await pilot.pause()

        review_screen = PreparationScreen({
            "stage": "resolution",
            "status": "waiting",
            "request": {
                "id": 2,
                "kind": "review",
                "resolved": True,
                "questions": [],
                "fields": {"Model": "demo/model", "Revision": None},
            },
        })
        app.push_screen(review_screen)
        await pilot.pause()
        fields_grid = review_screen.query_one("#preparation-fields", FactGrid)
        assert not list(fields_grid.query(Collapsible))
        values = list(fields_grid.query(FactValue))
        assert str(values[0].render()).startswith("✓ ")
        assert values[0].has_class("status-success")
        assert str(values[1].render()).startswith("- ")
        assert values[1].has_class("status-neutral")


async def test_remaining_modals_support_horizontal_arrow_navigation(tmp_path) -> None:
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(100, 32)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        fact_modal = FactDetailScreen(FactRow("模型", "demo/model", copyable=True))
        app.push_screen(fact_modal)
        await pilot.pause()
        fact_modal.query_one("#fact-copy", Button).focus()
        await pilot.press("right")
        assert app.focused is fact_modal.query_one("#fact-close", Button)
        await pilot.press("left")
        assert app.focused is fact_modal.query_one("#fact-copy", Button)
        fact_modal.dismiss(None)
        await pilot.pause()

        overview = ProfilerOverview(Path("/workspace/results/fixture"), (
            ProfilerStatus("torch", "complete", 8, 8, 0),
            ProfilerStatus("ncu", "missing", 0, 4, 0),
        ))
        profiler_modal = ProfilerChoiceScreen(overview)
        app.push_screen(profiler_modal)
        await pilot.pause()
        profiler_modal.query_one("#profile-choice-close", Button).focus()
        await pilot.press("right")
        assert app.focused is profiler_modal.query_one("#profile-choice-plan", Button)
        await pilot.press("right")
        assert app.focused is profiler_modal.query_one("#profile-choice-run", Button)
        await pilot.press("left")
        assert app.focused is profiler_modal.query_one("#profile-choice-plan", Button)
        profiler_modal.dismiss(None)
        await pilot.pause()

        def load_dirs(screen: ExportDirectoryScreen, directory: Path, _nearest: bool) -> None:
            screen.call_later(screen.loaded, directory, (), "")

        export_modal = LogExportScreen(
            tmp_path,
            "test.log",
            load_dirs,
            filtered_lines=("line 1",),
            retained_lines=("line 1",),
        )
        app.push_screen(export_modal)
        await pilot.pause()
        export_modal.query_one("#log-export-cancel", Button).focus()
        await pilot.press("right")
        assert app.focused is export_modal.query_one("#log-export-save", Button)
        await pilot.press("left")
        assert app.focused is export_modal.query_one("#log-export-cancel", Button)
        export_modal.dismiss(None)
        await pilot.pause()

        dir_modal = ExportDirectoryScreen(tmp_path, load_dirs)
        app.push_screen(dir_modal)
        await pilot.pause()
        dir_modal.query_one("#export-directory-parent", Button).focus()
        await pilot.press("right")
        assert app.focused is dir_modal.query_one("#export-directory-home", Button)
        dir_modal.query_one("#export-directory-cancel", Button).focus()
        await pilot.press("right")
        assert app.focused is dir_modal.query_one("#export-directory-use", Button)
        dir_modal.dismiss(None)
        await pilot.pause()

        env_modal = EnvironmentSettingsScreen(tmp_path, tmp_path / "backups")
        app.push_screen(env_modal)
        await pilot.pause()
        env_modal.query_one("#close-environment-settings", Button).focus()
        await pilot.press("right")
        assert app.focused is env_modal.query_one("#save-environment-settings", Button)
        env_modal.query_one("#environment-tabs", TabbedContent).active = "permissions-tab"
        await pilot.pause()
        env_modal.query_one("#check-environment-permissions", Button).focus()
        await pilot.press("right")
        assert app.focused is env_modal.query_one("#configure-environment-permissions", Button)
        env_modal.dismiss(False)
        await pilot.pause()


async def test_environment_settings_permissions_uses_canonical_preflight_presentations(tmp_path) -> None:
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    checks = [
        PreflightCheck("Docker", "ok", "daemon ready", check_id="docker"),
        PreflightCheck("perf", "fail", "permission denied", check_id="cpu_instructions"),
        PreflightCheck("RAPL", "warn", "partial access", check_id="cpu_energy"),
        PreflightCheck("Torch", "pending", "not probed yet", check_id="torch"),
    ]
    async with app.run_test(size=(100, 32)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        env_modal = EnvironmentSettingsScreen(tmp_path, tmp_path / "backups")
        with patch("acprof.tui.settings.environment.quick_preflight", return_value=checks):
            app.push_screen(env_modal)
            await pilot.pause()
            env_modal.query_one("#environment-tabs", TabbedContent).active = "permissions-tab"
            await pilot.pause()
            assert await pilot.click("#check-environment-permissions")
            await app.workers.wait_for_complete()
            await pilot.pause()
            report = str(env_modal.query_one("#permission-report", Static).content)
            assert "✓ Docker [通过]: daemon ready" in report
            assert "× perf 指令计数 [不可用]: permission denied" in report
            assert "! RAPL CPU 能耗 [不可用]: partial access" in report
            assert "- Torch Profiler [待检查]: not probed yet" in report
            assert "[未知]" not in report


async def test_picker_queued_events_after_dismiss_do_not_touch_removed_widgets() -> None:
    app = _ModalHarness()
    screen = SearchPickerScreen("选择实验", ("模型",), lambda cancelled: ((), ()))
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(screen)
        await app.workers.wait_for_complete()
        await pilot.pause()
        table = screen.query_one("#picker-table", PickerTable)
        event = DataTable.HeaderSelected(table, table.ordered_columns[0].key, 0, "模型")
        await pilot.press("escape")
        await pilot.pause()
        assert not screen.is_attached
        # Events may already be queued when the screen is popped.
        screen.on_resize()
        screen.sort_column(event)
        screen.activate_selected_row()
        screen.search_changed()
        screen.row_highlighted()
        screen.loaded((PickerChoice("late", ("late",), "", "", frozenset()),), (), (), ())
        assert not screen.choices


async def test_model_store_late_worker_callbacks_after_unmount_are_ignored(tmp_path) -> None:
    app = _ModalHarness()
    screen = ModelStoreScreen(tmp_path)
    with patch.object(screen, "refresh_report"), patch.object(screen, "apply_prune") as prune:
        async with app.run_test(size=(80, 24)) as pilot:
            app.push_screen(screen)
            await pilot.pause()
            screen.preview = {"entries": ["late-entry"], "reclaimable_bytes": 0}
            await app.pop_screen()
            await pilot.pause()
            assert not screen.is_attached
            screen._show_capacity({"path": str(tmp_path), "models": []})
            screen._show_waiting()
            screen._complete()
            screen._pruned()
            screen._confirmed(True)
            prune.assert_not_called()
