"""Storage modal interaction, accounting display and measurement exclusion."""
import asyncio
import os
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from io import StringIO
from pathlib import Path
from threading import Event
from unittest.mock import patch

import pytest
from rich.cells import cell_len
from rich.console import Console
from textual.widgets import Button, DataTable, Static

from acprof.experiments.configuration import RunConfig
from acprof.host.images.inventory import (
    DiskSpace,
    DockerConnection,
    DockerStorage,
    ImageManagementError,
    StorageUsage,
    list_images,
)
from acprof.host.images.storage_errors import storage_query_error
from acprof.tui.run.progress import ProgressSnapshot
from acprof.tui.ui.structured import StatusBanner
from tests.runtime.containers.test_image_management import FINAL, DockerFixture
from tests.support.environment_fixtures import isolated_environment
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition


class TestTuiStorage:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path):
        self._request = request
        temporary = tmp_path
        self.directory = Path(str(temporary))
        self.docker = DockerFixture()
        for patcher in (patch("acprof.host.images.inventory.run_command", side_effect=self.docker.run),
                        patch.dict(os.environ, isolated_environment(), clear=True),
                        patch.object(AcprofTui, "IMAGE_REFRESH_INTERVAL", 3600)):
            patcher.start()
            self._request.addfinalizer(partial(patcher.stop))

    def make_app(self):
        return AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.directory / "tui.json")

    async def load_images(self, app, pilot):
        await pilot.pause()
        app.images.show_images()
        await pilot.pause()
        await app.workers.wait_for_complete()
        app.images.refresh_timer.pause()
        await pilot.pause()
        assert not (app.is_busy())
        # 树首次聚焦公共环境；选择最后一个模型节点用于勾选回归。
        await pilot.click("#image-view-list")
        table = app.query_one("#image-table", DataTable)
        table.move_cursor(row=table.get_row_index(FINAL))
        await pilot.pause()

    @staticmethod
    def snapshot(app):
        return DockerStorage(
            app.images.inventory.connection, app.images.inventory.daemon_id, "/var/lib/docker",
            DiskSpace(512 * 1024**3, 384 * 1024**3, 128 * 1024**3),
            tuple(StorageUsage(kind, used * 1024**3, reclaim * 1024**3) for kind, used, reclaim in (
                ("Images", 46, 18), ("Containers", 3, 1), ("Local Volumes", 12, 2), ("Build Cache", 9, 9),
            )),
            observed_at=datetime(2026, 11, 16, 4, 5, 6, tzinfo=UTC),
        )

    @staticmethod
    def rendered(widget):
        stream = StringIO()
        Console(file=stream, width=72, color_system=None).print(widget.content)
        return stream.getvalue()

    async def open_storage(self, app, pilot):
        assert (await pilot.click("#image-storage"))
        await wait_for_condition(pilot, lambda: (
            app.images.storage_screen is not None and app.images.storage_screen.is_mounted
            and app.screen.query_one(StatusBanner).summary.level != "info" and not app.images.storage_loading
        ), description="storage popup result", timeout=5)
        await pilot.pause()

    async def refresh_storage(self, app, pilot):
        refresh = app.screen.query_one("#storage-refresh", Button)
        await wait_for_condition(pilot, lambda: not refresh.has_class("-active"),
                                 description="refresh button release", timeout=3)
        assert await pilot.click("#storage-refresh")
        await wait_for_condition(pilot, lambda: (
            not app.images.storage_loading and app.screen.query_one(StatusBanner).summary.level != "info"
        ), description="storage refresh result", timeout=5)
        await pilot.pause()

    async def test_storage_values_refresh_and_failure_recovery(self):
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-model")
            snapshot = self.snapshot(app)
            updated = replace(snapshot, disk=DiskSpace(512 * 1024**3, 256 * 1024**3, 256 * 1024**3))
            with patch("acprof.tui.images.controller.read_storage", side_effect=[
                snapshot, ImageManagementError("Docker 操作失败", "permission denied"), updated,
            ]) as read:
                await self.open_storage(app, pilot)
                assert (app.screen.query_one("#storage-root", Static).content) == ("/var/lib/docker")
                assert (app.screen.query_one("#storage-disk-percent", Static).content) == ("75.0%")
                bar = app.screen.query_one("#storage-disk-bar")
                assert (bar.region.height) == (1), "磁盘占用条应有可见高度"
                visible = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
                assert ("━") in (visible), "磁盘使用率需要可见的进度条"
                totals = self.rendered(app.screen.query_one("#storage-totals", Static))
                assert ("75.2 GB") in (totals)
                assert ("32.2 GB") in (totals)
                selection = self.rendered(app.screen.query_one("#storage-selection", Static))
                assert ("≈310 B") in (selection), "所选父子镜像的共享层只能计一次"
                assert ("≈137 GB") in (selection)
                for expected in ("permission denied", ""):
                    refresh = app.screen.query_one("#storage-refresh", Button)

                    async def button_ready():
                        while refresh.has_class("-active"):
                            await refresh.wait_for_refresh()

                    # Textual 在按下动画期间忽略重复点击；等待可观察状态恢复。
                    await asyncio.wait_for(button_ready(), timeout=3)
                    assert (await pilot.click("#storage-refresh"))
                    await pilot.pause()
                    await app.workers.wait_for_complete()
                    await pilot.pause()
                    status = self.rendered(app.screen.query_one("#storage-status", Static))
                    if expected:
                        assert "Socket 权限" in status
                        assert expected in self.rendered(app.screen.query_one("#storage-details", Static))
                        assert ("75.2 GB") not in (self.rendered(app.screen.query_one("#storage-totals", Static)))
                    else:
                        assert ("permission denied") not in (status)
                        assert ("≈275 GB") in (self.rendered(app.screen.query_one("#storage-selection", Static)))
                assert (read.call_count) == (3)
                await pilot.press("escape")
                await pilot.pause()
                assert not (app.is_busy())
                assert not (self.docker.removals)

    @pytest.mark.parametrize(("width", "height"), [(80, 24), (120, 30), (150, 45)])
    @pytest.mark.parametrize('language', ['zh', 'en'])
    async def test_modal_layout_language_keyboard_and_resize(self, width, height, language):
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot)
            snapshot = self.snapshot(app)
            with patch("acprof.tui.images.controller.read_storage", return_value=snapshot) as read:
                app.configuration.preferences = replace(app.configuration.preferences, language=language)
                app.configuration.apply_ui_preferences()
                await pilot.resize_terminal(80, 24)
                await pilot.pause()
                entry = app.query_one("#image-storage", Button)
                assert (cell_len(str(entry.label))) <= (entry.content_region.width)
                assert (app.get_widget_at(*entry.region.center)[0]) is (entry)
                await self.open_storage(app, pilot)
                title = "存储空间" if language == "zh" else "Storage space"
                assert (app.screen.query_one("#storage-dialog").border_title) == (title)
                assert ("容器" if language == "zh" else "Containers") in (self.rendered(app.screen.query_one("#storage-usage", Static)))
                await pilot.resize_terminal(width, height)
                await pilot.pause()
                dialog = app.screen.query_one("#storage-dialog")
                assert (dialog.region.x) > (0)
                assert (dialog.region.right) < (width)
                for selector in ("#storage-refresh", "#storage-close"):
                    button = app.screen.query_one(selector, Button)
                    assert (button.region.height) == (3)
                    assert (button.region.bottom) <= (height)
                    assert (app.get_widget_at(*button.region.center)[0]) is (button)
                content = app.screen.query_one("#storage-content")
                content.focus()
                await pilot.press("end")
                await pilot.pause()
                assert (content.scroll_y) >= (content.max_scroll_y)
                visible = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
                assert ("尚未勾选镜像。" if language == "zh" else "No images selected.") in (visible)
                refresh = app.screen.query_one("#storage-refresh", Button)
                refresh.focus()
                await pilot.press("tab")
                assert (app.focused) is (app.screen.query_one("#storage-close", Button))
                await pilot.press("shift+tab")
                assert (app.focused) is (refresh)
                await pilot.press("tab", "enter")
                await pilot.pause()
                assert (len(app.screen_stack)) == (1)
                assert not (app.is_busy())
                # Each parameter opens one dialog; the old two-language loop
                # accumulated one read per opening.
                read.assert_called_once()

    async def test_closing_during_query_keeps_launch_blocked_until_worker_finishes(self):
        app = self.make_app()
        entered, release = Event(), Event()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.load_images(app, pilot)
            snapshot = self.snapshot(app)

            def slow_read(*_args, **_kwargs):
                entered.set()
                if not release.wait(timeout=10):
                    raise TimeoutError("test worker was not released")
                return snapshot

            with patch("acprof.tui.images.controller.read_storage", side_effect=slow_read) as read:
                try:
                    assert (await pilot.click("#image-storage"))
                    await pilot.pause()
                    assert (await asyncio.to_thread(entered.wait, 2))
                    assert (app.screen.query_one("#storage-refresh", Button).disabled)
                    visible = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
                    assert "正在读取" in visible
                    assert "未知" not in visible
                    app.images.refresh_storage()
                    assert (read.call_count) == (1)
                    await pilot.press("escape")
                    await pilot.pause()
                    assert (len(app.screen_stack)) == (1)
                    assert (app.is_busy())
                    assert (app.query_one("#start-run", Button).disabled)
                    with patch("acprof.tui.images.controller.list_images") as inventory:
                        app.images.refresh_images()
                        inventory.assert_not_called()
                finally:
                    release.set()
                await app.workers.wait_for_complete()
                await pilot.pause()
                assert not (app.is_busy())
                assert not (app.query_one("#start-run", Button).disabled)

    async def test_measurement_prevents_storage_query(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            app.execution.session.snapshot = ProgressSnapshot(measurement_active=True)
            app.set_busy(True)
            with patch("acprof.tui.images.controller.read_storage") as read:
                app.images.open_storage()
                await pilot.pause()
                assert (app.query_one("#image-storage", Button).disabled)
                assert (len(app.screen_stack)) == (1)
                read.assert_not_called()
            app.execution.session.snapshot = ProgressSnapshot()
            app.set_busy(False)

    @pytest.mark.parametrize(("blocker", "reason"), [
        ("measurement", "正式测量"), ("preflight", "预检"), ("operation", "其他操作"),
    ])
    async def test_refresh_blocked_explains_reason_in_open_modal(self, blocker, reason):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.load_images(app, pilot)
            with patch("acprof.tui.images.controller.read_storage", return_value=self.snapshot(app)) as read:
                await self.open_storage(app, pilot)
                if blocker == "measurement":
                    app.execution.session.snapshot = ProgressSnapshot(measurement_active=True)
                elif blocker == "preflight":
                    app.preflight.check_running = True
                else:
                    app.images.operation = "delete"
                app.images.refresh_storage()
                await pilot.pause()
                visible = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
                assert reason in visible
                read.assert_called_once()
                assert not app.screen.query_one("#storage-refresh", Button).disabled
                app.execution.session.snapshot = ProgressSnapshot()
                app.preflight.check_running = False
                app.images.operation = "storage"
                await self.refresh_storage(app, pilot)
                assert read.call_count == 2

    @pytest.mark.parametrize("language", ["zh", "en"])
    async def test_initial_permission_error_is_visible_without_scrolling(self, language):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.load_images(app, pilot)
            app.configuration.preferences = replace(app.configuration.preferences, language=language)
            app.configuration.apply_ui_preferences()
            with patch("acprof.tui.images.controller.read_storage", side_effect=
                       ImageManagementError("Docker 操作失败", "permission denied")):
                await self.open_storage(app, pilot)
                visible = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
                assert ("Socket 权限" if language == "zh" else "Socket permission denied") in visible
                assert ("检查" if language == "zh" else "check") in visible
                assert not app.screen.query_one("#storage-refresh", Button).disabled
                content = app.screen.query_one("#storage-content")
                content.focus()
                await pilot.press("end")
                await pilot.pause()
                visible = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
                assert ("Socket 权限" if language == "zh" else "Socket permission denied") in visible

    @pytest.mark.parametrize("part", ["disk", "usage"])
    async def test_partial_failure_keeps_timestamped_last_read_and_other_fresh_data(self, part):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-model")
            snapshot = self.snapshot(app)
            error = storage_query_error(PermissionError("permission denied"), filesystem=part == "disk")
            updated = replace(snapshot, observed_at=datetime(2026, 11, 16, 4, 10, 0, tzinfo=UTC))
            updated = (replace(updated, disk=None, disk_error=error) if part == "disk" else
                       replace(updated, usage=tuple(StorageUsage(row.kind) for row in snapshot.usage), usage_error=error))
            with patch("acprof.tui.images.controller.read_storage", side_effect=[snapshot, updated]):
                await self.open_storage(app, pilot)
                old_state = str(app.screen.query_one(f"#storage-{part}-state", Static).content)
                await self.refresh_storage(app, pilot)
                stale_state = str(app.screen.query_one(f"#storage-{part}-state", Static).content)
                assert "上次读取" in stale_state
                assert old_state.removeprefix("本次读取 ") in stale_state
                other = "usage" if part == "disk" else "disk"
                assert "本次读取" in str(app.screen.query_one(f"#storage-{other}-state", Static).content)
                assert "75.2 GB" in self.rendered(app.screen.query_one("#storage-totals", Static))
                assert app.screen.query_one("#storage-disk-percent", Static).content == "75.0%"
                if part == "disk":
                    assert "未知" in self.rendered(app.screen.query_one("#storage-selection", Static))
                assert not self.docker.removals

    async def test_estimate_failure_preserves_other_sections_and_allows_retry(self):
        app = self.make_app()
        entered, release = Event(), Event()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-model")
            selected = set(app.images.selected_ids)

            def fail_estimate(*args, **kwargs):
                entered.set()
                if not release.wait(timeout=10):
                    raise TimeoutError("test estimate was not released")
                raise ImageManagementError("Docker 操作失败", "permission denied")

            with patch("acprof.tui.images.controller.read_storage", return_value=self.snapshot(app)):
                with patch("acprof.tui.images.controller.list_images", side_effect=fail_estimate):
                    try:
                        assert await pilot.click("#image-storage")
                        assert await asyncio.to_thread(entered.wait, 3)
                        await pilot.pause()
                        assert app.screen.query_one("#storage-disk-percent", Static).content == "75.0%"
                        assert "75.2 GB" in self.rendered(app.screen.query_one("#storage-totals", Static))
                        assert "正在读取" in self.rendered(app.screen.query_one("#storage-selection", Static))
                        assert app.screen.query_one("#storage-refresh", Button).disabled
                    finally:
                        release.set()
                    await wait_for_condition(pilot, lambda: not app.images.storage_loading,
                                             description="estimate failure", timeout=5)
                    assert "Socket 权限" in self.rendered(app.screen.query_one("#storage-status", Static))
                    assert "未知" in self.rendered(app.screen.query_one("#storage-selection", Static))
                await self.refresh_storage(app, pilot)
                assert "≈310 B" in self.rendered(app.screen.query_one("#storage-selection", Static))
                assert app.images.selected_ids == selected
                assert not self.docker.removals

    @pytest.mark.parametrize("identity", ["daemon", "context", "during_layers"])
    async def test_identity_change_drops_previous_storage_and_estimate(self, identity):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-model")
            snapshot = self.snapshot(app)
            with patch("acprof.tui.images.controller.read_storage", return_value=snapshot):
                await self.open_storage(app, pilot)
                if identity == "daemon":
                    self.docker.daemon_id = "daemon-switched"
                    await self.refresh_storage(app, pilot)
                elif identity == "during_layers":
                    def switch_after_read(*args, **kwargs):
                        current = list_images(*args, **kwargs)
                        self.docker.daemon_id = "daemon-switched-during-layers"
                        return current

                    with patch("acprof.tui.images.controller.list_images", side_effect=switch_after_read):
                        await self.refresh_storage(app, pilot)
                else:
                    error = storage_query_error(PermissionError("permission denied"))
                    changed = replace(snapshot, connection=DockerConnection(("--context", "other"), "other"),
                                      disk=None, disk_error=error,
                                      usage=tuple(StorageUsage(row.kind) for row in snapshot.usage), usage_error=error)
                    with patch("acprof.tui.images.controller.read_storage", return_value=changed):
                        await self.refresh_storage(app, pilot)
                assert "75.2 GB" not in self.rendered(app.screen.query_one("#storage-totals", Static))
                assert "310 B" not in self.rendered(app.screen.query_one("#storage-selection", Static))
                assert "上次读取" not in str(app.screen.query_one("#storage-disk-state", Static).content)
                assert "环境已改变" in self.rendered(app.screen.query_one("#storage-status", Static))

    async def test_worker_schedule_failure_restores_refresh(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.load_images(app, pilot)
            with patch.object(app.images, "execute_storage_refresh", side_effect=RuntimeError("worker unavailable")):
                await self.open_storage(app, pilot)
                assert not app.images.storage_loading
                assert not app.screen.query_one("#storage-refresh", Button).disabled
                assert "worker unavailable" in self.rendered(app.screen.query_one("#storage-details", Static))
            await pilot.press("escape")
            await pilot.pause()
            assert not app.is_busy()

    @pytest.mark.parametrize(("field", "reference"), [
        ("RepoTags", "changed:tag"), ("RepoDigests", "changed@sha256:" + "d" * 64),
    ])
    async def test_changed_selection_references_show_unknown_reclaim_without_deletion(self, field, reference):
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-model")
            selected = set(app.images.selected_ids)
            self.docker.images[FINAL][field].append(reference)
            with patch("acprof.tui.images.controller.read_storage", return_value=self.snapshot(app)):
                await self.open_storage(app, pilot)
                selection = self.rendered(app.screen.query_one("#storage-selection", Static))
                assert ("未知") in (selection)
                assert ("≈") not in (selection)
                status = self.rendered(app.screen.query_one("#storage-status", Static))
                assert ("所选镜像或引用已改变") in (status)
                assert (app.images.selected_ids) == (selected)
                assert not (self.docker.removals)
                await pilot.press("escape")
                await pilot.pause()
                app.configuration.preferences = replace(app.configuration.preferences, language="en")
                app.configuration.apply_ui_preferences()
                await pilot.pause()
                await self.open_storage(app, pilot)
                selection = self.rendered(app.screen.query_one("#storage-selection", Static))
                assert ("Unknown") in (selection)
                assert ("未知") not in (selection)
