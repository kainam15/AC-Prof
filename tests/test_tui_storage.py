"""Storage modal interaction, accounting display and measurement exclusion."""

import asyncio
import os
import tempfile
import unittest
from dataclasses import replace
from io import StringIO
from pathlib import Path
from threading import Event
from unittest.mock import patch

from rich.cells import cell_len
from rich.console import Console
from test_image_management import FINAL, DockerFixture
from textual.widgets import Button, DataTable, Static

from acprof.experiment import RunConfig
from acprof.host.image_management import (
    DiskSpace,
    DockerStorage,
    ImageManagementError,
    StorageUsage,
)
from acprof.tui.app import AcprofTui
from acprof.tui.progress import ProgressSnapshot


class TuiStorageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.docker = DockerFixture()
        for patcher in (patch("acprof.host.image_management.run_command", side_effect=self.docker.run),
                        patch.dict(os.environ, {}, clear=True),
                        patch.object(AcprofTui, "IMAGE_REFRESH_INTERVAL", 3600)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_app(self):
        return AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.directory / "tui.json")

    async def load_images(self, app, pilot):
        await pilot.pause()
        app.action_show_images()
        await pilot.pause()
        await app.workers.wait_for_complete()
        app._image_refresh_timer.pause()
        await pilot.pause()
        self.assertFalse(app._is_busy())
        # 树首次聚焦公共环境；选择最后一个模型节点用于勾选回归。
        await pilot.click("#image-view-list")
        table = app.query_one("#image-table", DataTable)
        table.move_cursor(row=table.get_row_index(FINAL))
        await pilot.pause()

    @staticmethod
    def snapshot(app):
        return DockerStorage(
            app._image_inventory.connection, app._image_inventory.daemon_id, "/var/lib/docker",
            DiskSpace(512 * 1024**3, 384 * 1024**3, 128 * 1024**3),
            tuple(StorageUsage(kind, used * 1024**3, reclaim * 1024**3) for kind, used, reclaim in (
                ("Images", 46, 18), ("Containers", 3, 1), ("Local Volumes", 12, 2), ("Build Cache", 9, 9),
            )),
        )

    @staticmethod
    def rendered(widget):
        stream = StringIO()
        Console(file=stream, width=72, color_system=None).print(widget.content)
        return stream.getvalue()

    async def open_storage(self, app, pilot):
        self.assertTrue(await pilot.click("#image-storage"))
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

    async def test_storage_values_refresh_and_failure_recovery(self):
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-model")
            snapshot = self.snapshot(app)
            updated = replace(snapshot, disk=DiskSpace(512 * 1024**3, 256 * 1024**3, 256 * 1024**3))
            with patch("acprof.tui.image_actions.read_storage", side_effect=[
                snapshot, ImageManagementError("Docker 操作失败", "permission denied"), updated,
            ]) as read:
                await self.open_storage(app, pilot)
                self.assertEqual(app.screen.query_one("#storage-root", Static).content, "/var/lib/docker")
                self.assertEqual(app.screen.query_one("#storage-disk-percent", Static).content, "75.0%")
                bar = app.screen.query_one("#storage-disk-bar")
                self.assertEqual(bar.region.height, 1, "磁盘占用条应有可见高度")
                visible = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
                self.assertIn("━", visible, "磁盘使用率需要可见的进度条")
                totals = self.rendered(app.screen.query_one("#storage-totals", Static))
                self.assertIn("70.00 GiB", totals)
                self.assertIn("30.00 GiB", totals)
                selection = self.rendered(app.screen.query_one("#storage-selection", Static))
                self.assertIn("≈310 B", selection, "所选父子镜像的共享层只能计一次")
                self.assertIn("≈128.00 GiB", selection)
                for expected in ("permission denied", ""):
                    refresh = app.screen.query_one("#storage-refresh", Button)

                    async def button_ready():
                        while refresh.has_class("-active"):
                            await refresh.wait_for_refresh()

                    # Textual 在按下动画期间忽略重复点击；等待可观察状态恢复。
                    await asyncio.wait_for(button_ready(), timeout=3)
                    self.assertTrue(await pilot.click("#storage-refresh"))
                    await pilot.pause()
                    await app.workers.wait_for_complete()
                    await pilot.pause()
                    status = self.rendered(app.screen.query_one("#storage-status", Static))
                    if expected:
                        self.assertIn(expected, status)
                        self.assertNotIn("70.00 GiB", self.rendered(app.screen.query_one("#storage-totals", Static)))
                    else:
                        self.assertNotIn("permission denied", status)
                        self.assertIn("≈256.00 GiB", self.rendered(app.screen.query_one("#storage-selection", Static)))
                self.assertEqual(read.call_count, 3)
                await pilot.press("escape")
                await pilot.pause()
                self.assertFalse(app._is_busy())
                self.assertFalse(self.docker.removals)

    async def test_modal_layout_language_keyboard_and_resize(self):
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot)
            snapshot = self.snapshot(app)
            with patch("acprof.tui.image_actions.read_storage", return_value=snapshot) as read:
                for language in ("zh", "en"):
                    app.ui_preferences = replace(app.ui_preferences, language=language)
                    app._apply_ui_preferences()
                    await pilot.resize_terminal(80, 24)
                    await pilot.pause()
                    entry = app.query_one("#image-storage", Button)
                    self.assertLessEqual(cell_len(str(entry.label)), entry.content_region.width)
                    self.assertIs(app.get_widget_at(*entry.region.center)[0], entry)
                    await self.open_storage(app, pilot)
                    title = "存储空间" if language == "zh" else "Storage space"
                    self.assertEqual(app.screen.query_one("#storage-dialog").border_title, title)
                    self.assertIn("容器" if language == "zh" else "Containers",
                                  self.rendered(app.screen.query_one("#storage-usage", Static)))
                    for width, height in ((80, 24), (120, 30), (150, 45)):
                        with self.subTest(language=language, size=(width, height)):
                            await pilot.resize_terminal(width, height)
                            await pilot.pause()
                            dialog = app.screen.query_one("#storage-dialog")
                            self.assertGreater(dialog.region.x, 0)
                            self.assertLess(dialog.region.right, width)
                            for selector in ("#storage-refresh", "#storage-close"):
                                button = app.screen.query_one(selector, Button)
                                self.assertEqual(button.region.height, 3)
                                self.assertLessEqual(button.region.bottom, height)
                                self.assertIs(app.get_widget_at(*button.region.center)[0], button)
                            content = app.screen.query_one("#storage-content")
                            content.focus()
                            await pilot.press("end")
                            await pilot.pause()
                            self.assertGreaterEqual(content.scroll_y, content.max_scroll_y)
                            visible = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
                            self.assertIn("尚未勾选镜像。" if language == "zh" else "No images selected.", visible)
                            refresh = app.screen.query_one("#storage-refresh", Button)
                            refresh.focus()
                            await pilot.press("tab")
                            self.assertIs(app.focused, app.screen.query_one("#storage-close", Button))
                            await pilot.press("shift+tab")
                            self.assertIs(app.focused, refresh)
                    await pilot.press("tab", "enter")
                    await pilot.pause()
                    self.assertEqual(len(app.screen_stack), 1)
                    self.assertFalse(app._is_busy())
                self.assertEqual(read.call_count, 2, "缩放和滚动不会额外查询 Docker")

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

            with patch("acprof.tui.image_actions.read_storage", side_effect=slow_read) as read:
                try:
                    self.assertTrue(await pilot.click("#image-storage"))
                    await pilot.pause()
                    self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                    self.assertTrue(app.screen.query_one("#storage-refresh", Button).disabled)
                    app.refresh_storage()
                    self.assertEqual(read.call_count, 1)
                    await pilot.press("escape")
                    await pilot.pause()
                    self.assertEqual(len(app.screen_stack), 1)
                    self.assertTrue(app._is_busy())
                    self.assertTrue(app.query_one("#start-run", Button).disabled)
                    with patch("acprof.tui.image_actions.list_images") as inventory:
                        app.refresh_images()
                        inventory.assert_not_called()
                finally:
                    release.set()
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertFalse(app._is_busy())
                self.assertFalse(app.query_one("#start-run", Button).disabled)

    async def test_measurement_prevents_storage_query(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            app._latest_snapshot = ProgressSnapshot(measurement_active=True)
            app._set_busy(True)
            with patch("acprof.tui.image_actions.read_storage") as read:
                app.open_storage()
                await pilot.pause()
                self.assertTrue(app.query_one("#image-storage", Button).disabled)
                self.assertEqual(len(app.screen_stack), 1)
                read.assert_not_called()
            app._latest_snapshot = ProgressSnapshot()
            app._set_busy(False)

    async def test_changed_selection_references_show_unknown_reclaim_without_deletion(self):
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-model")
            selected = set(app._selected_image_ids)
            self.docker.images[FINAL]["RepoTags"].append("changed:tag")
            with patch("acprof.tui.image_actions.read_storage", return_value=self.snapshot(app)):
                await self.open_storage(app, pilot)
                selection = self.rendered(app.screen.query_one("#storage-selection", Static))
                self.assertIn("未知", selection)
                self.assertNotIn("≈", selection)
                status = self.rendered(app.screen.query_one("#storage-status", Static))
                self.assertIn("所选镜像或引用已改变", status)
                self.assertEqual(app._selected_image_ids, selected)
                self.assertFalse(self.docker.removals)
                await pilot.press("escape")
                await pilot.pause()
                app.ui_preferences = replace(app.ui_preferences, language="en")
                app._apply_ui_preferences()
                await pilot.pause()
                await self.open_storage(app, pilot)
                selection = self.rendered(app.screen.query_one("#storage-selection", Static))
                self.assertIn("Unknown", selection)
                self.assertNotIn("未知", selection)


if __name__ == "__main__":
    unittest.main()
