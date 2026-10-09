"""镜像管理应提供可操作列表，并与采集和其它 Docker 操作互斥。"""
import asyncio
import os
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
from environment_fixtures import isolated_environment
from rich.cells import cell_len
from test_image_management import FINAL, RUNTIME, WEIGHTS, DockerFixture, dependency_images, image
from test_tui_table_resize import drag, header_offset
from textual import events
from textual.widgets import (
    Button,
    Collapsible,
    ContentSwitcher,
    DataTable,
    Input,
    Static,
    TabbedContent,
    TabPane,
    Tooltip,
    Tree,
)
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.host.image_management import ImageManagementError, ManagedImage, list_images
from acprof.tui.commands import PendingLaunch
from acprof.tui.image_actions import ImageActions
from acprof.tui.images import (
    filtered_images,
    image_created_text,
    image_display_name,
    image_metadata,
    layer_image_detail,
)
from acprof.tui.progress import ProgressSnapshot


def test_platform_is_omitted_only_when_parent_provides_matching_context():
    parent = ManagedImage(RUNTIME, ("acprof-platform-cu124:base",), 100, "", "base", platform_id="cu124")
    child = ManagedImage(WEIGHTS, ("acprof-runtime-env:opaque",), 400, "", "runtime",
                         platform_id="cu124", environment_id="env-known", profiles=("nlp-cu124",))
    assert (image_display_name(parent)) == ("PyTorch CUDA 12.4")
    assert (image_display_name(child, parent)) == ("nlp")
    for context in (None, replace(parent, platform_id="cu128"), replace(parent, kind="model")):
        assert (image_display_name(child, context)) == ("nlp · PyTorch CUDA 12.4")
    assert (image_display_name(replace(child, profiles=()))) == ("env-env-known · PyTorch CUDA 12.4")
    assert (image_display_name(replace(parent, platform_id="cpu"))) == ("PyTorch CPU")
    assert (child.profiles) == (("nlp-cu124",))
    assert (child.tags) == (("acprof-runtime-env:opaque",))

def test_pinned_python_upstream_is_named_only_with_matching_repository_digest():
    from acprof.runtime_profiles import PYTHON_BASE_IMAGE
    from acprof.tui.i18n import translate

    upstream = ManagedImage(
        "sha256:" + "e" * 64, (), 20, "", "untagged",
        repo_digests=(PYTHON_BASE_IMAGE.replace(":3.10-slim@", "@"),),
    )
    name = image_display_name(upstream)
    assert name == "Python 3.10 Slim（上游基础）"
    assert translate(name, "en") == "Python 3.10 Slim (upstream base)"
    for other in (replace(upstream, repo_digests=()),
                  replace(upstream, repo_digests=("another.example/python@" + PYTHON_BASE_IMAGE.split("@")[1],)),
                  replace(upstream, repo_digests=("docker.m.daocloud.io/library/python@sha256:" + "0" * 64,)),
                  replace(upstream, acprof=True)):
        assert image_display_name(other) == upstream.image_id[7:19]
    assert image_display_name(replace(upstream, tags=("some-other-image:latest",))) == "some-other-image"


def test_dotted_versions_and_model_names_are_preserved_without_guessing():
    item = ManagedImage(WEIGHTS, (), 400, "", "runtime", platform_id="cpu", environment_id="known",
                        profiles=("audio-cpu", "multimodal-transformers4576-cpu", "custom-v1.12.3-cpu", "custom123-cpu"))
    assert (image_display_name(item)) == ("audio / multimodal-transformers4.57.6 / custom-v1.12.3 / custom123 · PyTorch CPU")
    assert (image_display_name(replace(item, kind="model", model_id="Qwen/Qwen2.5-0.5B"))) == ("Qwen/Qwen2.5-0.5B")


@pytest.mark.parametrize(("raw", "expected"), (
    ("", "未知"),
    ("not-a-timestamp", "未知"),
    ("2026-09-13T00:00:00", "未知"),
    ("2026-09-13T08:12:42.123456789+08:00",
     datetime.fromisoformat("2026-09-13T08:12:42+08:00").astimezone().strftime("%Y-%m-%d %H:%M")),
    ("2026-09-13T00:00:00Z",
     datetime.fromisoformat("2026-09-13T00:00:00+00:00").astimezone().strftime("%Y-%m-%d %H:%M")),
))
def test_image_created_text_uses_docker_instant(raw, expected):
    assert image_created_text(raw) == expected


class TestTuiImages:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        temporary = tmp_path
        self.directory = Path(str(temporary))
        self.docker = DockerFixture()
        docker_patch = patch("acprof.host.image_management.run_command", side_effect=self.docker.run)
        docker_patch.start()
        self._request.addfinalizer(partial(docker_patch.stop))
        environment = patch.dict(os.environ, isolated_environment(), clear=True)
        environment.start()
        self._request.addfinalizer(partial(environment.stop))
        # 定时刷新单独验证；其它交互测试不依赖机器运行速度。
        interval = patch.object(AcprofTui, "IMAGE_REFRESH_INTERVAL", 3600)
        interval.start()
        self._request.addfinalizer(partial(interval.stop))

    def make_app(self):
        return AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.directory / "tui.json")

    @pytest.mark.parametrize('view', ('tree', 'list', 'layers'))
    @pytest.mark.parametrize('language', ('zh', 'en'))
    @pytest.mark.parametrize('size', ((80, 24), (120, 30), (150, 45)))
    async def test_detail_boundary_drags_in_all_views_sizes_and_languages(self, view, language, size):
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot, view="tree")
            browser = app.query_one("#image-browser")
            detail = app.query_one("#image-detail-scroll")
            before = len(self.docker.commands)
            await pilot.resize_terminal(*size)
            app.ui_preferences = replace(app.ui_preferences, language=language)
            app._apply_ui_preferences()
            await pilot.pause()
            await pilot.click("#image-view-" + view)
            await pilot.pause()
            initial = detail.size.height
            browser_height = browser.size.height
            current = detail._detail_key
            start = (browser.region.x + browser.size.width // 2, browser.region.bottom)
            await drag(pilot, app.screen, start, 0, 2, release_click=True)
            assert (detail.size.height) == (initial - 2), "向下拖动分隔线应缩小详情区"
            assert (browser.size.height) == (browser_height + 2)
            assert (app.mouse_captured) is None
            start = (start[0], browser.region.bottom)
            await drag(pilot, app.screen, start, 0, -2, release_click=True)
            assert (detail.size.height) == (initial)
            assert (browser.size.height) == (browser_height)
            assert (detail._detail_key) == (current)
            assert not (app._selected_image_ids)
            assert (detail.region.bottom) <= (app.query_one("#image-panel").content_region.bottom)
            assert (len(self.docker.commands)) == (before), "调整详情高度不能查询 Docker"

    @pytest.mark.parametrize('view, widget_id', (
        ('tree', '#image-tree'), ('list', '#image-table'), ('layers', '#image-layer-table'),
    ))
    @pytest.mark.parametrize('size, language, scroll_y', (
        ((80, 24), 'zh', 0), ((120, 30), 'en', 0), ((150, 45), 'zh', 5),
    ))
    async def test_detail_drag_preserves_browser_scroll_position(self, view, widget_id, size, language, scroll_y):
        for number in range(30):
            key = f"sha256:{number:064x}"
            self.docker.images[key] = image(
                key, [f"acprof-runtime-extra-{number:02}:env"], 200, [f"extra-{number}"])
        app = self.make_app()
        async with app.run_test(size=size) as pilot:
            await self.load_images(app, pilot, view="tree")
            app.ui_preferences = replace(app.ui_preferences, language=language)
            app._apply_ui_preferences()
            await pilot.pause()
            await pilot.click("#image-view-" + view)
            browser = app.query_one(widget_id)
            handle = app.query_one("#image-detail-resize")
            detail = app.query_one("#image-detail-scroll")
            # 先给短终端的列表留出两行可缩空间，再逐行向上拖动。
            handle.focus()
            await pilot.press("down", "down")
            browser.scroll_to(y=scroll_y, animate=False, immediate=True)
            await pilot.pause()
            assert browser.max_scroll_y > scroll_y
            assert browser.scroll_y == scroll_y
            initial_height = detail.size.height
            current = detail._detail_key
            before = len(self.docker.commands)
            x, y = handle.region.x + handle.size.width // 2, handle.region.y
            await pilot.mouse_down(handle, offset=(handle.size.width // 2, 0))
            # Pilot.hover 的 delta_y 固定为 0，无法触发真实拖动的选区自动滚动。
            previous_y = y
            for moved in (-1, -2, 0):
                target_y = y + moved
                app.post_message(events.MouseMove(
                    app.screen, x, target_y, delta_x=0, delta_y=target_y - previous_y, button=1,
                    shift=False, meta=False, ctrl=False, screen_x=x, screen_y=target_y,
                ))
                await pilot.pause()
                assert detail.size.height == initial_height - moved
                assert browser.scroll_y == scroll_y, "按住分隔条拖动不能触发浏览区自动滚动"
                assert not app.screen.get_selected_text(), "分隔条拖动不能建立文本选区"
                assert app.mouse_captured is handle
                previous_y = target_y
            await pilot.mouse_up(offset=(x, y))
            await pilot.pause()
            assert app.mouse_captured is None
            assert browser.scroll_y == scroll_y
            assert detail._detail_key == current
            assert not app._selected_image_ids
            assert len(self.docker.commands) == before

    async def test_detail_resize_clamps_restores_height_and_preserves_reading_state(self):
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot)
            browser = app.query_one("#image-browser")
            detail = app.query_one("#image-detail-scroll")
            initial = detail.size.height
            handle = app.query_one("#image-detail-resize")
            x = handle.region.x + handle.size.width // 2
            await drag(pilot, app.screen, (x, handle.region.y), 0, -handle.region.y)
            assert (browser.size.height) == (3), "拖出区域仍需保留列表表头和行"
            expanded = detail.size.height
            assert (expanded) > (initial)
            assert (app.mouse_captured) is None
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert (browser.size.height) >= (3)
            assert (detail.size.height) >= (3)
            assert (detail.region.bottom) <= (app.query_one("#image-panel").content_region.bottom)
            await pilot.resize_terminal(150, 45)
            await pilot.pause()
            assert (detail.size.height) == (expanded), "窗口恢复后还原用户设置的高度"
            await drag(pilot, app.screen, (x, handle.region.y), 0, app.size.height - 1 - handle.region.y)
            assert (detail.size.height) == (3)
            handle.focus()
            await pilot.press("home", "up", "up")
            await pilot.pause()
            assert (detail.size.height) == (initial + 2)
            await pilot.press("down")
            await pilot.pause()
            assert (detail.size.height) == (initial + 1)
            diagnostics = app.query_one("#image-diagnostics", Collapsible)
            diagnostics.collapsed = False
            await pilot.pause()
            detail.scroll_end(animate=False, immediate=True)
            await pilot.pause()
            offset = detail.scroll_y
            assert (offset) > (0)
            await drag(pilot, app.screen, (x, handle.region.y), 0, 2, release_click=True)
            assert not (diagnostics.collapsed)
            assert (detail.scroll_y) == (offset), "拖动分隔条不应把详情滚回摘要"
            height = detail.size.height
            app.refresh_images()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert (detail.size.height) == (height)
            assert not (diagnostics.collapsed)
            detail.focus()
            await pilot.press("end")
            await pilot.pause()
            assert (detail.scroll_y) == (detail.max_scroll_y), "调整高度后仍能滚动到末尾"
            assert not (self.directory.joinpath("tui.json").exists()), "高度仅在本次会话保留"

    @pytest.mark.parametrize('interruption', ('escape', 'capture_lost', 'hidden', 'measurement', 'resize'))
    async def test_detail_resize_releases_mouse_after_interruption(self, interruption):
        # 断言失败也会退出本次应用，避免隐藏页面或 busy 状态影响下一场景。
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            handle = app.query_one("#image-detail-resize")
            detail = app.query_one("#image-detail-scroll")
            initial = detail.size.height
            await pilot.mouse_down(handle, offset=(10, 0))
            assert (app.mouse_captured) is (handle)
            before = len(self.docker.commands)
            app.refresh_images()
            await pilot.pause()
            assert (len(self.docker.commands)) == (before), "拖动期间暂停扫描"
            if interruption == "escape":
                await pilot.press("escape")
            elif interruption == "capture_lost":
                handle.release_mouse()
            elif interruption == "hidden":
                app.action_show_settings()
            elif interruption == "measurement":
                app._process_kind = "run"
                app._latest_snapshot = ProgressSnapshot(measurement_active=True)
                app._set_busy(True)
            else:
                await pilot.resize_terminal(150, 45)
            await pilot.pause()
            # Pilot.pause() 末尾的布局刷新可能刚排入 Hide；等控件处理后再断言。
            await asyncio.wait_for(handle.wait_for_refresh(), timeout=3)
            assert (app.mouse_captured) is None
            await pilot.hover(offset=(60, 5))
            await pilot.mouse_up(offset=(60, 5))
            if interruption == "measurement":
                assert (handle.disabled)
                app._process_kind = ""
                app._latest_snapshot = ProgressSnapshot()
                app._set_busy(False)
            elif interruption == "hidden":
                await self.load_images(app, pilot)
            elif interruption == "resize":
                await pilot.resize_terminal(120, 30)
            await pilot.pause()
            assert (detail.size.height) == (initial)
            start = (handle.region.x + 10, handle.region.y)
            await drag(pilot, app.screen, start, 0, 1)
            assert (detail.size.height) == (initial - 1)
            handle.focus()
            await pilot.press("home")
            await pilot.pause()

    async def test_images_page_loads_automatically_only_after_opening(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            tabs = app.query_one("#main-tabs", TabbedContent)
            assert ("images-tab") in ([pane.id for pane in tabs.query(TabPane)]), "用户应能从 TUI 进入 Docker 镜像管理"
            assert not (self.docker.commands)
            field = app.query_one("#slash-command", Input)
            field.value = "/images"
            field.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert (tabs.active) == ("images-tab")
            assert (app.query_one("#image-table", DataTable).row_count) == (3)
            assert not (app.query("#image-refresh"))
            assert (app.query_one("#image-delete", Button).disabled)
            assert (self.docker.commands)

    async def test_storage_button_opens_modal_without_changing_image_selection(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-toggle")
            selected = set(app._selected_image_ids)
            buttons = app.query("#image-storage")
            assert (buttons), "镜像视图切换按钮旁应提供存储空间入口"
            button = buttons.first(Button)
            assert (str(button.label)) == ("存储空间")
            layers = app.query_one("#image-view-layers", Button)
            assert (button.region.x) > (layers.region.right)
            assert (button.region.y, button.region.height) == (layers.region.y, layers.region.height)
            assert (await pilot.click("#image-storage"))
            await pilot.pause()
            assert (app.screen.query_one("#storage-dialog").border_title) == ("存储空间")
            assert (await pilot.click("#storage-close"))
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert (len(app.screen_stack)) == (1)
            assert (app._selected_image_ids) == (selected)
            assert not (self.docker.removals)

    @pytest.mark.parametrize("language", ("zh", "en"))
    @pytest.mark.parametrize("size", ((80, 24), (120, 30), (150, 45)))
    async def test_image_creation_date_is_right_aligned_and_hidden_for_layers(self, language, size):
        app = self.make_app()
        async with app.run_test(size=size) as pilot:
            await self.load_images(app, pilot, view="list")
            app.ui_preferences = replace(app.ui_preferences, language=language)
            app._apply_ui_preferences()
            await pilot.pause()
            created = app.query_one("#image-detail-created", Static)
            title = app.query_one("#image-detail-title", Static)
            heading = app.query_one("#image-detail-heading")
            value = image_created_text(self.docker.images[FINAL]["Created"])
            expected = f"创建日期：{value}" if language == "zh" else f"Created: {value}"
            assert created.display
            assert str(created.content) == expected
            assert created.region.y == title.region.y
            assert created.region.right == heading.region.right
            assert created.region.x >= title.region.right
            before = len(self.docker.commands)
            await pilot.click("#image-view-layers")
            await pilot.pause()
            assert not created.display
            assert len(self.docker.commands) == before, "Changing the detail view must not query Docker"
            app.query_one("#image-search", Input).value = "no-such-image"
            await pilot.pause()
            assert not heading.display, "Empty details must not reserve a blank header row"
            assert not created.display

    async def test_image_creation_date_shows_unknown_instead_of_invalid_raw_value(self):
        self.docker.images[FINAL]["Created"] = "bad-date"
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.load_images(app, pilot, view="list")
            assert str(app.query_one("#image-detail-created", Static).content) == "创建日期：未知"

    async def test_detail_summary_separates_packages_and_diagnostics_with_interactive_folds(self):
        dependency_images(self.docker, profile="nlp-cu128")
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot, view="list")
            inventory = app._image_inventory
            packages = (("transformers", "4.57.6"), *(
                (f"example-package-{index:02}", "1.0") for index in range(43)))
            app._show_images(replace(inventory, images=tuple(
                replace(item, python_dependencies=packages) if item.image_id == WEIGHTS else item
                for item in inventory.images)))
            table = app.query_one("#image-table", DataTable)
            table.move_cursor(row=table.get_row_index(WEIGHTS), animate=False)
            await pilot.pause()
            summary = str(app.query_one("#image-detail", Static).content)
            assert ("sha256:") not in (summary), "完整 SHA 应只出现在默认折叠的诊断信息中"
            assert ("transformers==") not in (summary), "包清单不应挤占摘要"
            assert ("history") not in (summary)
            assert ("完整大小") in (summary)
            assert ("预计可释放") in (summary)
            dependencies = app.query_one("#image-dependencies", Collapsible)
            metadata = app.query_one("#image-metadata", Collapsible)
            diagnostics = app.query_one("#image-diagnostics", Collapsible)
            assert ("44") in (dependencies.title)
            before = len(self.docker.commands)
            app.query_one("#image-detail-scroll").focus()
            await pilot.press("tab")
            await pilot.pause()
            assert (app.focused) is (dependencies.query_one("CollapsibleTitle"))
            for group, content_id in ((dependencies, "image-dependency-detail"),
                                       (metadata, "image-metadata-detail"),
                                       (diagnostics, "image-diagnostic-detail")):
                assert (group.collapsed)
                content = app.query_one("#" + content_id, Static)
                assert (content.region.height) == (0)
                title = group.query_one("CollapsibleTitle")
                title.scroll_visible(animate=False, immediate=True)
                await pilot.pause()
                assert (await pilot.click(title))
                await pilot.pause()
                assert not (group.collapsed)
                assert (content.region.height) > (0)
                if group is dependencies:
                    assert ("transformers==4.57.6") in (str(content.content))
                    assert ("example-package-42==1.0") in (str(content.content))
                    assert ("依赖来源") not in (str(content.content))
                    scroll = app.query_one("#image-detail-scroll")
                    scroll.focus()
                    scroll.scroll_end(animate=False, immediate=True)
                    await pilot.pause()
                    screen_text = ""
                    for _ in range(6):
                        screen_text += "\n".join(strip.text for strip in app.screen._compositor.render_strips())
                        await pilot.press("up")
                        await pilot.pause()
                    assert ("example-package-42==1.0") in (screen_text)
                elif group is diagnostics:
                    assert (WEIGHTS) in (str(content.content))
                    assert ("依赖来源") in (str(content.content))
                    assert ("history") in (str(content.content))
                title.focus()
                await pilot.press("enter")
                await pilot.pause()
                assert (group.collapsed)
            assert (len(self.docker.commands)) == (before), "折叠交互不能扫描 Docker"

    async def test_detail_folds_preserve_state_on_resize_and_language_but_reset_for_another_image(self):
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot)
            assert ("sha256:") not in (str(app.query_one("#image-detail", Static).content))
            diagnostics = app.query_one("#image-diagnostics", Collapsible)
            title = diagnostics.query_one("CollapsibleTitle")
            title.focus()
            await pilot.press("enter")
            await pilot.pause()
            for size in ((80, 24), (120, 30), (150, 45)):
                await pilot.resize_terminal(*size)
                for language in ("zh", "en"):
                    app.ui_preferences = replace(app.ui_preferences, language=language)
                    app._apply_ui_preferences()
                    await pilot.pause()
                    assert not (diagnostics.collapsed)
                    assert (diagnostics.title) == ("诊断信息" if language == "zh" else "Diagnostics")
                    title.focus()
                    await pilot.press("enter")
                    await pilot.pause()
                    assert (diagnostics.collapsed)
                    await pilot.press("enter")
                    await pilot.pause()
                    assert not (diagnostics.collapsed)
            table = app.query_one("#image-table", DataTable)
            table.focus()
            await pilot.press("space")
            await pilot.pause()
            assert (app._selected_image_ids) == ({FINAL})
            assert not (diagnostics.collapsed), "勾选同一镜像时应保留展开状态"
            table.move_cursor(row=table.get_row_index(WEIGHTS), animate=False)
            await pilot.pause()
            assert (diagnostics.collapsed)
            assert (app.query_one("#image-detail-scroll").scroll_y) == (0)
            assert (WEIGHTS) in (str(app.query_one("#image-diagnostic-detail", Static).content))
            await pilot.click("#image-view-layers")
            await pilot.pause()
            assert not (app.query_one("#image-dependencies").display)
            assert (diagnostics.collapsed)
            assert ("Chain ID") in (str(app.query_one("#image-diagnostic-detail", Static).content))
            app.query_one("#image-search", Input).value = "no-such-image"
            await pilot.pause()
            for selector in ("#image-dependencies", "#image-metadata", "#image-diagnostics"):
                assert not (app.query_one(selector).display)
            assert ("sha256:") not in (str(app.query_one("#image-detail", Static).content))

    @pytest.mark.parametrize('language', ('zh', 'en'))
    @pytest.mark.parametrize('size', ((80, 24), (120, 30), (150, 45)))
    async def test_tree_dependencies_stay_in_details_and_search_after_resize_and_language(self, language, size):
        dependency_images(self.docker, profile="nlp-cu128")
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot, view="tree", expand_tree=True)
            before = len(self.docker.commands)
            await pilot.resize_terminal(*size)
            app.ui_preferences = replace(app.ui_preferences, language=language)
            app._apply_ui_preferences()
            await pilot.pause()
            tree = app.query_one("#image-tree", Tree)
            platform = tree.root.children[0]
            runtime = platform.children[0]
            assert ("torch==") not in (tree.render_line(platform.line).text)
            assert ("transformers==") not in (tree.render_line(runtime.line).text)
            tree.move_cursor(platform, animate=False)
            await pilot.pause()
            assert ("torch==2.11.0+cu128") in (str(app.query_one("#image-dependency-detail", Static).content))
            assert (app.query_one("#image-dependencies", Collapsible).collapsed)
            tree.move_cursor(runtime, animate=False)
            await pilot.pause()
            detail = str(app.query_one("#image-dependency-detail", Static).content)
            assert ("transformers==4.57.6") in (detail)
            assert ("sentence-transformers==5.1.2") in (detail)
            assert ("torch==") not in (detail)
            assert ("sha256:") not in (detail)
            assert ("依赖来源") not in (detail)
            panel = app.query_one("#image-detail-scroll")
            for group in panel.query(Collapsible):
                title = group.query_one("CollapsibleTitle")
                assert (title.region.bottom) <= (panel.region.bottom)
                assert (app.screen.get_widget_at(*title.region.offset)[0]) is (title)
            weights = runtime.children[0]
            tree.move_cursor(weights, animate=False)
            await pilot.pause()
            assert ("无新增包") not in (tree.render_line(weights.line).text)
            assert ("No new packages") not in (tree.render_line(weights.line).text)
            detail = str(app.query_one("#image-dependency-detail", Static).content)
            assert ("无新增包" if language == "zh" else "No new packages") in (detail)
            assert ("transformers==") not in (detail)
            app.query_one("#image-search", Input).value = "transformers==4.57.6"
            await pilot.pause()
            assert ([item.image_id for item in app._visible_images]) == ([WEIGHTS])
            tree = app.query_one("#image-tree", Tree)
            assert (tree.root.children[0].data.image_id) == (RUNTIME)
            assert (tree.root.children[0].children[0].data.image_id) == (WEIGHTS)
            assert (len(self.docker.commands)) == (before), "切换与搜索不能重新扫描 Docker"

    async def test_historical_dependencies_load_only_when_expanded_and_preserve_detail(self):
        dependency_images(self.docker, profile="nlp-cu128")
        self.docker.images[RUNTIME]["Config"]["Labels"]["org.acprof.platform-build-fingerprint"] = "historical-key"
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot, view="list")
            table = app.query_one("#image-table", DataTable)
            table.move_cursor(row=table.get_row_index(RUNTIME), animate=False)
            await pilot.pause()
            details = app.query_one("#image-dependencies", Collapsible)
            assert details.collapsed

            def resolved(inventory, image_id):
                images = tuple(replace(item, python_dependencies=(("torch", "old-version"),),
                                       dependency_source="historical-manifest", dependency_scope="full")
                               if item.image_id == image_id else item for item in inventory.images)
                return replace(inventory, images=images)

            with patch("acprof.host.image_dependencies.resolve_image_dependencies",
                       side_effect=resolved) as resolver:
                assert resolver.call_count == 0
                assert "未知" in str(details.query_one("CollapsibleTitle").render())
                details.collapsed = False
                await app.workers.wait_for_complete()
                await pilot.pause()
                assert resolver.call_count == 1
                assert not details.collapsed, "后台加载不应收起用户打开的清单"
                text = str(app.query_one("#image-dependency-detail", Static).content)
                assert "torch==old-version" in text
                assert "原始构建清单" in text
                assert "历史构建" in str(details.query_one("CollapsibleTitle").render())
                app._show_image_detail()
                await pilot.pause()
                assert resolver.call_count == 1

    async def load_images(self, app, pilot, *, view="list", expand_tree=False):
        await pilot.pause()
        app.action_show_images()
        await pilot.pause()
        await app.workers.wait_for_complete()
        app._image_refresh_timer.pause()
        await pilot.pause()
        assert not (app._is_busy())
        assert (app.query_one("#main-tabs", TabbedContent).active) == ("images-tab")
        if expand_tree:
            app.query_one("#image-tree", Tree).root.expand_all()
            await pilot.pause()
        if view == "list" and app.query("#image-view-list"):
            await pilot.click("#image-view-list")
            table = app.query_one("#image-table", DataTable)
            if FINAL in table.rows:
                table.move_cursor(row=table.get_row_index(FINAL))
            await pilot.pause()

    def marker_offset(self, widget, row):
        # 从实际渲染字符定位，避免用控件的点击元数据验证自身。
        text = widget.render_line(row).text
        index = next(index for index, char in enumerate(text) if char in "□☑◩✓—")
        return cell_len(text[:index]), row

    @pytest.mark.parametrize("view", ("tree", "list"))
    async def test_branch_selection_and_half_selection_never_promote_ancestors(self, view):
        sibling = "sha256:" + "d" * 64
        self.docker.images[sibling] = image(sibling, ["acprof-nlp-sibling:code"], 420,
                                            ["os", "deps", "weights", "sibling"])
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot, view=view, expand_tree=view == "tree")
            widget = app.query_one("#image-tree" if view == "tree" else "#image-table")

            async def toggle(image_id):
                if view == "tree":
                    def find(node):
                        if node.data and node.data.image_id == image_id:
                            return node
                        return next((found for child in node.children if (found := find(child))), None)
                    widget.move_cursor(find(widget.root), animate=False)
                else:
                    widget.move_cursor(row=widget.get_row_index(image_id))
                widget.focus()
                await pilot.pause()
                await pilot.press("space")
                await pilot.pause()

            before = len(self.docker.commands)
            await toggle(WEIGHTS)
            assert app._selected_image_ids == {WEIGHTS, FINAL, sibling}
            assert "◩" in app.query_one("#image-tree", Tree).root.children[0].label.plain
            await toggle(FINAL)
            assert app._selected_image_ids == {sibling}, "取消下层同时保留其上层"
            table = app.query_one("#image-table", DataTable)
            assert table.get_cell(WEIGHTS, "selected").plain == "◩"
            await toggle(FINAL)
            assert app._selected_image_ids == {FINAL, sibling}, "补齐子节点也不能自动删除父镜像"
            await toggle(RUNTIME)
            assert app._selected_image_ids == {RUNTIME, WEIGHTS, FINAL, sibling}
            await toggle(RUNTIME)
            assert not app._selected_image_ids
            assert len(self.docker.commands) == before

    @pytest.mark.parametrize("language", ("zh", "en"))
    async def test_collapsed_filtered_branch_includes_hidden_images_and_inferred_confirmation(self, language):
        app = self.make_app()
        app.ui_preferences = replace(app.ui_preferences, language=language)
        async with app.run_test(size=(80, 24)) as pilot:
            await self.load_images(app, pilot, view="tree")
            tree = app.query_one("#image-tree", Tree)
            tree.focus()
            await pilot.press("left")
            await pilot.pause()
            assert not tree.root.children[0].is_expanded
            app.query_one("#image-search", Input).value = "acprof-runtime"
            await pilot.pause()
            tree.focus()
            await pilot.press("space")
            await pilot.pause()
            assert app._selected_image_ids == {RUNTIME, WEIGHTS, FINAL}
            assert ("筛选外 2" if language == "zh" else "2 hidden") in str(app.query_one("#image-status", Static).content)
            assert not tree.root.children[0].is_expanded
            await pilot.click("#image-delete")
            await pilot.pause()
            content = str(app.screen.query_one("#image-confirm-text", Static).content)
            assert all(image_id in content for image_id in (RUNTIME, WEIGHTS, FINAL))
            assert ("筛选外 2" if language == "zh" else "2 filtered-out") in content
            assert "≈" in content and ("未确认 FROM" if language == "zh" else "FROM unconfirmed") in content
            await pilot.press("escape")
            assert not self.docker.removals

    @pytest.mark.parametrize("container_image", (FINAL, RUNTIME), ids=("leaf", "parent"))
    async def test_container_in_branch_preserves_ancestors_and_selects_free_siblings(self, container_image):
        sibling = "sha256:" + "d" * 64
        self.docker.images[sibling] = image(sibling, ["acprof-nlp-sibling:code"], 420,
                                            ["os", "deps", "other"])
        self.docker.containers["used"] = dict(Image=container_image, Name="/kept", State=dict(Status="exited"))
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot, view="tree")
            tree = app.query_one("#image-tree", Tree)
            tree.focus()
            await pilot.click(tree, offset=self.marker_offset(tree, 0))
            await pilot.pause()
            assert app._selected_image_ids == ({sibling} if container_image == FINAL else {WEIGHTS, FINAL, sibling})
            assert "◩" in tree.root.children[0].label.plain
            assert any("容器" in notification.message for notification in app._notifications)
            await pilot.press("space")
            await pilot.pause()
            assert not app._selected_image_ids

    @pytest.mark.parametrize("change", ("tag", "container", "new_child"))
    async def test_refresh_invalidates_ancestor_selection_without_selecting_new_images(self, change):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot, view="tree")
            tree = app.query_one("#image-tree", Tree)
            tree.focus()
            await pilot.press("space")
            await pilot.pause()
            assert app._selected_image_ids == {RUNTIME, WEIGHTS, FINAL}
            if change == "tag":
                self.docker.images[FINAL]["RepoTags"].append("extra:tag")
            elif change == "container":
                self.docker.containers["used"] = dict(Image=FINAL, Name="/kept", State=dict(Status="exited"))
            else:
                extra = "sha256:" + "d" * 64
                self.docker.images[extra] = image(extra, ["acprof-nlp-new:code"], 420,
                                                  ["os", "deps", "weights", "other"])
            app.refresh_images()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app._selected_image_ids == ({FINAL} if change == "new_child" else set())

    def add_tree_branches(self):
        for key, name, layers in (
            ("d", "alpha/model", ["os", "deps", "alpha"]),
            ("e", "alpha/model", ["os", "deps", "alpha", "code"]),
            ("f", "zulu/model", ["os", "deps", "zulu"]),
        ):
            image_id = "sha256:" + key * 64
            self.docker.images[image_id] = image(
                image_id, [f"acprof-weights-audio-{key}:test"], 400, layers, name)

    def assert_tree_path(self, app, tree, color, expected):
        # 检查最终屏幕的线条颜色，也能发现移动光标后祖先行没有重绘的问题。
        screen = app.screen._compositor.render_strips()
        region = tree.scrollable_content_region
        scroll_x, scroll_y = tree.scroll_offset
        highlighted = set()
        for y in range(region.height):
            x = 0
            for segment in screen[region.y + y].crop(region.x, region.right):
                for char in segment.text:
                    if char in "│├└─┃┣┗━" and segment.style.color == color:
                        highlighted.add((x + scroll_x, y + scroll_y))
                    x += cell_len(char)
        visible_expected = {(x, y) for x, y in expected
                            if scroll_x <= x < scroll_x + region.width
                            and scroll_y <= y < scroll_y + region.height}
        assert (highlighted) == (visible_expected)

    async def test_tree_path_follows_click_keyboard_and_collapse_without_highlighting_siblings(self):
        self.add_tree_branches()
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot, view="tree", expand_tree=True)
            tree = app.query_one("#image-tree", Tree)
            before = len(self.docker.commands)
            await pilot.click(tree, offset=(14, 4))
            await pilot.pause()
            assert (tree.cursor_node.data.image_id) == (FINAL)
            color = tree.get_component_rich_style("tree--cursor").bgcolor
            final_path = {(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (4, 4), (5, 4), (6, 4)}
            self.assert_tree_path(app, tree, color, final_path)
            await pilot.hover(tree, offset=(14, 1))
            await pilot.pause()
            self.assert_tree_path(app, tree, color, final_path)
            await pilot.press("down")
            await pilot.pause()
            self.assert_tree_path(app, tree, color, {(0, y) for y in range(1, 6)} | {(1, 5), (2, 5)})
            await pilot.press("up", "left")
            await pilot.pause()
            assert (tree.cursor_node.data.image_id) == (WEIGHTS)
            parent_path = {(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)}
            self.assert_tree_path(app, tree, color, parent_path)
            await pilot.press("left")
            await pilot.pause()
            assert not (tree.cursor_node.is_expanded)
            self.assert_tree_path(app, tree, color, parent_path)
            await pilot.press("left")
            await pilot.pause()
            self.assert_tree_path(app, tree, color, set())
            assert (app._selected_image_ids) == (set())
            assert (len(self.docker.commands)) == (before)

    @pytest.mark.parametrize('language,theme', (('zh', 'acprof-dark'), ('en', 'acprof-light')))
    @pytest.mark.parametrize('size', ((80, 24), (120, 30), (150, 45)))
    async def test_tree_path_survives_language_resize_scroll_blur_and_search(self, language, theme, size):
        self.add_tree_branches()
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot, view="tree", expand_tree=True)
            before = len(self.docker.commands)
            await pilot.resize_terminal(*size)
            app.ui_preferences = replace(app.ui_preferences, language=language, theme=theme)
            app._apply_ui_preferences()
            await pilot.pause()
            tree = app.query_one("#image-tree", Tree)
            target = tree.root.children[0].children[1].children[0]
            tree.move_cursor(target, animate=False)
            tree.focus()
            await pilot.pause()
            color = tree.get_component_rich_style("tree--cursor").bgcolor
            expected = {(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (4, 4), (5, 4), (6, 4)}
            self.assert_tree_path(app, tree, color, expected)
            app.query_one("#image-search", Input).focus()
            await pilot.pause()
            self.assert_tree_path(app, tree, color, expected)
            tree.styles.height = 3
            target.set_label(target.label.copy().append(" extra" * 30))
            # set_label 仅重绘行；按新标签重算虚拟宽度后才能横向滚动。
            tree._invalidate()
            await pilot.pause()
            tree.scroll_to(x=2, y=2, animate=False, force=True)
            await pilot.pause()
            assert (tuple(tree.scroll_offset)) == ((2, 2))
            self.assert_tree_path(app, tree, color, expected)
            tree.styles.height = "1fr"
            search = app.query_one("#image-search", Input)
            search.value = "code"
            await pilot.pause()
            self.assert_tree_path(app, tree, color, {(0, 1), (1, 1), (2, 1), (4, 2), (5, 2), (6, 2)})
            search.value = "no-such-image"
            await pilot.pause()
            self.assert_tree_path(app, tree, color, set())
            assert (len(self.docker.commands)) == (before)

    @pytest.mark.parametrize('view', ('tree', 'list'))
    async def test_image_row_clicks_only_focus_and_show_details(self, view):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot, view=view, expand_tree=view == "tree")
            widget = app.query_one("#image-tree" if view == "tree" else "#image-table")
            before = len(self.docker.commands)
            for image_id in (RUNTIME, FINAL):
                row = (0 if image_id == RUNTIME else 2) if view == "tree" else widget.get_row_index(image_id) + 1
                marker_x, _ = self.marker_offset(widget, row)
                # 名称、列内空白和数值重复点击也不能勾选或取消。
                for x in (marker_x + 3, marker_x + 3, marker_x + 30, widget.size.width - 20):
                    await pilot.click(widget, offset=(x, row))
                    await pilot.pause()
                    assert (app._current_image().image_id) == (image_id)
                    assert (image_id) in (str(app.query_one("#image-diagnostic-detail", Static).content))
                    assert (app._selected_image_ids) == (set())
            widget.focus()
            await pilot.press("space")
            await pilot.pause()
            assert (app._selected_image_ids) == ({FINAL})
            marker_x, row = self.marker_offset(widget, row)
            await pilot.click(widget, offset=(marker_x + 3, row), times=2)
            await pilot.pause()
            assert (app._selected_image_ids) == ({FINAL})
            assert (len(self.docker.commands)) == (before)

    @pytest.mark.parametrize('view', ('tree', 'list'))
    @pytest.mark.parametrize('language', ('zh', 'en'))
    @pytest.mark.parametrize('size', ((80, 24), (120, 30), (150, 45)))
    async def test_checkbox_and_adjacent_padding_toggle_once_after_language_and_resize(self, view, language, size):
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot, view="tree", expand_tree=True)
            before = len(self.docker.commands)
            await pilot.resize_terminal(*size)
            app.ui_preferences = replace(app.ui_preferences, language=language)
            app._apply_ui_preferences()
            await pilot.pause()
            await pilot.click("#image-view-" + view)
            await pilot.pause()
            widget = app.query_one("#image-tree" if view == "tree" else "#image-table")
            for image_id in (RUNTIME, FINAL):
                if view == "tree":
                    node = widget.root.children[0]
                    if image_id == FINAL:
                        node = node.children[0].children[0]
                    widget.move_cursor(node, animate=False)
                else:
                    widget.move_cursor(row=widget.get_row_index(image_id), animate=False)
                await pilot.pause()
                row = (node.line if view == "tree" else widget.cursor_row + 1) - int(widget.scroll_y)
                for delta in (-1, 0, 1):
                    for selected in (True, False):
                        marker_x, _ = self.marker_offset(widget, row)
                        assert (await pilot.click(widget, offset=(marker_x + delta, row)))
                        await pilot.pause()
                        expected = {RUNTIME, WEIGHTS, FINAL} if image_id == RUNTIME else {FINAL}
                        assert (app._selected_image_ids) == (expected if selected else set())
                        assert ("☑" if selected else "□") in (widget.render_line(row).text)
                        if view == "tree":
                            assert (widget.root.children[0].is_expanded), "勾选父镜像不能同时折叠子树"
            assert (len(self.docker.commands)) == (before)

    async def test_tree_arrow_clicks_only_fold_and_checkbox_click_selects_new_row(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot, view="tree", expand_tree=True)
            tree = app.query_one("#image-tree", Tree)
            for expanded in (False, True):
                await pilot.click(tree, offset=(0, 0))
                await pilot.pause()
                assert (tree.root.children[0].is_expanded) == (expanded)
                assert (app._selected_image_ids) == (set())
            for view in ("tree", "list"):
                await pilot.click("#image-view-" + view)
                await pilot.pause()
                widget = app.query_one("#image-tree" if view == "tree" else "#image-table")
                for image_id in (FINAL, RUNTIME):
                    row = (2 if image_id == FINAL else 0) if view == "tree" else widget.get_row_index(image_id) + 1
                    await pilot.click(widget, offset=self.marker_offset(widget, row))
                    await pilot.pause()
                    assert (app._current_image().image_id) == (image_id)
                    assert (app._selected_image_ids) == ({FINAL} if image_id == FINAL else {FINAL, WEIGHTS, RUNTIME})
                await pilot.click("#image-clear")
                await pilot.pause()

    async def test_default_tree_and_layer_view_preserve_image_selection_without_queries(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot, view="tree")
            assert (app.query("#image-tree")), "默认视图应展示可折叠的真实镜像树"
            tree = app.query_one("#image-tree", Tree)
            assert (app.query_one("#image-browser", ContentSwitcher).current) == ("image-tree-view")
            runtime = tree.root.children[0]
            assert (runtime.data.image_id) == (RUNTIME)
            assert (runtime.children[0].children[0].data.image_id) == (FINAL)
            assert not runtime.is_expanded, "首次打开只显示顶层镜像"
            assert not runtime.children[0].is_expanded
            assert tree.last_line == 0
            tree.move_cursor(runtime)
            tree.focus()
            await pilot.pause()
            await pilot.press("right")
            await pilot.pause()
            assert runtime.is_expanded
            assert not runtime.children[0].is_expanded, "展开上层不能自动展开全部下层"
            assert tree.last_line == 1
            await pilot.press("down", "right", "down", "space")
            await pilot.pause()
            assert (app._selected_image_ids) == ({FINAL})
            detail = str(app.query_one("#image-detail", Static).content)
            assert ("继承路径") in (str(app.query_one("#image-metadata-detail", Static).content))
            assert ("10 B") in (detail)
            before = len(self.docker.commands)
            await pilot.click("#image-view-layers")
            await pilot.pause()
            layers = app.query_one("#image-layer-table", DataTable)
            assert (layers.row_count) == (4)
            assert ("3") in (str(layers.get_row_at(0)))
            assert (app.query_one("#image-toggle", Button).disabled)
            assert ("acprof-runtime-audio") in (str(app.query_one("#image-metadata-detail", Static).content))
            await pilot.click("#image-view-list")
            await pilot.pause()
            assert (app._selected_image_ids) == ({FINAL})
            table = app.query_one("#image-table", DataTable)
            table.move_cursor(row=table.get_row_index(WEIGHTS))
            await pilot.pause()
            await pilot.click("#image-view-tree")
            await pilot.pause()
            assert (tree.cursor_node.data.image_id) == (WEIGHTS)
            assert (len(self.docker.commands)) == (before)

    async def test_default_image_tree_hides_external_parent_but_all_scope_shows_full_lineage(self):
        upstream = "sha256:" + "e" * 64
        self.docker.images[upstream] = image(
            upstream, ["docker.m.daocloud.io/library/python:3.10-slim"], 20, ["os"])
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot, view="tree", expand_tree=True)
            tree = app.query_one("#image-tree", Tree)
            assert [node.data.image_id for node in tree.root.children] == [RUNTIME]
            assert [node.data.image_id for node in tree.root.children[0].children] == [WEIGHTS]
            assert "≈" not in tree.root.children[0].label.plain
            assert upstream not in {item.image_id for item in app._visible_images}
            inventory = app._image_inventory
            platform = next(item for item in inventory.images if item.image_id == RUNTIME)
            assert platform.parent_id == upstream
            assert platform.parent_source == "layer-prefix"
            assert "docker.m.daocloud.io/library/python" in str(image_metadata(platform, inventory))

            before = len(self.docker.commands)
            app.query_one("#image-scope").value = "all"
            await pilot.pause()
            assert [node.data.image_id for node in tree.root.children] == [upstream]
            assert tree.root.children[0].children[0].data.image_id == RUNTIME
            assert "≈" in tree.root.children[0].children[0].label.plain

            app.query_one("#image-scope").value = "acprof"
            await pilot.pause()
            assert [node.data.image_id for node in tree.root.children] == [RUNTIME]
            tree.focus()
            await pilot.press("space")
            await pilot.pause()
            assert app._selected_image_ids == {RUNTIME, WEIGHTS, FINAL}
            assert upstream not in app._selected_image_ids
            assert len(self.docker.commands) == before, "切换筛选和勾选不应重新扫描 Docker"

    async def test_full_tree_names_digest_verified_untagged_python_upstream(self):
        from acprof.runtime_profiles import PYTHON_BASE_IMAGE
        from acprof.tui.images import image_diagnostics

        upstream = "sha256:" + "e" * 64
        digest = PYTHON_BASE_IMAGE.replace(":3.10-slim@", "@")
        upstream_image = image(upstream, [], 20, ["os"])
        upstream_image["RepoDigests"] = [digest]
        self.docker.images[upstream] = upstream_image
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot, view="tree", expand_tree=True)
            tree = app.query_one("#image-tree", Tree)
            assert [node.data.image_id for node in tree.root.children] == [RUNTIME]

            before = len(self.docker.commands)
            app.query_one("#image-scope").value = "all"
            await pilot.pause()
            root = tree.root.children[0]
            assert root.data.image_id == upstream
            assert "Python 3.10 Slim（上游基础）" in root.label.plain
            assert root.children[0].data.image_id == RUNTIME
            inventory = app._image_inventory
            assert digest in str(image_diagnostics(root.data, inventory))
            assert upstream in str(image_diagnostics(root.data, inventory))
            assert "Python 3.10 Slim（上游基础）" in str(image_metadata(root.data, inventory))

            app.ui_preferences = replace(app.ui_preferences, language="en")
            app._apply_ui_preferences()
            await pilot.pause()
            assert "Python 3.10 Slim (upstream base)" in tree.root.children[0].label.plain
            await pilot.click("#image-view-list")
            await pilot.pause()
            table = app.query_one("#image-table", DataTable)
            assert "Python 3.10 Slim (upstream base)" in table.get_cell(upstream, "name").plain
            assert "Python 3.10 Slim (upstream base)" in table.get_cell(RUNTIME, "parent").plain
            assert len(self.docker.commands) == before, "命名及筛选不能额外调用 Docker"


    async def test_tree_search_keeps_ancestors_and_language_resize_keeps_collapse(self):
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot, view="tree", expand_tree=True)
            assert (app.query("#image-tree")), "搜索镜像时应保留祖先路径"
            tree = app.query_one("#image-tree", Tree)
            app.query_one("#image-search", Input).value = "code"
            await pilot.pause()
            assert (len(app._visible_images)) == (1)
            assert (tree.root.children[0].data.image_id) == (RUNTIME)
            assert (tree.root.children[0].children[0].children[0].data.image_id) == (FINAL)
            tree.root.children[0].collapse()
            for size in ((80, 24), (120, 30)):
                await pilot.resize_terminal(*size)
                app.ui_preferences = replace(app.ui_preferences, language="en")
                app._apply_ui_preferences()
                await pilot.pause()
                assert not (tree.root.children[0].is_expanded)
                for selector in ("#image-view-tree", "#image-view-list", "#image-view-layers", "#image-delete"):
                    button = app.query_one(selector, Button)
                    assert (button.region.height) > (0)
                    assert (button.region.right) <= (size[0]), selector

    @pytest.mark.parametrize('language', ('zh', 'en'))
    @pytest.mark.parametrize('size', ((80, 24), (120, 30), (150, 45)))
    async def test_tree_platform_names_and_dotted_versions_survive_language_and_resize(self, language, size):
        from acprof.runtime_profiles import ENVIRONMENTS, environment_id

        root = Path(__file__).resolve().parents[1]
        self.docker.images[RUNTIME] = image(
            RUNTIME, ["acprof-platform-cu128:base"], 100, ["os", "deps"],
            labels={"org.acprof.image-kind": "platform", "org.acprof.platform": "cu128"})
        for image_id, profile, layer in ((WEIGHTS, "moss-transformers560", "weights"),
                                         (FINAL, "multimodal-transformers4576", "code")):
            self.docker.images[image_id] = image(
                image_id, ["acprof-runtime-env:" + profile], 400, ["os", "deps", layer],
                labels={"org.acprof.image-kind": "environment", "org.acprof.platform": "cu128",
                        "org.acprof.environment": environment_id(ENVIRONMENTS[profile], root)})
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot, view="tree", expand_tree=True)
            before = len(self.docker.commands)
            await pilot.resize_terminal(*size)
            app.ui_preferences = replace(app.ui_preferences, language=language)
            app._apply_ui_preferences()
            await pilot.pause()
            tree = app.query_one("#image-tree", Tree)
            platform = tree.root.children[0]
            assert (platform.label.plain.split("  ")[1]) == ("PyTorch CUDA 12.8")
            children = {node.data.image_id: node for node in platform.children}
            for image_id, expected in ((WEIGHTS, "moss-transformers5.6.0"),
                                       (FINAL, "multimodal-transformers4.57.6")):
                node = children[image_id]
                assert (expected) in (node.label.plain)
                assert ("cu128") not in (node.label.plain)
                assert ("CUDA") not in (node.label.plain)
                tree.move_cursor(node, animate=False)
                await pilot.pause()
                row = node.line - int(tree.scroll_y)
                assert (expected) in (tree.render_line(row).text)
            inventory = app._image_inventory
            runtime = next(item for item in inventory.images if item.image_id == WEIGHTS)
            assert ("PyTorch CUDA 12.8 › moss-transformers5.6.0") in (str(image_metadata(runtime, inventory)))
            assert ("moss-transformers5.6.0 · PyTorch CUDA 12.8") in (str(layer_image_detail(inventory.layers[0], inventory)))
            assert ({item.image_id for item in filtered_images(inventory, "CUDA 12.8", "all")}) == ({RUNTIME, WEIGHTS, FINAL})
            assert ([item.image_id for item in filtered_images(inventory, "5.6.0", "all")]) == ([WEIGHTS])
            assert ([item.image_id for item in filtered_images(inventory, "moss-transformers560", "all")]) == ([WEIGHTS])
            await pilot.click("#image-view-list")
            await pilot.pause()
            table = app.query_one("#image-table", DataTable)
            assert ("moss-transformers5.6.0 · PyTorch CUDA 12.8") in (table.get_cell(WEIGHTS, "name").plain)
            assert (table.get_cell(WEIGHTS, "parent").plain) == ("PyTorch CUDA 12.8")
            assert (table.get_cell(WEIGHTS, "tag").plain) == ("moss-transformers560")
            assert (len(self.docker.commands)) == (before)

    async def test_list_header_sorts_numeric_bytes_and_preserves_selection(self):
        for image_id, size in ((RUNTIME, 90), (WEIGHTS, 100), (FINAL, 1000)):
            self.docker.images[image_id]["Size"] = size
        self.docker.layer_sizes.update(deps=70, weights=10, code=900)
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot)
            table = app.query_one("#image-table", DataTable)
            table.focus()
            await pilot.pause()
            await pilot.press("space")
            await pilot.pause()
            before = len(self.docker.commands)
            offset = sum(column.get_render_width(table) for column in table.ordered_columns[:2]) + 1
            for expected in ((RUNTIME, WEIGHTS, FINAL), (FINAL, WEIGHTS, RUNTIME)):
                assert (await pilot.click("#image-table", offset=(offset, 0)))
                await pilot.pause()
                assert (tuple(item.image_id for item in app._visible_images)) == (expected)
                assert (app._current_image().image_id) == (FINAL)
                assert (app._selected_image_ids) == ({FINAL})
            assert (len(self.docker.commands)) == (before)

    async def test_header_drag_changes_width_without_sorting_or_selecting(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            table = app.query_one("#image-table", DataTable)
            before = table.columns["name"].width
            order = tuple(item.image_id for item in app._visible_images)
            focused = app._current_image().image_id
            commands = len(self.docker.commands)
            boundary = sum(column.get_render_width(table) for column in table.ordered_columns[:2]) - 1
            await pilot.mouse_down(table, offset=(boundary, 0))
            await pilot.hover(table, offset=(boundary - 12, 0))
            await pilot.mouse_up(table, offset=(boundary - 12, 0))
            await pilot.pause()
            assert (table.columns["name"].width) == (before - 12)
            assert (tuple(item.image_id for item in app._visible_images)) == (order)
            assert (app._current_image().image_id) == (focused)
            assert not (app._selected_image_ids)
            assert (app.mouse_captured) is None
            assert (len(self.docker.commands)) == (commands)

    async def test_manual_widths_survive_sort_filter_refresh_language_and_resize(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            table = app.query_one("#image-table", DataTable)
            await drag(pilot, table, header_offset(table, 1), -12, dy=1, release_click=True)
            assert (table.columns["name"].width) == (64)
            assert not (app._selected_image_ids), "拖到数据行松手不能勾选镜像"
            table.focus()
            await pilot.press("space")
            await pilot.pause()
            assert (app._selected_image_ids) == ({FINAL})
            assert (table.columns["name"].width) == (64)
            await pilot.click(table, offset=(header_offset(table, 1)[0] + 3, 0))
            assert (app._image_sort[0]) == ("size")
            assert (table.columns["name"].width) == (64)
            app.query_one("#image-search", Input).value = "demo/model"
            await pilot.pause()
            assert (table.row_count) == (2)
            assert (table.columns["name"].width) == (64)
            app.query_one("#image-search", Input).value = ""
            await pilot.pause()
            await pilot.click("#image-view-layers")
            layers = app.query_one("#image-layer-table", DataTable)
            await drag(pilot, layers, header_offset(layers), -5)
            assert (layers.columns["diff"].width) == (18)
            app.ui_preferences = replace(app.ui_preferences, language="en")
            app._apply_ui_preferences()
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert (table.columns["name"].width) == (64)
            assert (layers.columns["diff"].width) == (18)
            await pilot.click("#image-view-tree")
            await pilot.click("#image-view-list")
            assert (app._selected_image_ids) == ({FINAL})
            app.refresh_images()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert (table.columns["name"].width) == (64)
            assert (layers.columns["diff"].width) == (18)
            assert (len(table.columns)) == (9)
            assert (len(layers.columns)) == (4)
            assert (app._selected_image_ids) == ({FINAL}), "自动刷新保留有效勾选"
        restarted = self.make_app()
        async with restarted.run_test(size=(120, 30)) as pilot:
            await self.load_images(restarted, pilot)
            assert (restarted.query_one("#image-table", DataTable).columns["name"].width) == (76)

    @pytest.mark.parametrize('view,selector,key,index', (('list', '#image-table', 'name', 1), ('layers', '#image-layer-table', 'diff', 0)))
    @pytest.mark.parametrize('language', ('zh', 'en'))
    @pytest.mark.parametrize('size', ((80, 24), (120, 30), (150, 45)))
    async def test_header_drags_in_both_languages_and_all_terminal_sizes(self, view, selector, key, index, language, size):
        app = self.make_app()
        async with app.run_test(size=size) as pilot:
            await self.load_images(app, pilot)
            commands = len(self.docker.commands)
            app.ui_preferences = replace(app.ui_preferences, language=language)
            app._apply_ui_preferences()
            await pilot.pause()
            assert (await pilot.click("#image-view-" + view))
            table = app.query_one(selector, DataTable)
            width = table.columns[key].width
            start = header_offset(table, index)
            await drag(pilot, table, start, -5)
            assert (table.columns[key].width) == (width - 5)
            assert (header_offset(table, index)[0]) == (start[0] - 5)
            assert (table.row_count) == (3 if view == "list" else 4)
            assert (app.mouse_captured) is None
            if view == "list":
                checkbox_edge = header_offset(table)
                name_edge = header_offset(table, 1)
                row_before = table.render_line(1)
                table.scroll_to(x=6, animate=False, force=True)
                await pilot.pause()
                assert (header_offset(table)) == (checkbox_edge)
                assert (header_offset(table, 1)[0]) == (name_edge[0] - 6), "环境 / 模型列应随数据横向滚动"
                assert (table.render_line(1).crop(0, 3).text) == (row_before.crop(0, 3).text)
                assert (table.render_line(1).crop(3, 23).text) == (row_before.crop(9, 29).text)
                table.scroll_to(x=0, animate=False, force=True)
                await pilot.pause()
            assert (len(self.docker.commands)) == (commands)

    async def test_measurement_interrupts_drag_and_unlocks_afterwards(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            table = app.query_one("#image-table", DataTable)
            start = header_offset(table, 1)
            await pilot.mouse_down(table, offset=start)
            assert (app.mouse_captured) is (table)
            app._process_kind = "run"
            app._latest_snapshot = ProgressSnapshot(measurement_active=True)
            app._set_busy(True)
            await pilot.pause()
            assert (app.mouse_captured) is None
            await pilot.hover(table, offset=(start[0] - 6, 0))
            await pilot.mouse_up(table, offset=(start[0] - 6, 0))
            assert (table.columns["name"].width) == (76)
            app._process_kind = ""
            app._latest_snapshot = ProgressSnapshot()
            app._set_busy(False)
            await pilot.pause()
            await drag(pilot, table, header_offset(table, 1), -6)
            assert (table.columns["name"].width) == (70)

    @pytest.mark.parametrize('size', ((80, 24), (120, 30), (150, 45)))
    async def test_filter_selection_details_and_language_work_at_all_sizes(self, size):
        app = self.make_app()
        async with app.run_test(size=size) as pilot:
            await self.load_images(app, pilot)
            table = app.query_one("#image-table", DataTable)
            assert (table.row_count) == (3)
            table.focus()
            await pilot.pause()
            await pilot.press("space")
            await pilot.pause()
            assert (app._selected_image_ids) == ({FINAL})
            assert (await pilot.click("#image-model"))
            await pilot.pause()
            assert (app._selected_image_ids) == ({FINAL, WEIGHTS})
            assert (RUNTIME) not in (app._selected_image_ids)
            assert (table.row_count) == (2)
            table.focus()
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            assert ("acprof-build-source:") in (str(app.query_one("#image-metadata-detail", Static).content))
            assert ("HF_TOKEN") not in (str(app.query_one("#image-metadata-detail", Static).content))
            app.ui_preferences = replace(app.ui_preferences, language="en")
            app._apply_ui_preferences()
            await pilot.pause()
            assert (app._selected_image_ids) == ({FINAL, WEIGHTS})
            assert (app.query_one("#image-search", Input).value) == ("demo/model")
            assert (app.query_one("#image-delete", Button).label.plain) == ("Delete")
            assert ("All tags") in (str(app.query_one("#image-metadata-detail", Static).content))
            for selector in ("#image-search", "#image-scope", "#image-toggle", "#image-model",
                             "#image-clear", "#image-delete", "#image-table"):
                widget = app.query_one(selector)
                assert (widget.region.height) > (0), selector
                assert (widget.region.x) >= (0), selector
                assert (widget.region.right) <= (size[0]), selector
                assert (widget.region.bottom) <= (size[1] - 3), selector
            # 过滤别名仍命中合并后的同一行，清空按钮清掉筛选外的选择。
            app.query_one("#image-search", Input).value = "acprof-build-source"
            await pilot.pause()
            assert (table.row_count) == (1)
            assert ("(1 hidden)") in (str(app.query_one("#image-status", Static).content))
            assert (await pilot.click("#image-clear"))
            await pilot.pause()
            assert not (app._selected_image_ids)

    @pytest.mark.parametrize('query_case', range(2), ids=["('Qwen/Qwen2.5-0.5B', {FINAL, WEIGHTS})", "('qwen--qwen2_5-0_5b', {other})"])
    async def test_version_dots_distinguish_models_in_selection_and_search(self, query_case):
        self.docker.images[FINAL] = image(
            FINAL, ["acprof-nlp-qwen--qwen2.5-0.5b:code"], 410,
            ["os", "deps", "weights", "code"], "Qwen/Qwen2.5-0.5B")
        # 没有 MODEL_ID 时，从带点号的标签识别同一模型。
        self.docker.images[WEIGHTS] = image(
            WEIGHTS, ["acprof-weights-nlp-qwen--qwen2.5-0.5b:weights"], 400,
            ["os", "deps", "weights"])
        other = "sha256:" + "d" * 64
        self.docker.images[other] = image(
            other, ["acprof-nlp-qwen--qwen2_5-0_5b:code"], 410,
            ["os", "deps", "other-weights", "code"], "Qwen/Qwen2_5-0_5B")
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            table = app.query_one("#image-table", DataTable)
            assert (table.get_cell(FINAL, "repository").plain) == ("acprof-nlp-qwen--qwen2.5-0.5b")
            table.move_cursor(row=table.get_row_index(FINAL))
            await pilot.pause()
            assert (await pilot.click("#image-model"))
            await pilot.pause()
            assert (app._selected_image_ids) == ({FINAL, WEIGHTS})
            search = app.query_one("#image-search", Input)
            (query, expected) = tuple((('Qwen/Qwen2.5-0.5B', {FINAL, WEIGHTS}), ('qwen--qwen2_5-0_5b', {other})))[query_case]
            search.focus()
            await pilot.press("ctrl+a", *query)
            await pilot.pause()
            assert ({item.image_id for item in app._visible_images}) == (expected)

    async def test_confirmation_cancel_and_delete_all_model_tags_with_buttons_visible(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-model")
            await pilot.pause()
            await pilot.click("#image-delete")
            await pilot.pause()
            assert (app._is_busy())
            message = str(app.screen.query_one("#image-confirm-text", Static).content)
            assert ("acprof-build-source:") in (message)
            assert ("acprof-audio-demo--model:code") in (message)
            dialog = app.screen.query_one("#confirm-dialog")
            assert (dialog.region.x) > (0), "删除确认应沿用居中的有边框弹窗"
            assert (dialog.region.width) < (80)
            for selector in ("#confirm-no", "#confirm-yes"):
                button = app.screen.query_one(selector, Button)
                assert (button.region.height) > (0)
                assert (button.region.bottom) <= (24)
            await pilot.press("escape")
            await pilot.pause()
            assert not (self.docker.removals)
            assert not (app._is_busy())
            assert (app._selected_image_ids) == ({FINAL, WEIGHTS})
            await pilot.click("#image-delete")
            await pilot.pause()
            assert (await pilot.click("#confirm-yes"))
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert (list(self.docker.images)) == ([RUNTIME])
            assert (app.query_one("#image-table", DataTable).row_count) == (0)
            assert ("已处理 2") in (str(app.query_one("#image-status", Static).content))
            assert not (app._is_busy())
            assert not (app.query_one("#start-run", Button).disabled)

    @pytest.mark.parametrize("language", ("zh", "en"))
    @pytest.mark.parametrize("size", ((80, 24), (120, 30), (150, 45)))
    async def test_confirmation_reclaim_tooltip_on_hover(self, language, size):
        app = self.make_app()
        app.TOOLTIP_DELAY = 0.01
        async with app.run_test(size=(150, 45), tooltips=True) as pilot:
            await self.load_images(app, pilot)
            app.ui_preferences = replace(app.ui_preferences, language=language)
            app._apply_ui_preferences()
            await pilot.pause()
            await pilot.click("#image-model")
            await pilot.click("#image-delete")
            await pilot.resize_terminal(*size)
            await pilot.pause()
            estimate = app.screen.query_one("#image-reclaim-estimate", Static)
            info = app.screen.query_one("#image-reclaim-info", Static)
            expected = "预计可释放：约 310 B" if language == "zh" else "Estimated reclaim: about 310 B"
            hint = ("实际释放空间可能受共享镜像层和构建缓存影响，以清理后核验为准。" if language == "zh" else
                    "Actual space reclaimed may be affected by shared image layers and build cache; verify after cleanup.")
            assert str(estimate.content) == expected
            assert str(info.content) == "ℹ\ufe0e"
            assert info.region.y == estimate.region.y
            assert info.region.x == estimate.region.right + 1
            assert app.screen.region.contains_region(info.region)
            before = len(self.docker.commands)
            assert await pilot.hover(info)
            await pilot.pause(app.TOOLTIP_DELAY + 0.05)
            tooltip = app.screen.query_one(Tooltip)
            assert tooltip.display
            assert str(tooltip.content) == hint
            assert app.screen.region.contains_region(tooltip.region)
            assert await pilot.hover(estimate)
            await pilot.pause()
            assert not tooltip.display
            assert len(self.docker.commands) == before
            assert await pilot.click("#confirm-no")
            await pilot.pause()
            assert not self.docker.removals

    async def test_read_failure_keeps_inventory_and_selection_until_refresh_recovers(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-toggle")
            await pilot.pause()
            with patch("acprof.tui.image_actions.list_images", side_effect=ImageManagementError("无法执行 Docker", "permission denied")):
                app.refresh_images()
                await app.workers.wait_for_complete()
                await pilot.pause()
            assert (app.query_one("#image-table", DataTable).row_count) == (3)
            assert (app._selected_image_ids) == ({FINAL})
            assert ("permission denied") in (str(app.query_one("#image-status", Static).content))
            assert (app.query_one("#image-delete", Button).disabled)
            assert not (app._is_busy())
            app.refresh_images()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert (app.query_one("#image-table", DataTable).row_count) == (3)
            assert ("permission denied") not in (str(app.query_one("#image-status", Static).content))
            assert not (app.query_one("#image-delete", Button).disabled)

    async def test_timer_updates_visible_inventory_without_user_action(self):
        app = self.make_app()
        app.IMAGE_REFRESH_INTERVAL = 0.1
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            refreshed = asyncio.Event()
            loop = asyncio.get_running_loop()
            extra = "sha256:" + "d" * 64
            self.docker.images[extra] = image(extra, ["acprof-runtime-new:env"], 120, ["new"])

            def read():
                inventory = list_images()
                loop.call_soon_threadsafe(refreshed.set)
                return inventory

            with patch("acprof.tui.image_actions.list_images", side_effect=read):
                app._image_refresh_timer.reset()
                await asyncio.wait_for(refreshed.wait(), timeout=3)
                await app.workers.wait_for_complete()
                app._image_refresh_timer.pause()
                await pilot.pause()
            assert (app.query_one("#image-table", DataTable).row_count) == (4)
            assert (extra) in (app.query_one("#image-table", DataTable).rows)

    @pytest.mark.parametrize('callback', ('timer', 'worker'))
    async def test_queued_image_callbacks_are_safe_during_shutdown(self, callback):
        class ClosingImagesApp(AcprofTui):
            CSS_PATH = AcprofTui.CSS_PATH

            async def _close_all(self):
                # Textual marks the app as stopped before pruning the
                # screen, but App.on_unmount has not run yet. Deliver
                # an already queued callback after partial unmount.
                await self.query_one("#start-run").remove()
                try:
                    if callback == "timer":
                        self.refresh_images()
                    else:
                        self._show_images(self._image_inventory)
                finally:
                    await super()._close_all()

        app = ClosingImagesApp(
            RunConfig.smoke("demo/model"), settings_path=self.directory / "tui.json",
        )
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            commands_before_shutdown = len(self.docker.commands)
        assert (len(self.docker.commands)) == (commands_before_shutdown)

    async def test_background_page_and_confirmation_do_not_scan(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-toggle")
            app.action_show_settings()
            await pilot.pause()
            with patch("acprof.tui.image_actions.list_images") as read:
                app.refresh_images()
                await pilot.pause()
                read.assert_not_called()
                assert (app.query_one("#main-tabs", TabbedContent).active) == ("settings-tab")
            app.action_show_images()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#image-delete")
            await pilot.pause()
            with patch("acprof.tui.image_actions.list_images") as read:
                app.refresh_images()
                await pilot.pause()
                read.assert_not_called()
            await pilot.press("escape")
            await pilot.pause()
            assert (app._selected_image_ids) == ({FINAL})

    async def test_refresh_keeps_focus_selection_and_scrolled_list(self):
        for number in range(30):
            key = f"sha256:{number:064x}"
            self.docker.images[key] = image(key, [f"acprof-runtime-extra-{number:02}:env"], 200, ["extra"])
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            table = app.query_one("#image-table", DataTable)
            table.move_cursor(row=25)
            table.focus()
            await pilot.pause()
            await pilot.press("space")
            await pilot.pause()
            selected = set(app._selected_image_ids)
            current = app._current_image().image_id
            table.scroll_to(x=12, y=20, animate=False, force=True)
            await pilot.pause()
            offset = table.scroll_offset
            # 清单在读取期间变化时，搜索/勾选等交互仍可使用。
            self.docker.images[RUNTIME]["Size"] += 1
            with patch.object(ImageActions, "_execute_image_refresh") as read:
                app.refresh_images()
                app.refresh_images()
                await pilot.pause()
                read.assert_called_once()
                assert (app.focused) is (table)
                assert not (table.disabled)
                assert not (app.query_one("#image-search", Input).disabled)
                app._show_images(list_images())
            await pilot.pause()
            assert (app.focused) is (table)
            assert (app._current_image().image_id) == (current)
            assert (app._selected_image_ids) == (selected)
            assert (table.scroll_offset) == (offset)

    async def test_refresh_drops_changed_images_and_ancestors_but_preserves_free_descendants(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-model")
            self.docker.images[FINAL]["RepoTags"].append("acprof-extra:new")
            app.refresh_images()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not app._selected_image_ids, "下层标签变化时也应保留其上层"
            await pilot.click("#image-model")
            await pilot.pause()
            assert app._selected_image_ids == {WEIGHTS, FINAL}
            self.docker.containers["used"] = dict(Image=WEIGHTS, Name="/new-user", State=dict(Status="exited"))
            app.refresh_images()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app._selected_image_ids == {FINAL}, "上层被引用不妨碍删除空闲下层"
            app.query_one("#image-table", DataTable).move_cursor(row=0)
            await pilot.click("#image-model")
            assert (app._selected_image_ids) == ({FINAL})
            self.docker.daemon_id = "another-daemon"
            app.refresh_images()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not (app._selected_image_ids)

    @pytest.mark.parametrize("expanded", (False, True))
    async def test_refresh_preserves_tree_and_detail_folds(self, expanded):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.load_images(app, pilot, view="tree")
            tree = app.query_one("#image-tree", Tree)
            tree.focus()
            if expanded:
                await pilot.press("right")
            await pilot.pause()
            root = tree.cursor_node.data.image_id
            metadata = app.query_one("#image-metadata", Collapsible)
            metadata.collapsed = False
            tree.scroll_to(x=3, animate=False, force=True)
            await pilot.pause()
            offset = tree.scroll_offset
            self.docker.images[FINAL]["Size"] += 1
            app.refresh_images()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert (tree.cursor_node.data.image_id) == (root)
            assert tree.cursor_node.is_expanded == expanded
            assert not tree.cursor_node.children[0].is_expanded
            assert not (metadata.collapsed)
            assert (tree.scroll_offset) == (offset)
            current_node = tree.cursor_node
            app.refresh_images()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert (tree.cursor_node) is (current_node), "清单未变时不重建树"

    async def test_auto_refresh_pauses_during_measurement_and_resumes_after_failure(self):
        app = self.make_app()
        app.IMAGE_REFRESH_INTERVAL = 0.1
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            app._process_kind = "run"
            app._set_busy(True)
            app._consume_process_line("", ProgressSnapshot(measurement_active=True), False)
            commands = len(self.docker.commands)
            # 等待超过刷新间隔，确认实际计时回调不会读取 Docker。
            await pilot.pause(0.3)
            assert (len(self.docker.commands)) == (commands)
            assert not (app._image_refresh_timer._active.is_set())
            refreshed = asyncio.Event()
            loop = asyncio.get_running_loop()

            def read():
                inventory = list_images()
                loop.call_soon_threadsafe(refreshed.set)
                return inventory

            with patch("acprof.tui.image_actions.list_images", side_effect=read):
                app._process_finished("run", 1, None, "test failure")
                await asyncio.wait_for(refreshed.wait(), timeout=3)
                await app.workers.wait_for_complete()
                app._image_refresh_timer.pause()
                await pilot.pause()
            assert not (app._latest_snapshot.measurement_active)
            assert (app.query_one("#image-table", DataTable).row_count) == (3)

    async def test_measurement_and_image_operations_are_mutually_exclusive(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.load_images(app, pilot)
            app._process_kind = "run"
            app._latest_snapshot = ProgressSnapshot(measurement_active=True)
            app._set_busy(True)
            before = len(self.docker.commands)
            app.refresh_images()
            app.request_delete_images()
            app.select_model_images()
            app.action_show_images()
            await pilot.pause()
            assert (len(self.docker.commands)) == (before)
            assert (all(widget.disabled for widget in app.query(".image-control")))
            app._process_kind = ""
            app._latest_snapshot = ProgressSnapshot()
            app._set_busy(False)
            with patch.object(ImageActions, "_execute_image_refresh") as read, patch.object(app, "_execute_command") as execute:
                app.refresh_images()
                assert (app._is_busy())
                app._launch(PendingLaunch(("must-not-start",), "run"))
                app.action_quick_check()
                app.action_request_stop()
                execute.assert_not_called()
                read.assert_called_once()
                assert (app.query_one("#stop-run", Button).disabled)
                app._show_images(app._image_inventory)
            assert not (app._is_busy())

    async def test_container_reference_prevents_selecting_that_image(self):
        self.docker.containers["used"] = dict(Image=FINAL, Name="/kept-container", State=dict(Status="exited"))
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.load_images(app, pilot)
            table = app.query_one("#image-table", DataTable)
            table.focus()
            await pilot.pause()
            await pilot.press("space")
            await pilot.pause()
            assert not (app._selected_image_ids)
            assert (app.query_one("#image-toggle", Button).disabled)
            assert ("kept-container") in (str(app.query_one("#image-metadata-detail", Static).content))
            await pilot.click("#image-model")
            await pilot.pause()
            assert not app._selected_image_ids
            assert app.query_one("#image-delete", Button).disabled

    async def test_long_confirmation_can_scroll_and_keyboard_cancel_survives_resize(self):
        self.docker.images[WEIGHTS]["RepoTags"].extend(f"acprof-weights-audio-demo--model:old-{index}" for index in range(30))
        app = self.make_app()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-model")
            await pilot.pause()
            before = set(app._selected_image_ids)
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert (app._selected_image_ids) == (before)
            assert (app.query_one("#image-table", DataTable).columns["repository"].width) <= (38)
            await pilot.click("#image-delete")
            await pilot.pause()
            content = app.screen.query_one("#image-confirm-content")
            assert (content.max_scroll_y) > (0)
            content.focus()
            await pilot.pause()
            await pilot.press("end")
            await pilot.pause()
            assert (content.scroll_y) > (0)
            button = app.screen.query_one("#confirm-no", Button)
            assert (button.region.bottom) <= (24)
            button.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert not (self.docker.removals)
            assert not (app._is_busy())

    async def test_changed_tags_after_confirmation_are_reported_without_deleting(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.load_images(app, pilot)
            await pilot.click("#image-model")
            await pilot.pause()
            await pilot.click("#image-delete")
            await pilot.pause()
            self.docker.images[WEIGHTS]["RepoTags"].append("new-owner:keep")
            app.screen.query_one("#confirm-no", Button).focus()
            await pilot.pause()
            await pilot.press("tab", "enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not (self.docker.removals)
            assert ("标签已改变") in (str(app.query_one("#image-status", Static).content))
            assert not (app._selected_image_ids)
            assert not (app._is_busy())
