"""Image page handlers on Textual's message pump; state stays on AcprofTui."""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on, work
from textual.message_pump import MessagePump
from textual.widgets import Button, ContentSwitcher, DataTable, Select, Static, TabbedContent, Tree

from acprof.host.image_graph import reclaimable_image_bytes, retain_complete_image_selection
from acprof.host.image_management import (
    DockerStorage,
    ImageInventory,
    ImageLayer,
    ImageRemoval,
    ManagedImage,
    delete_images,
    list_images,
    read_storage,
)
from acprof.messages import join_messages, message
from acprof.tui.images import (
    IMAGE_KINDS,
    ImageDeleteScreen,
    ImageDetailPanel,
    ImageTree,
    filtered_images,
    image_display_name,
    image_error,
    image_selection_marker,
    render_image_tree,
)
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.log import SelectableLog
from acprof.tui.presentation import CALCULATING, UNKNOWN, format_bytes
from acprof.tui.storage import StorageSpaceScreen

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


class ImageActions(MessagePump):
    """Inherit MessagePump so Textual registers selector handlers through the MRO."""

    _focused_image_id: str
    _image_view: str
    _image_sort: tuple[str, bool]
    _selected_image_ids: set[str]
    _image_operation: str
    _image_inventory: ImageInventory | None
    _visible_images: tuple[ManagedImage, ...]
    _visible_image_layers: tuple[ImageLayer, ...]
    _image_refresh_error: str
    _storage_screen: StorageSpaceScreen | None
    _storage_loading: bool

    def action_show_images(self: AcprofTui) -> None:
        self._activate_tab("images-tab")

    @on(TabbedContent.TabActivated, "#main-tabs")
    def image_tab_activated(self: AcprofTui, event: TabbedContent.TabActivated) -> None:
        if event.pane.id == "images-tab":
            self.refresh_images()
        self._sync_image_refresh_timer()

    def _sync_image_refresh_timer(self: AcprofTui) -> None:
        if self._image_refresh_timer is None:
            return
        # A queued worker/form callback may run after Screen children have
        # unmounted, before App.on_unmount clears _form_ready.
        tabs = self.query("#main-tabs")
        if (not self.is_running or not self._form_ready or self._images_unavailable()
                or not tabs or tabs.first(TabbedContent).active != "images-tab"):
            self._image_refresh_timer.pause()
        else:
            # 从上一轮扫描完成后计时，慢查询不会排队或重叠。
            self._image_refresh_timer.reset()

    def _images_unavailable(self: AcprofTui, *, allow_refresh: bool = False) -> bool:
        state = self._operation_state()
        if allow_refresh and self._image_operation == "refresh":
            state = replace(state, maintenance=False)
        return not state.allows("cleanup") or self._pending_launch is not None

    def _current_image(self: AcprofTui) -> ManagedImage | None:
        if self._image_view == "layers":
            return None
        if self._image_view == "tree":
            node = self.query_one("#image-tree", ImageTree).cursor_node
            return node.data if node else None
        table = self.query_one("#image-table", DataTable)
        row = table.cursor_row
        return self._visible_images[row] if 0 <= row < len(self._visible_images) else None

    def _update_image_controls(self: AcprofTui) -> None:
        busy = self._images_unavailable(allow_refresh=True)
        current = self._current_image()
        self.query_one("#image-storage", Button).disabled = self._images_unavailable()
        self.query_one("#image-toggle", Button).disabled = (
            busy or current is None or bool(current.containers and not current.descendant_ids)
        )
        self.query_one("#image-model", Button).disabled = busy or current is None or not current.model_key or not current.acprof
        self.query_one("#image-clear", Button).disabled = busy or not self._selected_image_ids
        self.query_one("#image-delete", Button).disabled = (
            self._images_unavailable() or bool(self._image_refresh_error) or not self._selected_image_ids
        )

    def _image_selection_status(self: AcprofTui) -> None:
        if self._image_refresh_error:
            self._set_text(self.query_one("#image-status", Static), message(
                "镜像读取失败，将自动重试：{0}", self._image_refresh_error,
            ))
            return
        if self._image_inventory is None:
            return
        hidden = len(self._selected_image_ids - {item.image_id for item in self._visible_images})
        self._set_text(self.query_one("#image-status", Static), message(
            "环境 {0} · 匹配 {1}/{2} · 已选 {3}（筛选外 {4}）",
            self._image_inventory.connection.name, len(self._visible_images), len(self._image_inventory.images),
            len(self._selected_image_ids), hidden,
        ))

    def _render_images(self: AcprofTui, *, width: int | None = None, current_id: str | None = None,
                       preserve_scroll: bool = False) -> None:
        table = self.query_one("#image-table", DataTable)
        offsets = [(widget, widget.scroll_offset) for widget in (
            table, self.query_one("#image-layer-table", DataTable),
            self.query_one("#image-detail-scroll", ImageDetailPanel),
        )] if preserve_scroll else []
        current = self._current_image()
        self._visible_images = (() if self._image_inventory is None else filtered_images(
            self._image_inventory, self._input("image-search"), self._select("image-scope"),
        ))
        sort_key, reverse = self._image_sort
        key = {"name": lambda item: (image_display_name(item).casefold(), item.name),
               "size": lambda item: item.size_bytes, "added": lambda item: item.added_bytes if item.added_bytes is not None else -1,
               "containers": lambda item: len(item.containers), "repository": lambda item: item.name,
               "tag": lambda item: item.name.rsplit(":", 1)[-1], "kind": lambda item: item.kind,
               "parent": lambda item: item.parent_id}[sort_key]
        self._visible_images = tuple(sorted(self._visible_images, key=key, reverse=reverse))
        table.clear(columns=True)
        reference_width = max(30, (self.size.width if width is None else width) - 44)
        tag_width = max(12, min(28, reference_width * 2 // 5))
        repository_width = max(18, min(75, reference_width - tag_width))
        for title, key, column_width in (("☑", "selected", 1), ("环境 / 模型", "name", max(20, reference_width)),
                                         ("完整大小", "size", 10), ("新增大小", "added", 10),
                                         ("容器", "containers", 4), ("基于", "parent", 26),
                                         ("类型", "kind", 8), ("Repository", "repository", repository_width),
                                         ("Tag", "tag", tag_width)):
            table.add_column(Text(self.tr(title)), key=key, width=column_width)
        indexed = {item.image_id: item for item in self._image_inventory.images} if self._image_inventory else {}
        for item in self._visible_images:
            repository, separator, tag = item.name.rpartition(":")
            parent = indexed.get(item.parent_id)
            table.add_row(
                Text(image_selection_marker(item, self._selected_image_ids)),
                Text(self.tr(join_messages(" · ", (image_display_name(item), item.image_id[7:13]))),
                     overflow="ellipsis", no_wrap=True),
                Text(self.tr(format_bytes(item.size_bytes))), Text(self.tr(format_bytes(item.added_bytes))),
                Text(str(len(item.containers))), Text(self.tr(image_display_name(parent)) if parent else self.tr(UNKNOWN),
                     overflow="ellipsis", no_wrap=True),
                Text(self.tr(IMAGE_KINDS[item.kind])),
                Text(repository if separator else item.name, overflow="ellipsis", no_wrap=True),
                Text(tag if separator else "—", overflow="ellipsis", no_wrap=True),
                key=item.image_id,
            )
        if current_id is None:
            current_id = current.image_id if current else self._focused_image_id
        row = next((i for i, item in enumerate(self._visible_images) if item.image_id == current_id), 0)
        table.move_cursor(row=row, column=0, animate=False, scroll=not preserve_scroll)
        render_image_tree(self.query_one("#image-tree", ImageTree), self._image_inventory, self._visible_images,
                          self._selected_image_ids, current_id, self.tr, (width or self.size.width) - 6,
                          preserve_scroll=preserve_scroll, show_external_ancestors=self._select("image-scope") == "all")
        self._render_image_layers(preserve_scroll=preserve_scroll)
        self._image_selection_status()
        self._update_image_controls()
        self._show_image_detail()
        for widget, offset in offsets:
            widget.call_after_refresh(widget.scroll_to, x=offset.x, y=offset.y,
                                      animate=False, immediate=True, force=True)

    def _render_image_layers(self: AcprofTui, *, preserve_scroll: bool = False) -> None:
        table: DataTable[Text | str] = self.query_one("#image-layer-table", DataTable)
        current = self._visible_image_layers[table.cursor_row].chain_id if 0 <= table.cursor_row < len(self._visible_image_layers) else ""
        shown = {item.image_id for item in self._visible_images}
        self._visible_image_layers = tuple(sorted((layer for layer in self._image_inventory.layers
            if shown.intersection(layer.image_ids)), key=lambda layer: (-len(layer.image_ids), -(layer.size_bytes or 0), layer.chain_id))) if self._image_inventory else ()
        table.clear(columns=True)
        for title, key, width in (("Layer / Diff ID", "diff", 23), ("层大小", "size", 12),
                                  ("引用镜像", "refs", 10), ("Chain ID", "chain", 23)):
            table.add_column(self.tr(title), key=key, width=width)
        for layer in self._visible_image_layers:
            table.add_row(Text(layer.diff_id, overflow="ellipsis", no_wrap=True), self.tr(format_bytes(layer.size_bytes)),
                          str(len(layer.image_ids)), Text(layer.chain_id, overflow="ellipsis", no_wrap=True), key=layer.chain_id)
        table.move_cursor(row=next((i for i, layer in enumerate(self._visible_image_layers) if layer.chain_id == current), 0),
                          animate=False, scroll=not preserve_scroll)

    @on(Button.Pressed, "#image-view-tree, #image-view-list, #image-view-layers")
    def image_view_changed(self: AcprofTui, event: Button.Pressed) -> None:
        if self._images_unavailable(allow_refresh=True):
            return
        current = self._current_image()
        if current:
            self._focused_image_id = current.image_id
        self._image_view = event.button.id.removeprefix("image-view-")
        self.query_one("#image-browser", ContentSwitcher).current = {
            "tree": "image-tree-view", "list": "image-table", "layers": "image-layer-table"}[self._image_view]
        for view in ("tree", "list", "layers"):
            self.query_one("#image-view-" + view, Button).variant = "primary" if view == self._image_view else "default"
        self._render_images(current_id=self._focused_image_id)

    @on(DataTable.HeaderSelected, "#image-table")
    def sort_image_table(self: AcprofTui, event: DataTable.HeaderSelected) -> None:
        key = event.column_key.value
        if self._images_unavailable(allow_refresh=True) or key == "selected":
            return
        self._image_sort = (key, not self._image_sort[1] if self._image_sort[0] == key else False)
        self._render_images()

    @on(Input.Changed, "#image-search")
    @on(Select.Changed, "#image-scope")
    def image_filter_changed(self: AcprofTui) -> None:
        if self._form_ready and not self._images_unavailable(allow_refresh=True):
            self._render_images()

    @on(DataTable.RowHighlighted, "#image-table")
    @on(DataTable.RowHighlighted, "#image-layer-table")
    @on(Tree.NodeHighlighted, "#image-tree")
    def _show_image_detail(self: AcprofTui) -> None:
        detail = self.query_one("#image-detail-scroll", ImageDetailPanel)
        if self._image_view == "layers":
            row = self.query_one("#image-layer-table", DataTable).cursor_row
            layer = self._visible_image_layers[row] if 0 <= row < len(self._visible_image_layers) else None
            if layer is not None and self._image_inventory is not None:
                detail.show_layer(layer, self._image_inventory)
            else:
                detail.show_empty("没有匹配的层；可调整筛选，清单会自动刷新。")
            self._update_image_controls()
            return
        item = self._current_image()
        if item:
            self._focused_image_id = item.image_id
        if item and self._image_inventory:
            detail.show_image(item, self._image_inventory)
        else:
            detail.show_empty("没有匹配的镜像；可调整筛选，清单会自动刷新。")
        self._update_image_controls()

    @on(DataTable.RowSelected, "#image-table")
    @on(Tree.NodeSelected, "#image-tree")
    @on(Button.Pressed, "#image-toggle")
    def toggle_image_selection(self: AcprofTui) -> None:
        if self._images_unavailable(allow_refresh=True):
            return
        item = self._current_image()
        if item is None or self._image_inventory is None:
            return
        branch = {item.image_id, *item.descendant_ids}
        available = retain_complete_image_selection(self._image_inventory, branch)
        if available and available <= self._selected_image_ids:
            self._selected_image_ids.difference_update(branch)
        else:
            self._selected_image_ids.update(branch)
            if available != branch:
                self.notify("分支内有容器引用，已保留被引用镜像及其上层；其它下层仍可选择。", severity="warning")
        self._selected_image_ids = retain_complete_image_selection(self._image_inventory, self._selected_image_ids)
        self._render_images()

    @on(Button.Pressed, "#image-clear")
    def clear_image_selection(self: AcprofTui) -> None:
        if not self._images_unavailable(allow_refresh=True):
            self._selected_image_ids.clear()
            self._render_images()

    @on(Button.Pressed, "#image-model")
    def select_model_images(self: AcprofTui) -> None:
        item = self._current_image()
        if self._images_unavailable(allow_refresh=True) or item is None or not item.model_key or self._image_inventory is None:
            return
        self._selected_image_ids = retain_complete_image_selection(self._image_inventory, {
            candidate.image_id for candidate in self._image_inventory.images
            if candidate.acprof and candidate.model_key == item.model_key and not candidate.containers
            and candidate.kind not in {"runtime", "base"}
        })
        with self.prevent(Input.Changed, Select.Changed):
            self.query_one("#image-search", Input).value = item.model_id
            self.query_one("#image-scope", Select).value = "models"
        self._render_images()

    def _begin_image_operation(self: AcprofTui, operation: str, status: str) -> None:
        self._image_operation = operation
        self._activate_tab("images-tab")
        self._set_text(self.query_one("#image-status", Static), status)
        self._set_busy(True)

    def refresh_images(self: AcprofTui) -> None:
        if (not self.is_running or not self._form_ready or len(self.screen_stack) != 1 or self._images_unavailable()
                or self.query_one("#main-tabs", TabbedContent).active != "images-tab"
                or self.mouse_captured is not None):
            return
        self._image_operation = "refresh"
        if self._image_inventory is None and not self._image_refresh_error:
            self._set_text(self.query_one("#image-status", Static), message("{0} 正在读取 Docker 镜像与容器引用", CALCULATING))
        self._set_busy(True)
        self._execute_image_refresh()

    @work(thread=True, group="images", exclusive=True, exit_on_error=False)
    def _execute_image_refresh(self: AcprofTui) -> None:
        try:
            inventory, error = list_images(), ""
        except Exception as exc:
            inventory, error = None, image_error(exc)
        self.call_from_thread(self._show_images, inventory, error)

    def _show_images(self: AcprofTui, inventory: ImageInventory | None, error: str = "", *, clear_selection: bool = False) -> None:
        # Textual stops the app before pruning widgets; on_unmount runs later.
        # A queued timer or worker result must not access the disappearing UI.
        if not self.is_running or not self._form_ready:
            return
        previous = self._image_inventory
        previous_selection = set(self._selected_image_ids)
        if clear_selection:
            # 一次明确删除操作结束后，下一次删除仍需重新勾选。
            self._selected_image_ids.clear()
        self._image_operation = ""
        self._image_refresh_error = error
        if inventory is not None:
            old_images = {item.image_id: item for item in previous.images} if previous else {}
            if previous and (previous.connection, previous.daemon_id) == (inventory.connection, inventory.daemon_id):
                self._selected_image_ids.intersection_update(
                    item.image_id for item in inventory.images
                    if not item.containers and item.image_id in old_images
                    and item.tags == old_images[item.image_id].tags
                )
                self._selected_image_ids = retain_complete_image_selection(inventory, self._selected_image_ids)
            else:
                self._selected_image_ids.clear()
            self._image_inventory = inventory
        if self._image_inventory != previous or self._selected_image_ids != previous_selection:
            self._render_images(preserve_scroll=previous is not None)
        else:
            self._image_selection_status()
        self._set_busy(self._is_busy())

    @on(Button.Pressed, "#image-storage")
    def open_storage(self: AcprofTui) -> None:
        if self._images_unavailable() or len(self.screen_stack) != 1:
            return
        self._begin_image_operation("storage", "正在读取存储空间……")
        self._storage_screen = StorageSpaceScreen(self.refresh_storage, len(self._selected_image_ids))
        self.push_screen(self._storage_screen, self._storage_closed)

    def refresh_storage(self: AcprofTui) -> None:
        screen = self._storage_screen
        if (screen is None or self._storage_loading or self._image_operation != "storage"
                or self._latest_snapshot.measurement_active or self._check_running):
            return
        self._storage_loading = True
        screen.show_loading()
        self._execute_storage_refresh(self._image_inventory, tuple(sorted(self._selected_image_ids)))

    @work(thread=True, group="storage", exclusive=True, exit_on_error=False)
    def _execute_storage_refresh(self: AcprofTui, inventory: ImageInventory | None, ids: tuple[str, ...]) -> None:
        storage, estimate, error = None, None, ""
        try:
            storage = read_storage(inventory.connection if inventory else None,
                                   daemon_id=inventory.daemon_id if inventory else "")
            if ids and inventory:
                # 手动刷新时重新核验层和引用；弹窗期间不会覆盖页面勾选或滚动位置。
                current = list_images(storage.connection)
                previous = {item.image_id: item for item in inventory.images}
                available = {item.image_id for item in current.images if not item.containers
                             and item.image_id in previous and item.tags == previous[item.image_id].tags}
                if current.daemon_id == storage.daemon_id and set(ids) <= available:
                    estimate = reclaimable_image_bytes(current, ids)
                else:
                    error = message("所选镜像或引用已改变，请关闭弹窗并重新选择。")
            else:
                estimate = 0
        except Exception as exc:
            error = image_error(exc)
        try:
            self.call_from_thread(self._show_storage, storage, estimate, error)
        except RuntimeError:
            pass  # App may have exited while the bounded Docker query was finishing.

    def _show_storage(self: AcprofTui, storage: DockerStorage | None, estimate: int | None, error: str) -> None:
        if not self.is_running or not self._form_ready:
            return
        self._storage_loading = False
        if self._storage_screen is not None and self._storage_screen.is_mounted:
            self._storage_screen.show_storage(storage, estimate, error)
        else:
            self._storage_closed()

    def _storage_closed(self: AcprofTui, _result: None = None) -> None:
        self._storage_screen = None
        if not self.is_running or not self._form_ready:
            return
        if self._storage_loading:
            # 关闭弹窗不会终止正在执行的 subprocess；结束前继续阻止正式采集。
            self._set_text(self.query_one("#image-status", Static), "存储统计仍在读取，请稍候……")
            return
        self._image_operation = ""
        self._image_selection_status()
        self._set_busy(self._is_busy())

    @on(Button.Pressed, "#image-delete")
    def request_delete_images(self: AcprofTui) -> None:
        if (self._images_unavailable() or self._image_refresh_error
                or self._image_inventory is None or not self._selected_image_ids):
            return
        inventory, ids = self._image_inventory, tuple(sorted(self._selected_image_ids))
        hidden = len(self._selected_image_ids - {item.image_id for item in self._visible_images})
        self._begin_image_operation("confirm", "请核对待删除镜像及全部标签。")
        self.push_screen(
            ImageDeleteScreen(inventory, ids, hidden_count=hidden),
            lambda confirmed: self._confirmed_image_delete(confirmed, inventory, ids),
        )

    def _confirmed_image_delete(self: AcprofTui, confirmed: bool, inventory: ImageInventory, ids: tuple[str, ...]) -> None:
        self._image_operation = ""
        self._set_busy(self._is_busy())
        if not confirmed:
            self._set_text(self.query_one("#image-status", Static), "已取消删除，镜像保留。")
            return
        if self._images_unavailable():
            self.notify("请等待当前任务完成", severity="warning")
            return
        self._begin_image_operation("delete", "正在复核并删除镜像……")
        self._execute_image_delete(inventory, ids)

    @work(thread=True, group="images", exclusive=True, exit_on_error=False)
    def _execute_image_delete(self: AcprofTui, inventory: ImageInventory, ids: tuple[str, ...]) -> None:
        outcomes: tuple[ImageRemoval, ...] = ()
        error = ""
        try:
            outcomes = delete_images(inventory, ids)
        except Exception as exc:
            error = image_error(exc)
        try:
            current = list_images(inventory.connection)
        except Exception as exc:
            current = None
            error = join_messages("\n", (error, image_error(exc))) if error else image_error(exc)
        self.call_from_thread(self._image_delete_finished, current, outcomes, error)

    def _image_delete_finished(self: AcprofTui, inventory: ImageInventory | None, outcomes: tuple[ImageRemoval, ...], error: str) -> None:
        self._show_images(inventory, error, clear_selection=True)
        successful = sum(item.success for item in outcomes)
        result = message("已处理 {0} 个镜像，失败 {1} 个；操作详情见运行监控日志。", successful, len(outcomes) - successful)
        if error:
            result = join_messages("\n", (result, message("镜像操作未完成：{0}", error)))
        self._set_text(self.query_one("#image-status", Static), result)
        log = self.query_one("#run-log", SelectableLog)
        log.write(self.tr(result))
        for outcome in outcomes:
            log.write(f"{outcome.image_id}\n{outcome.detail}")
