"""Image inventory, selection and maintenance, scheduled by the owning App."""
from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

from rich.text import Text
from textual.timer import Timer
from textual.widgets import (
    Button,
    Collapsible,
    ContentSwitcher,
    DataTable,
    Select,
    Static,
    TabbedContent,
)
from textual.worker import Worker

from acprof.host.images.graph import reclaimable_image_bytes, retain_complete_image_selection
from acprof.host.images.inventory import (
    DockerStorage,
    ImageInventory,
    ImageLayer,
    ImageRemoval,
    ManagedImage,
    delete_images,
    list_images,
    read_storage,
)
from acprof.messages import Message, join_messages
from acprof.tui.i18n import message_id
from acprof.tui.images.presentation import (
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
from acprof.tui.images.storage import StorageSpaceScreen
from acprof.tui.logging.widget import SelectableLog
from acprof.tui.ui.input import BarCursorInput as Input
from acprof.tui.ui.presentation import CALCULATING, UNKNOWN, format_bytes

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


class ImageController:
    """Own images state and operations; the App owns lifecycle and scheduling."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app
        self.inventory: ImageInventory | None = None
        self.operation = ""
        self.refresh_timer: Timer | None = None
        self.refresh_error = ""
        self.storage_screen: StorageSpaceScreen | None = None
        self.storage_loading = False
        self.selected_ids: set[str] = set()
        self.visible_images: tuple[ManagedImage, ...] = ()
        self.visible_layers: tuple[ImageLayer, ...] = ()
        self.view = "tree"
        self.sort = ("name", False)
        self.focused_id = ""
        self.dependency_pending: set[str] = set()

    def show_images(self) -> None:
        self.app.activate_tab("images-tab")

    def image_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.pane.id == "images-tab":
            self.refresh_images()
        self.sync_image_refresh_timer()

    def sync_image_refresh_timer(self) -> None:
        if self.refresh_timer is None:
            return
        # A queued worker/form callback may run after Screen children have
        # unmounted, before App.on_unmount clears _form_ready.
        tabs = self.app.query("#main-tabs")
        if (not self.app.is_running or not self.app.form_ready or self.images_unavailable()
                or not tabs or tabs.first(TabbedContent).active != "images-tab"):
            self.refresh_timer.pause()
        else:
            # 从上一轮扫描完成后计时，慢查询不会排队或重叠。
            self.refresh_timer.reset()

    def images_unavailable(self, *, allow_refresh: bool = False) -> bool:
        state = self.app.operation_state()
        if allow_refresh and self.operation == "refresh":
            state = replace(state, maintenance=False)
        return not state.allows("cleanup") or self.app.experiments.pending_launch is not None

    def current_image(self) -> ManagedImage | None:
        if self.view == "layers":
            return None
        if self.view == "tree":
            node = self.app.query_one("#image-tree", ImageTree).cursor_node
            return node.data if node else None
        table = self.app.query_one("#image-table", DataTable)
        row = table.cursor_row
        return self.visible_images[row] if 0 <= row < len(self.visible_images) else None

    def update_image_controls(self) -> None:
        busy = self.images_unavailable(allow_refresh=True)
        current = self.current_image()
        self.app.query_one("#image-storage", Button).disabled = self.images_unavailable()
        self.app.query_one("#image-toggle", Button).disabled = (
            busy or current is None or bool(current.containers and not current.descendant_ids)
        )
        self.app.query_one("#image-model", Button).disabled = busy or current is None or not current.model_key or not current.acprof
        self.app.query_one("#image-clear", Button).disabled = busy or not self.selected_ids
        self.app.query_one("#image-delete", Button).disabled = (
            self.images_unavailable() or bool(self.refresh_error) or not self.selected_ids
        )

    def image_selection_status(self) -> None:
        if self.refresh_error:
            self.app.localization.set_text(self.app.query_one("#image-status", Static), message_id(
                'images.inventory.retrying', self.refresh_error,
            ))
            return
        if self.inventory is None:
            return
        hidden = len(self.selected_ids - {item.image_id for item in self.visible_images})
        self.app.localization.set_text(self.app.query_one("#image-status", Static), message_id(
            'images.selection.status',
            self.inventory.connection.name, len(self.visible_images), len(self.inventory.images),
            len(self.selected_ids), hidden,
        ))

    def render_images(self, *, width: int | None = None, current_id: str | None = None,
                       preserve_scroll: bool = False) -> None:
        table = self.app.query_one("#image-table", DataTable)
        offsets = [(widget, widget.scroll_offset) for widget in (
            table, self.app.query_one("#image-layer-table", DataTable),
            self.app.query_one("#image-detail-scroll", ImageDetailPanel),
        )] if preserve_scroll else []
        current = self.current_image()
        self.visible_images = (() if self.inventory is None else filtered_images(
            self.inventory, self.app.configuration.input("image-search"), self.app.configuration.select("image-scope"),
        ))
        sort_key, reverse = self.sort
        sorter = {"name": lambda item: (image_display_name(item).casefold(), item.name),
               "size": lambda item: item.size_bytes, "added": lambda item: item.added_bytes if item.added_bytes is not None else -1,
               "containers": lambda item: len(item.containers), "repository": lambda item: item.name,
               "tag": lambda item: item.name.rsplit(":", 1)[-1], "kind": lambda item: item.kind,
               "parent": lambda item: item.parent_id}[sort_key]
        self.visible_images = tuple(sorted(self.visible_images, key=sorter, reverse=reverse))
        table.clear(columns=True)
        reference_width = max(30, (self.app.size.width if width is None else width) - 44)
        tag_width = max(12, min(28, reference_width * 2 // 5))
        repository_width = max(18, min(75, reference_width - tag_width))
        for title, key, column_width in (("☑", "selected", 1), ("环境 / 模型", "name", max(20, reference_width)),
                                         ("完整大小", "size", 10), ("新增大小", "added", 10),
                                         ("容器", "containers", 4), ("基于", "parent", 26),
                                         ("类型", "kind", 8), ("Repository", "repository", repository_width),
                                         ("Tag", "tag", tag_width)):
            table.add_column(Text(self.app.localization.tr(title)), key=key, width=column_width)
        indexed = {item.image_id: item for item in self.inventory.images} if self.inventory else {}
        for item in self.visible_images:
            repository, separator, tag = item.name.rpartition(":")
            parent = indexed.get(item.parent_id)
            table.add_row(
                Text(image_selection_marker(item, self.selected_ids)),
                Text(self.app.localization.tr(join_messages(" · ", (image_display_name(item), item.image_id[7:13]))),
                     overflow="ellipsis", no_wrap=True),
                Text(self.app.localization.tr(format_bytes(item.size_bytes))), Text(self.app.localization.tr(format_bytes(item.added_bytes))),
                Text(str(len(item.containers))), Text(self.app.localization.tr(image_display_name(parent)) if parent else self.app.localization.tr(UNKNOWN),
                     overflow="ellipsis", no_wrap=True),
                Text(self.app.localization.tr(IMAGE_KINDS[item.kind])),
                Text(repository if separator else item.name, overflow="ellipsis", no_wrap=True),
                Text(tag if separator else "—", overflow="ellipsis", no_wrap=True),
                key=item.image_id,
            )
        if current_id is None:
            current_id = current.image_id if current else self.focused_id
        row = next((i for i, item in enumerate(self.visible_images) if item.image_id == current_id), 0)
        table.move_cursor(row=row, column=0, animate=False, scroll=not preserve_scroll)
        render_image_tree(self.app.query_one("#image-tree", ImageTree), self.inventory, self.visible_images,
                          self.selected_ids, current_id, self.app.localization.tr, (width or self.app.size.width) - 6,
                          preserve_scroll=preserve_scroll, show_external_ancestors=self.app.configuration.select("image-scope") == "all")
        self.render_image_layers(preserve_scroll=preserve_scroll)
        self.image_selection_status()
        self.update_image_controls()
        self.show_image_detail()
        for widget, offset in offsets:
            widget.call_after_refresh(widget.scroll_to, x=offset.x, y=offset.y,
                                      animate=False, immediate=True, force=True)

    def render_image_layers(self, *, preserve_scroll: bool = False) -> None:
        table: DataTable[Text | str] = self.app.query_one("#image-layer-table", DataTable)
        current = self.visible_layers[table.cursor_row].chain_id if 0 <= table.cursor_row < len(self.visible_layers) else ""
        shown = {item.image_id for item in self.visible_images}
        self.visible_layers = tuple(sorted((layer for layer in self.inventory.layers
            if shown.intersection(layer.image_ids)), key=lambda layer: (-len(layer.image_ids), -(layer.size_bytes or 0), layer.chain_id))) if self.inventory else ()
        table.clear(columns=True)
        for title, key, width in (("Layer / Diff ID", "diff", 23), ("层大小", "size", 12),
                                  ("引用镜像", "refs", 10), ("Chain ID", "chain", 23)):
            table.add_column(self.app.localization.tr(title), key=key, width=width)
        for layer in self.visible_layers:
            table.add_row(Text(layer.diff_id, overflow="ellipsis", no_wrap=True), self.app.localization.tr(format_bytes(layer.size_bytes)),
                          str(len(layer.image_ids)), Text(layer.chain_id, overflow="ellipsis", no_wrap=True), key=layer.chain_id)
        table.move_cursor(row=next((i for i, layer in enumerate(self.visible_layers) if layer.chain_id == current), 0),
                          animate=False, scroll=not preserve_scroll)

    def image_view_changed(self, event: Button.Pressed) -> None:
        if self.images_unavailable(allow_refresh=True):
            return
        current = self.current_image()
        if current:
            self.focused_id = current.image_id
        if event.button.id is None:
            return
        self.view = event.button.id.removeprefix("image-view-")
        self.app.query_one("#image-browser", ContentSwitcher).current = {
            "tree": "image-tree-view", "list": "image-table", "layers": "image-layer-table"}[self.view]
        for view in ("tree", "list", "layers"):
            self.app.query_one("#image-view-" + view, Button).variant = "primary" if view == self.view else "default"
        self.render_images(current_id=self.focused_id)

    def sort_image_table(self, event: DataTable.HeaderSelected) -> None:
        key = event.column_key.value
        if self.images_unavailable(allow_refresh=True) or key is None or key == "selected":
            return
        self.sort = (key, not self.sort[1] if self.sort[0] == key else False)
        self.render_images()

    def image_filter_changed(self) -> None:
        if self.app.form_ready and not self.images_unavailable(allow_refresh=True):
            self.render_images()

    def show_image_detail(self) -> None:
        detail = self.app.query_one("#image-detail-scroll", ImageDetailPanel)
        if self.view == "layers":
            row = self.app.query_one("#image-layer-table", DataTable).cursor_row
            layer = self.visible_layers[row] if 0 <= row < len(self.visible_layers) else None
            if layer is not None and self.inventory is not None:
                detail.show_layer(layer, self.inventory)
            else:
                detail.show_empty("没有匹配的层；可调整筛选，清单会自动刷新。")
            self.update_image_controls()
            return
        item = self.current_image()
        if item:
            self.focused_id = item.image_id
        if item and self.inventory:
            detail.show_image(item, self.inventory)
        else:
            detail.show_empty("没有匹配的镜像；可调整筛选，清单会自动刷新。")
        self.update_image_controls()

    def expand_image_dependencies(self) -> None:
        inventory, item = self.inventory, self.current_image()
        if inventory is None or item is None or item.dependency_source != "unknown":
            return
        if self.app.execution.session.snapshot.measurement_active:
            self.app.localization.set_text(self.app.query_one("#image-dependency-detail", Static),
                           message_id('images.dependencies.measurement_blocked'))
            return
        pending = self.dependency_pending
        if item.image_id in pending:
            return
        pending.add(item.image_id)
        panel = self.app.query_one("#image-dependencies", Collapsible)
        self.app.localization.set_text(panel, message_id('images.dependencies.loading_title'), "title")
        self.app.localization.set_text(self.app.query_one("#image-dependency-detail", Static),
                       message_id('images.dependencies.loading_description'))
        self.execute_image_dependency_resolution(inventory, item.image_id)

    def execute_image_dependency_resolution(self, inventory: ImageInventory, image_id: str) -> Worker[None]:
        """Schedule this operation on the owning App worker manager."""
        return self.app.run_worker(
            partial(self._execute_image_dependency_resolution_worker, inventory, image_id),
            name="execute_image_dependency_resolution", thread=True, group='image-dependency-resolution', exit_on_error=False,
        )

    def _execute_image_dependency_resolution_worker(self, inventory: ImageInventory, image_id: str) -> None:
        from acprof.host.images.dependencies import resolve_image_dependencies

        try:
            resolved = resolve_image_dependencies(inventory, image_id)
        except Exception as error:
            from dataclasses import replace as dataclass_replace
            images = tuple(dataclass_replace(item, dependency_note=f"依赖查询失败（{type(error).__name__}）。")
                           if item.image_id == image_id else item for item in inventory.images)
            resolved = dataclass_replace(inventory, images=images)
        self.app.call_from_thread(self.complete_image_dependency_resolution, inventory, image_id, resolved)

    def complete_image_dependency_resolution(self, original: ImageInventory,
                                              image_id: str, resolved: ImageInventory) -> None:
        self.dependency_pending.discard(image_id)
        if not self.app.is_running or not self.app.form_ready or self.inventory is not original:
            return
        self.inventory = resolved
        # Preserve the existing selection, expanded group and scroll positions.
        self.render_images(current_id=self.focused_id, preserve_scroll=True)

    def toggle_image_selection(self) -> None:
        if self.images_unavailable(allow_refresh=True):
            return
        item = self.current_image()
        if item is None or self.inventory is None:
            return
        branch = {item.image_id, *item.descendant_ids}
        available = retain_complete_image_selection(self.inventory, branch)
        if available and available <= self.selected_ids:
            self.selected_ids.difference_update(branch)
        else:
            self.selected_ids.update(branch)
            if available != branch:
                self.app.notify("分支内有容器引用，已保留被引用镜像及其上层；其它下层仍可选择。", severity="warning")
        self.selected_ids = retain_complete_image_selection(self.inventory, self.selected_ids)
        self.render_images()

    def clear_image_selection(self) -> None:
        if not self.images_unavailable(allow_refresh=True):
            self.selected_ids.clear()
            self.render_images()

    def select_model_images(self) -> None:
        item = self.current_image()
        if self.images_unavailable(allow_refresh=True) or item is None or not item.model_key or self.inventory is None:
            return
        self.selected_ids = retain_complete_image_selection(self.inventory, {
            candidate.image_id for candidate in self.inventory.images
            if candidate.acprof and candidate.model_key == item.model_key and not candidate.containers
            and candidate.kind not in {"runtime", "base"}
        })
        with self.app.prevent(Input.Changed, Select.Changed):
            self.app.query_one("#image-search", Input).value = item.model_id
            self.app.query_one("#image-scope", Select).value = "models"
        self.render_images()

    def begin_image_operation(self, operation: str, status: str) -> None:
        self.operation = operation
        self.app.activate_tab("images-tab")
        self.app.localization.set_text(self.app.query_one("#image-status", Static), status)
        self.app.set_busy(True)

    def refresh_images(self) -> None:
        if (not self.app.is_running or not self.app.form_ready or len(self.app.screen_stack) != 1 or self.images_unavailable()
                or self.app.query_one("#main-tabs", TabbedContent).active != "images-tab"
                or self.app.mouse_captured is not None):
            return
        self.operation = "refresh"
        if self.inventory is None and not self.refresh_error:
            self.app.localization.set_text(self.app.query_one("#image-status", Static), message_id('images.inventory.loading', CALCULATING))
        self.app.set_busy(True)
        self.execute_image_refresh()

    def execute_image_refresh(self) -> Worker[None]:
        """Schedule this operation on the owning App worker manager."""
        return self.app.run_worker(
            partial(self._execute_image_refresh_worker),
            name="execute_image_refresh", thread=True, group='images', exclusive=True, exit_on_error=False,
        )

    def _execute_image_refresh_worker(self) -> None:
        try:
            inventory, error = list_images(), ""
        except Exception as exc:
            inventory, error = None, image_error(exc)
        self.app.call_from_thread(self.show_inventory, inventory, error)

    def show_inventory(self, inventory: ImageInventory | None, error: str = "", *, clear_selection: bool = False) -> None:
        # Textual stops the app before pruning widgets; on_unmount runs later.
        # A queued timer or worker result must not access the disappearing UI.
        if not self.app.is_running or not self.app.form_ready:
            return
        previous = self.inventory
        previous_selection = set(self.selected_ids)
        if clear_selection:
            # 一次明确删除操作结束后，下一次删除仍需重新勾选。
            self.selected_ids.clear()
        self.operation = ""
        self.refresh_error = error
        if inventory is not None:
            old_images = {item.image_id: item for item in previous.images} if previous else {}
            if previous and (previous.connection, previous.daemon_id) == (inventory.connection, inventory.daemon_id):
                self.selected_ids.intersection_update(
                    item.image_id for item in inventory.images
                    if not item.containers and item.image_id in old_images
                    and item.tags == old_images[item.image_id].tags
                )
                self.selected_ids = retain_complete_image_selection(inventory, self.selected_ids)
            else:
                self.selected_ids.clear()
            self.inventory = inventory
        if self.inventory != previous or self.selected_ids != previous_selection:
            self.render_images(preserve_scroll=previous is not None)
        else:
            self.image_selection_status()
        self.app.set_busy(self.app.is_busy())

    def open_storage(self) -> None:
        if self.images_unavailable() or len(self.app.screen_stack) != 1:
            return
        self.begin_image_operation("storage", "正在读取存储空间……")
        self.storage_screen = StorageSpaceScreen(self.refresh_storage, len(self.selected_ids))
        self.app.push_screen(self.storage_screen, self.storage_closed)

    def refresh_storage(self) -> None:
        screen = self.storage_screen
        if (screen is None or self.storage_loading or self.operation != "storage"
                or self.app.execution.session.snapshot.measurement_active or self.app.preflight.check_running):
            return
        self.storage_loading = True
        screen.show_loading()
        self.execute_storage_refresh(self.inventory, tuple(sorted(self.selected_ids)))

    def execute_storage_refresh(self, inventory: ImageInventory | None, ids: tuple[str, ...]) -> Worker[None]:
        """Schedule this operation on the owning App worker manager."""
        return self.app.run_worker(
            partial(self._execute_storage_refresh_worker, inventory, ids),
            name="execute_storage_refresh", thread=True, group='storage', exclusive=True, exit_on_error=False,
        )

    def _execute_storage_refresh_worker(self, inventory: ImageInventory | None, ids: tuple[str, ...]) -> None:
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
                    error = message_id('images.selection.references_changed')
            else:
                estimate = 0
        except Exception as exc:
            error = image_error(exc)
        try:
            self.app.call_from_thread(self.show_storage, storage, estimate, error)
        except RuntimeError:
            pass  # App may have exited while the bounded Docker query was finishing.

    def show_storage(self, storage: DockerStorage | None, estimate: int | None, error: str) -> None:
        if not self.app.is_running or not self.app.form_ready:
            return
        self.storage_loading = False
        if self.storage_screen is not None and self.storage_screen.is_mounted:
            self.storage_screen.show_storage(storage, estimate, error)
        else:
            self.storage_closed()

    def storage_closed(self, _result: None = None) -> None:
        self.storage_screen = None
        if not self.app.is_running or not self.app.form_ready:
            return
        if self.storage_loading:
            # 关闭弹窗不会终止正在执行的 subprocess；结束前继续阻止正式采集。
            self.app.localization.set_text(self.app.query_one("#image-status", Static), "存储统计仍在读取，请稍候……")
            return
        self.operation = ""
        self.image_selection_status()
        self.app.set_busy(self.app.is_busy())

    def request_delete_images(self) -> None:
        if (self.images_unavailable() or self.refresh_error
                or self.inventory is None or not self.selected_ids):
            return
        inventory, ids = self.inventory, tuple(sorted(self.selected_ids))
        hidden = len(self.selected_ids - {item.image_id for item in self.visible_images})
        self.begin_image_operation("confirm", "请核对待删除镜像及全部标签。")
        self.app.push_screen(
            ImageDeleteScreen(inventory, ids, hidden_count=hidden),
            lambda confirmed: self.confirmed_image_delete(confirmed, inventory, ids),
        )

    def confirmed_image_delete(self, confirmed: bool, inventory: ImageInventory, ids: tuple[str, ...]) -> None:
        self.operation = ""
        self.app.set_busy(self.app.is_busy())
        if not confirmed:
            self.app.localization.set_text(self.app.query_one("#image-status", Static), "已取消删除，镜像保留。")
            return
        if self.images_unavailable():
            self.app.notify("请等待当前任务完成", severity="warning")
            return
        self.begin_image_operation("delete", "正在复核并删除镜像……")
        self.execute_image_delete(inventory, ids)

    def execute_image_delete(self, inventory: ImageInventory, ids: tuple[str, ...]) -> Worker[None]:
        """Schedule this operation on the owning App worker manager."""
        return self.app.run_worker(
            partial(self._execute_image_delete_worker, inventory, ids),
            name="execute_image_delete", thread=True, group='images', exclusive=True, exit_on_error=False,
        )

    def _execute_image_delete_worker(self, inventory: ImageInventory, ids: tuple[str, ...]) -> None:
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
        self.app.call_from_thread(self.image_delete_finished, current, outcomes, error)

    def image_delete_finished(self, inventory: ImageInventory | None, outcomes: tuple[ImageRemoval, ...], error: str) -> None:
        self.show_inventory(inventory, error, clear_selection=True)
        successful = sum(item.success for item in outcomes)
        result: Message = message_id('images.delete.result_summary', successful, len(outcomes) - successful)
        if error:
            result = join_messages("\n", (result, message_id('images.operation.incomplete', error)))
        self.app.localization.set_text(self.app.query_one("#image-status", Static), result)
        log = self.app.query_one("#run-log", SelectableLog)
        log.write(self.app.localization.tr(result))
        for outcome in outcomes:
            log.write(f"{outcome.image_id}\n{outcome.detail}")
