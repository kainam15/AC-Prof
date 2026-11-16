"""镜像列表的筛选、文案和删除确认；不执行 Docker 命令。"""

from __future__ import annotations

from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import var
from textual.strip import Strip
from textual.widgets import Button, DataTable, Static, Tree
from textual.widgets.tree import TreeNode

from acprof.host.images.inventory import ImageInventory, ManagedImage
from acprof.tui.i18n import message_id
from acprof.tui.images.detail_panel import ImageDetailPanel, ImageDetailResizeHandle
from acprof.tui.images.formatting import (
    IMAGE_HINT,
    IMAGE_KINDS,
    deletion_message,
    dependency_detail,
    dependency_packages,
    dependency_source_detail,
    dependency_title,
    filtered_images,
    image_created_text,
    image_diagnostics,
    image_error,
    image_metadata,
    image_path,
    image_selection_marker,
    image_summary,
    layer_diagnostics,
    layer_image_detail,
    reclaimable_text,
)
from acprof.tui.images.names import image_display_name
from acprof.tui.ui.contracts import translated_app
from acprof.tui.ui.dialogs import ConfirmActionScreen
from acprof.tui.ui.presentation import (
    COMPACT_COLLAPSED_SYMBOL,
    COMPACT_EXPANDED_SYMBOL,
    UNKNOWN,
    format_bytes,
)
from acprof.tui.ui.table import ResizableDataTable

__all__ = [
    "IMAGE_HINT",
    "IMAGE_KINDS",
    "ImageDeleteScreen",
    "ImageDetailPanel",
    "ImageDetailResizeHandle",
    "ImageTable",
    "ImageTree",
    "ImageTreeHeader",
    "ImageWorkspace",
    "deletion_message",
    "dependency_detail",
    "dependency_packages",
    "dependency_source_detail",
    "dependency_title",
    "filtered_images",
    "image_created_text",
    "image_diagnostics",
    "image_error",
    "image_metadata",
    "image_path",
    "image_selection_marker",
    "image_summary",
    "layer_diagnostics",
    "layer_image_detail",
    "reclaimable_text",
    "render_image_tree",
]


class ImageTable(ResizableDataTable):
    BINDINGS = [Binding("space", "select_cursor", "勾选镜像", show=False)]

    # Textual allows async event callbacks even when its base stub is synchronous.
    async def on_click(self, event: events.Click) -> None:  # type: ignore[override]
        event.prevent_default()
        event.stop()
        if self.disabled or event.button != 1 or self._consume_resize_click(event):
            return
        if self._consume_invalid_header_click(event):
            return
        # 保留原生表头、光标和滚动处理，禁止重复点当前行发出勾选事件。
        with self.prevent(DataTable.RowSelected):
            await super()._on_click(event)
        meta = event.style.meta
        if meta.get("column") == 0 and self.is_valid_row_index(meta.get("row", -1)) and not meta.get("out_of_bounds"):
            self.action_select_cursor()


class ImageTreeHeader(ResizableDataTable):
    """树表头复用相同拖动交互，并与树的内容宽度和横向滚动同步。"""

    image_tree: ImageTree | None = None
    minimum_name_width = 6

    def on_mount(self) -> None:
        parent = self.parent
        assert parent is not None
        self.image_tree = parent.query_one(ImageTree)
        self.image_tree.column_header = self
        self.set_columns(self.size.width)

    def minimum_column_width(self, key: str | None) -> int:
        return self.minimum_name_width if key == "name" else 1

    def set_columns(self, width: int) -> None:
        self.clear(columns=True)
        for title, key, size in (("镜像依赖", "name", max(15, width - 32)),
                                 ("完整大小", "size", 10), ("新增大小", "added", 10), ("容器", "containers", 4)):
            self.add_column(Text(translated_app(self.app).tr(title)), key=key, width=size)

    @on(ResizableDataTable.ColumnResized)
    def resize_tree_columns(self, event: ResizableDataTable.ColumnResized) -> None:
        event.stop()
        if self.image_tree is not None:
            self.image_tree.set_column_widths(tuple(column.get_render_width(self) for column in self.ordered_columns))

    def watch_scroll_x(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_x(old_value, new_value)
        if self.image_tree is not None and self.image_tree.scroll_x != new_value:
            self.image_tree.scroll_to(x=new_value, animate=False, force=True)


class ImageTree(Tree[ManagedImage]):
    ICON_NODE = COMPACT_COLLAPSED_SYMBOL
    ICON_NODE_EXPANDED = COMPACT_EXPANDED_SYMBOL
    auto_expand = var(False)
    COMPONENT_CLASSES = Tree.COMPONENT_CLASSES | {"image-tree--path"}
    BINDINGS = [Binding("space", "select_cursor", show=False),
                Binding("left", "collapse_branch", show=False),
                Binding("right", "expand_branch", show=False)]
    column_header: ImageTreeHeader | None = None

    def _ancestor_line_range(self, line: int) -> range:
        node = self.get_node_at_line(line)
        if node is None or node.line < 0:
            return range(0)
        top = node
        while top.parent is not None and (self.show_root or top.parent is not self.root):
            top = top.parent
        start = top.line + 1 if (top.parent is self.root and not self.show_root) else max(0, top.line)
        return range(0) if start > node.line else range(start, node.line + 1)

    def watch_cursor_line(self, previous_line: int, line: int) -> None:
        super().watch_cursor_line(previous_line, line)
        if previous_line == line or not self.is_mounted:
            return
        # 原生 Tree 只刷新选中节点的子树；仅局部刷新旧/新祖先路径经过的可见行。
        scroll_y = self.scroll_offset.y
        visible_end = scroll_y + max(0, self.size.height)
        affected = set(self._ancestor_line_range(previous_line)) | set(self._ancestor_line_range(line))
        for target_line in sorted(affected):
            if scroll_y <= target_line < visible_end:
                self.refresh_line(target_line)

    def render_line(self, y: int) -> Strip:
        strip = super().render_line(y)
        scroll_x, scroll_y = self.scroll_offset
        line = y + scroll_y
        # 复用 Textual 8.2.8 的标签位置，沿用折叠、隐藏根节点和滚动的布局。
        region = self._get_label_region(line)
        if region is None or not self.show_guides:
            return strip
        spans = [(0, region.x, self.get_component_rich_style("tree--guides", partial=True))]
        node = self.cursor_node
        while node is not None and node.parent is not None:
            parent = node.parent
            if parent is self.root and not self.show_root:
                break
            if parent.line < line <= node.line:
                child_region = self._get_label_region(node.line)
                if child_region is not None:
                    x = child_region.x - self.guide_depth
                    # 经过其它分支时只亮竖线；到路径节点才延伸横线。
                    end = child_region.x - 1 if line == node.line else x + 1
                    spans.append((x, end, self.get_component_rich_style("image-tree--path", partial=True)))
                break
            node = parent
        for start, end, style in spans:
            start, end = max(0, start - scroll_x), min(strip.cell_length, end - scroll_x)
            if start < end:
                strip = Strip.join((
                    strip.crop(0, start),
                    Strip(Segment.apply_style(strip.crop(start, end), post_style=style)),
                    strip.crop(end),
                ))
        return strip

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name_labels: dict[str, Text] = {}

    def set_column_widths(self, widths: tuple[int, ...]) -> None:
        def update(node, depth):
            for child in node.children:
                item = child.data
                label = self.name_labels[item.image_id].copy()
                name_width = max(4, widths[0] - depth * self.guide_depth - 2)
                label.truncate(name_width, overflow="ellipsis", pad=True)
                for value, width in zip((format_bytes(item.size_bytes), format_bytes(item.added_bytes),
                                         str(len(item.containers))), widths[1:], strict=True):
                    cell = Text(translated_app(self.app).tr(value))
                    cell.truncate(width - 2, overflow="ellipsis")
                    cell.align("left", width - 2)
                    label.append(" ")
                    label.append(cell)
                    label.append(" ")
                child.set_label(label)
                update(child, depth + 1)
        update(self.root, 0)
        # TreeNode.set_label 只刷新行；重算虚拟宽度后才能滚到新增的列区域。
        self._invalidate()

    def watch_scroll_x(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_x(old_value, new_value)
        if self.column_header is not None and self.column_header.scroll_x != new_value:
            self.column_header.scroll_to(x=new_value, animate=False, force=True)

    async def on_click(self, event: events.Click) -> None:
        event.prevent_default()
        event.stop()
        if self.disabled or event.button != 1:
            return
        meta = event.style.meta
        if meta.get("toggle") or meta.get("image_checkbox"):
            await super()._on_click(event)
        elif "line" in meta:
            self.cursor_line = meta["line"]

    def render_label(self, node, base_style, style):
        label = super().render_label(node, base_style, style)
        # 原生 Tree 的叶节点没有展开箭头；保留同宽空白，避免数值列左移两格。
        return label if node.allow_expand else Text("  ") + label

    def action_collapse_branch(self) -> None:
        node = self.cursor_node
        if node and node.is_expanded and node.children:
            node.collapse()
        elif node and node.parent is not self.root:
            self.move_cursor(node.parent)

    def action_expand_branch(self) -> None:
        node = self.cursor_node
        if node and node.children:
            if node.is_expanded:
                self.move_cursor(node.children[0])
            else:
                node.expand()


class ImageWorkspace(Vertical):
    """在可用空间内分配列表和详情高度，保留本次会话的手动值。"""

    _detail_height: int | None = None
    _layout_width: int = 0

    class WidthChanged(Message):
        """The image page has a new width after terminal or sidebar layout."""

    def on_resize(self, event: events.Resize) -> None:
        self.query_one(ImageDetailResizeHandle).finish_resize()
        self.call_after_refresh(self._apply_detail_height)
        if self._layout_width != event.size.width:
            self._layout_width = event.size.width
            self.post_message(self.WidthChanged())

    def _clamp_detail_height(self, height: int) -> int:
        # 为列表保留三行、分隔条保留一行；短终端仍能独立滚动两侧。
        return max(3, min(height, self.size.height - 4))

    def set_detail_height(self, height: int | None) -> None:
        self._detail_height = None if height is None else self._clamp_detail_height(height)
        self._apply_detail_height()

    def _apply_detail_height(self) -> None:
        if not self.is_mounted or not self.size.height:
            return
        height = self._detail_height
        if height is None:
            height = 7 if self.app.has_class("short") else 9
        # 缩窗只限制显示高度，不覆盖手动值，放大后可以恢复。
        self.query_one(ImageDetailPanel).styles.height = self._clamp_detail_height(height)


class ImageDeleteScreen(ConfirmActionScreen):
    """长标签清单可独立滚动，确认按钮在小终端仍保持可见。"""

    CSS = ConfirmActionScreen.CSS + """
    ImageDeleteScreen { align: center middle; }
    ImageDeleteScreen #confirm-dialog { height: 85%; max-height: 40; }
    #image-reclaim-row { height: auto; margin-bottom: 1; }
    #image-reclaim-estimate { width: auto; }
    #image-reclaim-info {
        width: 3; margin-left: 1; content-align: center middle;
        color: $accent; text-style: bold;
    }
    #image-confirm-content { height: 1fr; margin-bottom: 1; }
    #image-confirm-text { height: auto; }
    """

    def __init__(self, inventory: ImageInventory, image_ids: tuple[str, ...], *, hidden_count: int = 0) -> None:
        super().__init__("删除所选镜像？", deletion_message(inventory, image_ids, hidden_count=hidden_count), "删除镜像")
        self.estimate = message_id('images.delete.reclaim_detail', reclaimable_text(inventory, image_ids))

    def compose(self) -> ComposeResult:
        tr = translated_app(self.app).tr
        with Vertical(id="confirm-dialog"):
            yield Static(tr(self.dialog_title), id="confirm-title", markup=False)
            with Horizontal(id="image-reclaim-row"):
                yield Static(tr(self.estimate), id="image-reclaim-estimate", markup=False)
                # 使用文本样式的信息符号，避免终端自动切换成 emoji。
                info = Static("ℹ\ufe0e", id="image-reclaim-info", markup=False)
                info.tooltip = tr("实际释放空间可能受共享镜像层和构建缓存影响，以清理后核验为准。")
                yield info
            with VerticalScroll(id="image-confirm-content"):
                yield Static(tr(self.message), id="image-confirm-text", markup=False)
            with Horizontal(id="confirm-buttons"):
                yield Button(tr("取消"), id="confirm-no")
                yield Button(tr(self.confirm_label), id="confirm-yes", variant="error")


def render_image_tree(tree: ImageTree, inventory: ImageInventory | None, visible: tuple[ManagedImage, ...],
                      selected: set[str], current_id: str, tr, width: int, *, preserve_scroll: bool = False,
                      show_external_ancestors: bool = False) -> None:
    offset = tree.scroll_offset
    previous = {}
    def collect(node):
        for child in node.children:
            previous[child.data.image_id] = child.is_expanded
            collect(child)
    collect(tree.root)
    tree.clear()
    tree.name_labels.clear()
    tree.show_root = False
    indexed = {item.image_id: item for item in inventory.images} if inventory is not None else {}
    matches = {item.image_id for item in visible}
    # The default AC-Prof view retains managed ancestors, but not third-party
    # parent images. The complete "all" view still shows the real Docker graph.
    shown = matches | {key for item in visible for key in item.ancestor_ids
                       if show_external_ancestors or indexed[key].acprof}
    header = tree.screen.query_one("#image-tree-header", ImageTreeHeader)
    header.minimum_name_width = max((sum(key in shown for key in item.ancestor_ids) * tree.guide_depth + 4
                                     for item in visible), default=6)
    header.set_columns(width)
    if inventory is None:
        return
    nodes: dict[str, TreeNode[ManagedImage]] = {}
    root_node = tree.root
    if root_node is None:
        raise RuntimeError("image tree root missing after reset")
    for item in sorted((indexed[key] for key in shown), key=lambda item: (len(item.ancestor_ids), image_display_name(item), item.image_id)):
        parent = nodes.get(item.parent_id) or root_node
        marker = image_selection_marker(item, selected)
        # 同名镜像仍按 image ID 分别管理；完整 ID 在详情中查看。
        logical = image_display_name(item, parent.data)
        if item.model_id:
            logical = f"{tr(IMAGE_KINDS[item.kind])} · {logical}"
        evidence = (" ≈" if item.parent_source == "layer-prefix" and item.parent_id in nodes
                    else f" · {tr(UNKNOWN)}" if item.parent_source in {"ambiguous", "missing", "conflict"}
                    else "")
        name = f" {marker}  {tr(logical)}{evidence}"
        label = Text(name, style="dim" if item.image_id not in matches else "")
        # 复选框及左右各一格留白可点击，不覆盖箭头、名称或数值。
        if not item.containers or item.descendant_ids:
            label.stylize(Style(meta={"image_checkbox": True}), 0, 3)
        tree.name_labels[item.image_id] = label
        nodes[item.image_id] = parent.add(label, data=item, expand=previous.get(item.image_id, False))
    for node in nodes.values():
        node.allow_expand = bool(node.children)
    tree.set_column_widths(tuple(column.get_render_width(header) for column in header.ordered_columns))
    target = nodes.get(current_id) or next(iter(nodes.values()), None)
    while (target is not None and target.parent is not None
           and target.parent is not tree.root and not target.parent.is_expanded):
        target = target.parent
    # 新节点要等 Textual 完成行布局，才有可用于 move_cursor 的行号。
    tree.call_after_refresh(tree.move_cursor, target)
    if preserve_scroll:
        tree.call_after_refresh(tree.scroll_to, x=offset.x, y=offset.y, animate=False, immediate=True, force=True)
