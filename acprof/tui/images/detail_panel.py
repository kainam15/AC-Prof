"""Image detail panel and drag-resize separator."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from rich.rule import Rule
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Collapsible, Static

from acprof.host.images.inventory import ImageInventory, ImageLayer, ManagedImage
from acprof.messages import message
from acprof.tui.i18n import message_id
from acprof.tui.images.formatting import (
    IMAGE_KINDS,
    dependency_detail,
    dependency_title,
    image_created_text,
    image_diagnostics,
    image_metadata,
    image_summary,
    layer_diagnostics,
    layer_image_detail,
)
from acprof.tui.images.names import image_row_label
from acprof.tui.ui.contracts import image_detail_app
from acprof.tui.ui.presentation import (
    COMPACT_COLLAPSED_SYMBOL,
    COMPACT_EXPANDED_SYMBOL,
    format_bytes,
)
from acprof.tui.ui.structured import StatusBanner
from acprof.tui.ui.view_models import StatusSummary

if TYPE_CHECKING:
    from acprof.tui.images.presentation import ImageWorkspace


# Textual supports can_focus as a widget subclass keyword.
class ImageDetailResizeHandle(Static, can_focus=True):
    """单行分隔条；使用屏幕坐标拖动，避免控件移动时位置跳变。"""

    # Screen 在控件捕获鼠标前启动文本选择；分隔条拖动不能触发选区自动滚动。
    ALLOW_SELECT = False

    BINDINGS = [
        Binding("up", "adjust(1)", show=False),
        Binding("down", "adjust(-1)", show=False),
        Binding("home", "reset_height", show=False),
        Binding("escape", "cancel_resize", show=False),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(markup=False, **kwargs)
        self.tooltip = "上下拖动调整详情高度；聚焦后 ↑/↓ 微调，Home 恢复默认。本次会话保留。"
        self._drag_origin: tuple[int, int] | None = None

    @property
    def _workspace(self) -> ImageWorkspace:
        parent = self.parent
        if parent is None or not hasattr(parent, "set_detail_height"):
            raise RuntimeError("image resize handle requires an ImageWorkspace parent")
        return cast("ImageWorkspace", parent)

    def render(self) -> Rule:
        return Rule("⋯", style="")

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if self.disabled or event.button != 1 or event.screen_offset not in self.region:
            return
        event.prevent_default()
        event.stop()
        self.focus(scroll_visible=False)
        detail = self._workspace.query_one(ImageDetailPanel)
        self._drag_origin = (event.screen_y, detail.size.height)
        self.capture_mouse()
        self.add_class("-dragging")

    def _resize_to(self, screen_y: int) -> None:
        if self._drag_origin is not None:
            origin_y, height = self._drag_origin
            self._workspace.set_detail_height(height + origin_y - screen_y)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._drag_origin is not None:
            event.prevent_default()
            event.stop()
            self._resize_to(event.screen_y)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._drag_origin is not None and event.button == 1:
            event.prevent_default()
            event.stop()
            self._resize_to(event.screen_y)
            self.finish_resize()

    def on_click(self, event: events.Click) -> None:
        event.prevent_default()
        event.stop()

    def finish_resize(self) -> None:
        self._drag_origin = None
        if self.app.mouse_captured is self:
            self.release_mouse()
        self.remove_class("-dragging")

    def on_mouse_release(self) -> None:
        self.finish_resize()

    def on_hide(self) -> None:
        self.finish_resize()

    def on_unmount(self) -> None:
        self.finish_resize()

    def watch_disabled(self, disabled: bool) -> None:
        super().watch_disabled(disabled)
        if disabled:
            self.finish_resize()

    def action_adjust(self, delta: int) -> None:
        if not self.disabled:
            self.finish_resize()
            workspace = self._workspace
            workspace.set_detail_height(workspace.query_one(ImageDetailPanel).size.height + delta)

    def action_reset_height(self) -> None:
        if not self.disabled:
            self.finish_resize()
            self._workspace.set_detail_height(None)

    def action_cancel_resize(self) -> None:
        self.finish_resize()


class ImageDetailPanel(VerticalScroll):
    """常显摘要与按需展开的详情；同一对象重绘时保留阅读状态。"""

    _detail_key: tuple[str, str] | None = None

    def compose(self) -> ComposeResult:
        app = image_detail_app(self.app)
        with Horizontal(id="image-detail-heading"):
            yield StatusBanner(
                StatusSummary("neutral", ""),
                id="image-detail-banner",
                headline_id="image-detail-title",
                include_actions=False,
            )
            yield app.localization.localized_widget(Static("", id="image-detail-created", markup=False))
        yield app.localization.localized_widget(Static(
            "选择一行查看镜像摘要；展开分组查看详情。", id="image-detail", markup=False,
        ))
        for group, content, title in (("dependencies", "dependency", "依赖清单"),
                                       ("metadata", "metadata", "镜像信息"),
                                       ("diagnostics", "diagnostic", "诊断信息")):
            with app.localization.localized_widget(Collapsible(
                title=title, collapsed=True, id="image-" + group,
                collapsed_symbol=COMPACT_COLLAPSED_SYMBOL, expanded_symbol=COMPACT_EXPANDED_SYMBOL,
            )):
                yield app.localization.localized_widget(Static("", id=f"image-{content}-detail", markup=False))

    def on_mount(self) -> None:
        self.show_empty("选择一行查看镜像摘要；展开分组查看详情。")

    def _show(self, key: tuple[str, str] | None, title: str, summary: str,
              sections: dict[str, tuple[str, str]], *, created: str | None = None,
              level: str = "neutral") -> None:
        changed = key != self._detail_key
        self._detail_key = key
        app = image_detail_app(self.app)
        self.query_one("#image-detail-heading", Horizontal).display = bool(title)
        self.query_one("#image-detail-banner", StatusBanner).update_summary(StatusSummary(level, title))
        created_label = self.query_one("#image-detail-created", Static)
        created_label.display = created is not None
        if created is not None:
            app.localization.set_text(created_label, message_id("images.summary.created", image_created_text(created)))
        app.localization.set_text(self.query_one("#image-detail", Static), summary)
        for group_id in ("image-dependencies", "image-metadata", "image-diagnostics"):
            group = self.query_one(f"#{group_id}", Collapsible)
            section_title, content = sections.get(group_id, ("", ""))
            group.display = bool(section_title)
            app.localization.set_text(group, section_title, "title")
            app.localization.set_text(group.query_one("Contents > Static", Static), content)
            if changed:
                group.collapsed = True
        if changed:
            # Collapsible 本身也会请求滚动，最后回到新对象的摘要。
            self.call_after_refresh(self.scroll_home, animate=False, immediate=True)

    def show_empty(self, text: str) -> None:
        self._show(None, "", text, {})

    def show_image(self, item: ManagedImage, inventory: ImageInventory) -> None:
        level = "warning" if inventory.warnings else "success" if item.acprof else "neutral"
        kind_label = IMAGE_KINDS.get(item.kind, IMAGE_KINDS["other"])
        self._show(("image", item.image_id), message_id("images.summary.title", message(kind_label), image_row_label(item)),
                   image_summary(item, inventory), {
                       "image-dependencies": (dependency_title(item), dependency_detail(item)),
                       "image-metadata": (message_id("images.metadata.tags_and_containers", len(item.tags), len(item.containers)),
                                          image_metadata(item, inventory)),
                       "image-diagnostics": (message("诊断信息 · 有警告" if inventory.warnings else "诊断信息"),
                                             image_diagnostics(item, inventory)),
                   }, created=item.created, level=level)

    def show_layer(self, layer: ImageLayer, inventory: ImageInventory) -> None:
        self._show(("layer", layer.chain_id), message_id("images.layers.summary_title"),
                   message_id("images.layers.size_and_references", format_bytes(layer.size_bytes), len(layer.image_ids)), {
                       "image-metadata": (message_id("images.layers.references_title", len(layer.image_ids)), layer_image_detail(layer, inventory)),
                       "image-diagnostics": (message_id("common.diagnostics.title"), layer_diagnostics(layer)),
                   }, level="neutral")
