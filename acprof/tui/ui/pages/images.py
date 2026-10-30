"""AC-Prof TUI ui/pages/images.py components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, ContentSwitcher, Static, TabPane

from acprof.tui.ui.input import BarCursorInput as Input
from acprof.tui.ui.table import ResizableDataTable

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui

def compose_images_tab(app: AcprofTui) -> ComposeResult:
    from acprof.tui.images.presentation import (
        IMAGE_HINT,
        IMAGE_PLATFORMS,
        ImageDetailPanel,
        ImageDetailResizeHandle,
        ImageTable,
        ImageTree,
        ImageTreeHeader,
        ImageWorkspace,
    )

    with TabPane("镜像管理", id="images-tab"):
        with Horizontal(id="image-header", classes="page-header"):
            with Vertical(id="image-heading"):
                yield app.localization.localized_widget(Static(IMAGE_HINT, id="image-status", classes="page-summary", markup=False))
        with Vertical(id="image-panel"):
            with Horizontal(id="image-filters"):
                yield app.localization.localized_select(
                    (("AC-Prof 镜像", "acprof"), ("全部镜像", "all"), ("模型相关", "models"),
                     ("公共依赖", "runtime"), ("公共基础", "base"),
                     *((IMAGE_PLATFORMS[key], key) for key in ("cpu", "cu124", "cu128")),
                     ("无标签", "untagged")),
                    value="acprof", allow_blank=False, id="image-scope", classes="image-control",
                )
                yield app.localization.localized_widget(Input(
                    placeholder="搜索模型、依赖、标签或镜像 ID", id="image-search", classes="image-control",
                ))
                for label, view in (("镜像树", "tree"), ("镜像列表", "list"), ("层共享", "layers")):
                    yield app.localization.localized_widget(Button(
                        label, id="image-view-" + view, classes="image-control image-view-button",
                        variant="primary" if view == "tree" else "default",
                    ))
                yield app.localization.localized_widget(Button("存储空间", id="image-storage", classes="image-control"))
            with ImageWorkspace(id="image-workspace"):
                with ContentSwitcher(initial="image-tree-view", id="image-browser"):
                    with Vertical(id="image-tree-view"):
                        yield app.localization.localized_widget(ImageTreeHeader(
                            id="image-tree-header", classes="image-control", show_cursor=False))
                        yield ImageTree("", id="image-tree", classes="image-control")
                    for table in (
                        ImageTable(id="image-table", classes="image-control", cursor_type="row",
                                   zebra_stripes=True, fixed_columns=1),
                        ResizableDataTable(id="image-layer-table", classes="image-control", cursor_type="row", zebra_stripes=True),
                    ):
                        yield app.localization.localized_widget(table)
                yield app.localization.localized_widget(ImageDetailResizeHandle(id="image-detail-resize", classes="image-control"))
                yield ImageDetailPanel(id="image-detail-scroll")
        with Horizontal(id="image-actions", classes="action-bar"):
            with Horizontal(classes="action-secondary"):
                for label, widget_id in (("勾选/取消", "image-toggle"),
                                         ("选择同模型", "image-model"), ("清空选择", "image-clear")):
                    yield app.localization.localized_widget(Button(label, id=widget_id, classes="image-control", disabled=True))
            with Horizontal(classes="action-primary"):
                yield app.localization.localized_widget(Button(
                    "删除所选", id="image-delete", classes="image-control", variant="error", disabled=True,
                ))
