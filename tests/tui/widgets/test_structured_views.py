"""Keyboard, copy and layout contracts for the shared structured components."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.css.styles import Styles
from textual.screen import Screen
from textual.theme import Theme
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Static, Tooltip

from acprof.tui.i18n import message_id, translate
from acprof.tui.ui.rendering import CjkCompositor
from acprof.tui.ui.structured import (
    ActionButtons,
    ActionRequested,
    Callout,
    FactDetailScreen,
    FactGrid,
    FactValue,
    StatusBanner,
    StructuredWidget,
)
from acprof.tui.ui.themes import THEME_CATALOG
from acprof.tui.ui.view_models import (
    CalloutContent,
    FactRow,
    StatusSummary,
    ViewAction,
    ViewSection,
)

RAW_PATH = "/home/kainam/实验 [原文]/e\u0301/" + "模型目录/" * 18 + "summary[1].csv"
RAW_DETAIL = "ValueError: [red]原始异常[/red]\npath=" + RAW_PATH


class StructuredHarness(App[None]):
    """Use production styles without mounting or starting collection controllers."""

    CSS_PATH = Path(__file__).resolve().parents[3] / "acprof/tui/tui.tcss"
    TOOLTIP_DELAY = 0.01

    def __init__(self, *components: Widget, language: str = "zh") -> None:
        super().__init__()
        self.components = components
        self.language = language
        self.action_ids: list[str] = []
        palette = next(item for item in THEME_CATALOG if item.name == "acprof-graphite")
        self.register_theme(Theme(**palette.theme_kwargs()))
        self.theme = palette.name

    def get_default_screen(self) -> Screen:
        screen = super().get_default_screen()
        screen._compositor = CjkCompositor()
        return screen

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="fixture-content"):
            yield from self.components

    def tr(self, source: str) -> str:
        return translate(source, self.language)

    def set_language(self, language: str) -> None:
        self.language = language
        for component in self.query(StructuredWidget):
            component.refresh_language()

    def on_action_requested(self, event: ActionRequested) -> None:
        event.stop()
        self.action_ids.append(event.action_id)


async def focus_with_tab(app, pilot, target) -> None:
    """Exercise a bounded traversal of the real keyboard focus chain."""
    for _ in range(len(app.screen.focus_chain) + 1):
        if app.focused is target:
            return
        await pilot.press("tab")
    assert app.focused is target, f"{target!r} is not reachable with Tab"


def rendered_widget_text(app, widget) -> str:
    """Read terminal cells occupied by a widget, including wrapped CJK text."""
    region = widget.content_region
    strips = app.screen._compositor.render_strips()
    return "".join(strips[y].crop(region.x, region.right).text.strip() for y in region.line_range)


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_fact_keyboard_details_and_copy_keep_the_full_unescaped_value(language):
    row = FactRow("路径", RAW_PATH, detail=RAW_DETAIL, copyable=True)
    grid = FactGrid(ViewSection("身份", (row,)))
    app = StructuredHarness(grid, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        value = grid.query_one(FactValue)
        hint = translate(message_id("structured.fact.hint"), language)
        assert hint not in rendered_widget_text(app, value)
        await focus_with_tab(app, pilot, value)
        await pilot.pause()
        assert hint in rendered_widget_text(app, value)
        assert value.render().plain != RAW_PATH
        assert "…" in value.render().plain
        await pilot.press("c")
        assert app.clipboard == RAW_PATH

        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, FactDetailScreen)
        contents = [str(item.content) for item in app.screen.query("#fact-detail-body Static")]
        assert RAW_PATH in contents
        assert RAW_DETAIL in contents
        app.copy_to_clipboard("unchanged sentinel")
        assert await pilot.click("#fact-copy")
        assert app.clipboard == RAW_PATH
        await pilot.press("escape")
        await pilot.pause()
        assert app.focused is value
        assert hint in rendered_widget_text(app, value)
        await pilot.press("tab")
        await pilot.pause()
        assert not value.has_focus
        assert hint not in rendered_widget_text(app, value)
        assert app.clipboard == RAW_PATH
        assert not app.action_ids


async def test_raw_values_matching_catalog_text_are_not_translated():
    grid = FactGrid(ViewSection("原始值", (
        FactRow("raw", "未知", copyable=True),
        FactRow("localized", message_id("common.value.unknown")),
    )))
    app = StructuredHarness(grid, language="en")
    async with app.run_test(size=(80, 24)) as pilot:
        raw, localized = list(grid.query(FactValue))
        assert raw.render().plain == "未知"
        assert str(raw.tooltip) == "未知"
        assert localized.render().plain == "Unknown"
        await focus_with_tab(app, pilot, raw)
        await pilot.press("c")
        assert app.clipboard == "未知"
        await pilot.press("enter")
        await pilot.pause()
        assert await pilot.click("#fact-copy")
        assert app.clipboard == "未知"


async def test_hover_displays_the_complete_value_and_original_detail():
    grid = FactGrid(ViewSection("身份", (FactRow("路径", RAW_PATH, detail=RAW_DETAIL, copyable=True),)))
    app = StructuredHarness(grid)
    async with app.run_test(size=(80, 24), tooltips=True) as pilot:
        value = grid.query_one(FactValue)
        assert await pilot.hover(value)
        tooltip = app.screen.query_one(Tooltip)
        async with asyncio.timeout(2):
            while not tooltip.display:
                await pilot.pause()
        assert str(tooltip.content) == RAW_PATH + "\n" + RAW_DETAIL
        assert app.screen.region.contains_region(tooltip.region)


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_readonly_details_do_not_offer_a_copy_action(language):
    grid = FactGrid(ViewSection("身份", (
        FactRow("状态", "只读"),
        FactRow("诊断", "说明", detail=RAW_DETAIL),
    )))
    app = StructuredHarness(grid, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        plain, detailed = list(grid.query(FactValue))
        hint = translate(message_id("structured.fact.details_hint"), language)
        assert plain not in app.screen.focus_chain
        assert hint not in rendered_widget_text(app, detailed)
        await focus_with_tab(app, pilot, detailed)
        await pilot.pause()
        assert hint in rendered_widget_text(app, detailed)
        assert translate(message_id("structured.fact.hint"), language) not in rendered_widget_text(app, detailed)
        app.copy_to_clipboard("sentinel")
        await pilot.press("c")
        assert app.clipboard == "sentinel"
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen.query_one("#fact-copy", Button).disabled
        assert RAW_DETAIL in [str(item.content) for item in app.screen.query("#fact-detail-body Static")]
        await pilot.press("escape")
        await pilot.pause()
        assert app.focused is detailed
        await pilot.press("tab")
        await pilot.pause()
        assert not detailed.has_focus
        assert hint not in rendered_widget_text(app, detailed)
        assert app.clipboard == "sentinel"


async def test_fact_details_expand_with_keyboard_and_mouse_without_hidden_focus():
    grid = FactGrid(ViewSection("身份", (FactRow("模型", "demo/model"),), (
        FactRow("Run ID", "run-[原文]", copyable=True),
    )))
    app = StructuredHarness(grid)
    async with app.run_test(size=(80, 24)) as pilot:
        details = grid.query_one(".fact-more", Collapsible)
        title = details.query_one("CollapsibleTitle")
        value = details.query_one(FactValue)
        assert details.collapsed
        assert value not in app.screen.focus_chain

        await focus_with_tab(app, pilot, title)
        await pilot.press("enter")
        assert not details.collapsed
        await focus_with_tab(app, pilot, value)
        await pilot.press("c")
        assert app.clipboard == "run-[原文]"

        await focus_with_tab(app, pilot, title)
        await pilot.press("enter")
        assert details.collapsed
        assert value not in app.screen.focus_chain
        assert await pilot.click(title)
        assert not details.collapsed
        await focus_with_tab(app, pilot, value)
        assert app.focused is value


async def test_banner_dispatches_only_explicit_enabled_current_actions():
    summary = StatusSummary("error", message_id("common.state.failed"), (
        ViewAction("inspect", message_id("structured.action.details"), primary=True),
        ViewAction("close", message_id("structured.action.close"), disabled=True),
    ))
    banner = StatusBanner(summary)
    app = StructuredHarness(banner)
    async with app.run_test(size=(80, 24)) as pilot:
        first, disabled = list(banner.query(Button))
        assert not app.action_ids
        await focus_with_tab(app, pilot, first)
        await pilot.press("enter")
        assert app.action_ids == ["inspect"]
        assert await pilot.click(disabled)
        await pilot.pause()
        assert app.action_ids == ["inspect"]

        first.focus()
        banner.update_summary(replace(summary, actions=(
            ViewAction("retry", message_id("structured.action.details"), primary=True),
        )))
        app.set_language("en")
        await pilot.pause()
        assert app.focused is first
        assert not disabled.display
        assert app.action_ids == ["inspect"]
        assert await pilot.click(first)
        await pilot.pause()
        assert app.action_ids == ["inspect", "retry"]
        assert not list(app.workers)


@pytest.mark.parametrize("size", [(80, 24), (120, 30)])
async def test_chinese_labels_wrap_and_status_values_and_actions_survive_resize(size):
    label = "模型的完整说明以及用于验证窄屏的较长标签"
    grid = FactGrid(ViewSection("身份", (
        FactRow(label, RAW_PATH, status="complete", copyable=True),
    )))
    banner = StatusBanner(StatusSummary("error", "实验已中断 · 0/4 case · 尚无有效测量窗口", (
        ViewAction("resume", "续跑", primary=True),
        ViewAction("inspect", "查看原因"),
    )))
    app = StructuredHarness(banner, grid)
    async with app.run_test(size=size) as pilot:
        for target_size in (size, (120, 30) if size == (80, 24) else (80, 24)):
            await pilot.resize_terminal(*target_size)
            await pilot.pause()
            key = grid.query_one(".fact-label", Static)
            value = grid.query_one(FactValue)
            assert key.region.height > 1
            assert key.region.right <= value.region.x
            assert rendered_widget_text(app, key) == label
            assert cell_len(value.render().plain) <= value.content_size.width
            assert value.region.height == 1
            assert "✓" in rendered_widget_text(app, value)
            for button, action_id in zip(banner.query(Button), ("resume", "inspect"), strict=True):
                assert button.region.width > 0
                assert button.region.height == 1
                assert app.screen.region.contains_region(button.region)
                assert await pilot.click(button)
                await pilot.pause()
                assert button.region.height == 1
                assert app.action_ids[-1] == action_id


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_callout_keeps_actionable_summary_visible_and_raw_details_copyable(language):
    content = CalloutContent(
        "warning",
        message_id("structured.reason.timeout.problem"),
        message_id("structured.reason.timeout.impact"),
        message_id("structured.reason.timeout.next"),
        detail=RAW_DETAIL,
        actions=(ViewAction("retry", "重试", primary=True),),
    )
    callout = Callout(content)
    app = StructuredHarness(callout, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        for kind, expected in (("problem", content.problem), ("impact", content.impact), ("next", content.next_step)):
            widget = callout.query_one(".callout-" + kind, Static)
            assert str(widget.content) == translate(expected, language)
            assert app.screen.region.contains_region(widget.region)
        details = callout.query_one(".callout-details", Collapsible)
        copy_button = callout.query_one(".callout-copy", Button)
        assert details.collapsed
        assert copy_button not in app.screen.focus_chain
        assert await pilot.click(callout.query_one(ActionButtons).query_one(Button))
        await pilot.pause()
        assert app.action_ids == ["retry"]

        await focus_with_tab(app, pilot, details.query_one("CollapsibleTitle"))
        await pilot.press("enter")
        assert not details.collapsed
        assert str(callout.query_one(".callout-raw", Static).content) == RAW_DETAIL
        copy_button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(copy_button)
        assert app.clipboard == RAW_DETAIL


async def test_language_changes_preserve_focus_expansion_and_raw_values():
    grid = FactGrid(ViewSection(message_id("structured.fact.details"), (
        FactRow(message_id("structured.callout.problem"), "摘要"),
    ), (FactRow("路径", RAW_PATH, copyable=True),)))
    callout = Callout(CalloutContent("warning", "问题", "影响", "下一步", detail=RAW_DETAIL))
    app = StructuredHarness(grid, callout)
    async with app.run_test(size=(120, 30)) as pilot:
        more = grid.query_one(".fact-more", Collapsible)
        await focus_with_tab(app, pilot, more.query_one("CollapsibleTitle"))
        await pilot.press("enter")
        value = more.query_one(FactValue)
        await focus_with_tab(app, pilot, value)
        app.set_language("en")
        await pilot.pause()
        assert app.focused is value
        assert not more.collapsed
        assert more.title == "More details"
        assert str(value.tooltip) == RAW_PATH
        assert callout.query_one(".callout-details", Collapsible).collapsed

        callout.show_details()
        await pilot.pause()
        copy_button = callout.query_one(".callout-copy", Button)
        assert app.focused is copy_button
        app.set_language("zh")
        await pilot.pause()
        assert app.focused is copy_button
        assert not more.collapsed
        assert not callout.query_one(".callout-details", Collapsible).collapsed
        await pilot.press("enter")
        assert app.clipboard == RAW_DETAIL

async def test_pending_fact_updates_show_latest_values_without_stealing_focus():
    banner = StatusBanner(StatusSummary("info", "读取中", (
        ViewAction("inspect", message_id("structured.action.details"), primary=True),
    )))
    grid = FactGrid(ViewSection(message_id("structured.fact.details"), ()))
    app = StructuredHarness(banner, grid)
    async with app.run_test(size=(80, 24)) as pilot:
        action = banner.query_one(Button)
        await focus_with_tab(app, pilot, action)

        # Independent readers can project metadata and rows before the DOM refresh.
        # Keep all updates in the same event-loop turn, including a locale refresh.
        grid.update_section(ViewSection(message_id("structured.fact.details"), (
            FactRow("路径", "metadata-only", copyable=True),
        )))
        grid.update_section(ViewSection(message_id("structured.fact.details"), (
            FactRow("路径", RAW_PATH, detail=RAW_DETAIL, copyable=True),
        )))
        app.set_language("en")

        async with asyncio.timeout(2):
            while len(grid.query(FactValue)) != 1:
                await pilot.pause()
        await pilot.pause()
        value = grid.query_one(FactValue)
        assert str(value.tooltip) == RAW_PATH + "\n" + RAW_DETAIL
        assert grid.query_one(".fact-section", Collapsible).title == "More details"
        assert app.focused is action
        assert not app.action_ids
        await focus_with_tab(app, pilot, value)
        await pilot.press("c")
        assert app.clipboard == RAW_PATH


@pytest.mark.parametrize(("state", "symbol", "token"), [
    ("complete", "✓", "success"),
    ("incomplete", "!", "warning"),
    ("failed", "×", "error"),
    ("running", "…", "accent"),
    ("unknown", "-", "text-muted"),
])
async def test_fact_status_uses_semantic_color_and_keeps_it_during_keyboard_focus(state, symbol, token):
    grid = FactGrid(ViewSection("状态", (
        FactRow("测量状态", "状态值", status=state, copyable=True),
    )))
    app = StructuredHarness(grid)
    async with app.run_test(size=(80, 24)) as pilot:
        value = grid.query_one(FactValue)
        app.screen.set_focus(None)
        await pilot.pause()
        # Tokens may use Textual's automatic foreground and opacity syntax.
        expected = Styles.parse(f"color: {app.get_css_variables()[token]};", (__file__, "status-token"))
        assert not value.has_focus
        assert rendered_widget_text(app, value) == symbol + " 状态值"
        assert (value.styles.color, value.styles.auto_color) == (expected.color, expected.auto_color)

        await focus_with_tab(app, pilot, value)
        await pilot.pause()
        assert value.has_focus
        assert rendered_widget_text(app, value).startswith(symbol + " 状态值")
        assert translate(message_id("structured.fact.hint"), "zh") in rendered_widget_text(app, value)
        assert (value.styles.color, value.styles.auto_color) == (expected.color, expected.auto_color)
