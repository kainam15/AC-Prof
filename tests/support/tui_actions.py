"""Shared Pilot actions for workflow tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from textual import events
from textual.widgets import Input, TabbedContent, TabPane

from tests.support.tui_waits import wait_for_condition

if TYPE_CHECKING:
    from textual.pilot import Pilot


async def open_page(pilot: Pilot[Any], page_id: str, *, timeout: float = 3) -> None:
    """Open a main page through its navigation button and await its visible pane.

    Reopening the current page does not click its button or toggle the sidebar.
    Page-specific background work still needs its own completion condition.
    """
    # Finish queued startup activation before requesting another page.
    await pilot.pause()
    tabs = pilot.app.query_one("#main-tabs", TabbedContent)
    pane = tabs.query_one(f"#{page_id}", TabPane)
    if tabs.active != page_id:
        assert await pilot.click(f"#nav-{page_id}"), f"Navigation to {page_id} was not hit"
    await wait_for_condition(
        pilot,
        lambda: tabs.active == page_id and pane.display and pane.region.area > 0,
        description=f"page {page_id} to become visible", timeout=timeout,
        diagnostics=lambda: f"active={tabs.active}; display={pane.display}; region={pane.region}",
    )
    await pilot.pause()


async def fill_form(
    pilot: Pilot[Any], values: Mapping[str, str], *, timeout: float = 3,
) -> None:
    """Replace Input values by clicking, selecting and pasting in mapping order.

    Callers expose the relevant page/section first and wait for their specific
    validation or worker result afterwards. This only waits for input delivery.
    """
    for selector, value in values.items():
        field = pilot.app.screen.query_one(selector, Input)
        assert not field.disabled, f"Input {selector} is disabled"
        field.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(field), f"Input {selector} was not hit"
        assert pilot.app.focused is field, f"Input {selector} did not receive focus"
        await pilot.press("home", "shift+end", "backspace")
        if value:
            pilot.app.post_message(events.Paste(value))
        await wait_for_condition(
            pilot, lambda: field.value == value,
            description=f"input {selector} to contain {value!r}", timeout=timeout,
            diagnostics=lambda: f"value={field.value!r}; focused={pilot.app.focused!r}",
        )
        await pilot.pause()
