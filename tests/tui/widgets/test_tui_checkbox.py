"""A checkbox's mark conveys its value without depending on terminal colors."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Checkbox

from acprof.tui.ui.checkbox import StatusCheckbox
from tests.support.tui_fixtures import checkbox_text


class CheckboxApp(App):
    def __init__(self, compact: bool):
        super().__init__()
        self.compact = compact
        self.changes: list[tuple[str | None, bool]] = []

    def compose(self) -> ComposeResult:
        yield StatusCheckbox("First 第一项", id="first", compact=self.compact)
        yield StatusCheckbox("Second 第二项", value=True, id="second", compact=self.compact)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        self.changes.append((event.checkbox.id, event.value))


@pytest.mark.parametrize("compact", [False, True])
async def test_checkbox_marks_follow_value_without_extra_events_or_layout_shift(compact):
    app = CheckboxApp(compact)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        first = app.query_one("#first", Checkbox)
        second = app.query_one("#second", Checkbox)
        unchecked = checkbox_text(first)
        assert "First 第一项" in unchecked
        assert "✓" not in unchecked
        assert checkbox_text(second).count("✓") == 1
        assert app.changes == []

        assert await pilot.click(first)
        await pilot.pause()
        checked = checkbox_text(first)
        assert checked.count("✓") == 1
        assert checked.replace("✓", " ") == unchecked
        assert first.value
        assert second.value
        assert first.has_class("-on")
        assert app.changes == [("first", True)]

        await pilot.press("space")
        await pilot.pause()
        assert checkbox_text(first) == unchecked
        assert not first.value
        assert second.value
        assert not first.has_class("-on")
        assert app.changes == [("first", True), ("first", False)]

        second.value = False
        second.disabled = True
        await pilot.pause()
        assert "✓" not in checkbox_text(second)
        assert app.changes == [("first", True), ("first", False), ("second", False)]
        assert await pilot.click(second)
        await pilot.pause()
        assert not second.value
        assert len(app.changes) == 3

        second.value = True
        await pilot.pause()
        assert checkbox_text(second).count("✓") == 1
        assert app.changes[-1] == ("second", True)
        assert len(app.changes) == 4
        second.value = True
        await pilot.pause()
        assert len(app.changes) == 4
