"""A checkbox's mark conveys its value without depending on terminal colors."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Checkbox, Collapsible

from acprof.tui.ui.checkbox import StatusCheckbox
from tests.support.tui_fixtures import AcprofTui, checkbox_text


def _toggle_cells(checkbox: Checkbox) -> list[tuple[str, str | None, str | None, bool]]:
    """Return rendered (char, fg, bg, underline) cells for the 3-cell `[ ]`/`[✓]` box."""
    region = checkbox.content_region
    line = checkbox.screen._compositor.render_strips()[region.y].crop(region.x, region.x + 3)
    cells: list[tuple[str, str | None, str | None, bool]] = []
    for segment in line:
        style = segment.style
        fg = style.color.name if style and style.color else None
        bg = style.bgcolor.name if style and style.bgcolor else None
        underline = bool(style and style.underline)
        for char in segment.text:
            cells.append((char, fg, bg, underline))
    assert len(cells) == 3
    return cells


def _rendered_bg(checkbox: Checkbox, x: int, y: int) -> str | None:
    """Return the composited background color hex at screen coordinate (x, y)."""
    segment = next(iter(checkbox.screen._compositor.render_strips()[y].crop(x, x + 1)))
    style = segment.style
    return style.bgcolor.name if style and style.bgcolor else None


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


@pytest.mark.parametrize("compact", [False, True])
async def test_checkbox_bracket_interior_matches_widget_background(compact):
    app = CheckboxApp(compact)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        first = app.query_one("#first", StatusCheckbox)
        second = app.query_one("#second", StatusCheckbox)
        for checkbox in (first, second):
            expected_bg = checkbox.background_colors[1].hex.lower()
            cells = _toggle_cells(checkbox)
            assert [bg for _, _, bg, _ in cells] == [expected_bg] * 3
            assert all(fg is not None and fg != expected_bg for _, fg, _, _ in cells)

        second.disabled = True
        await pilot.pause()
        label_bg = _rendered_bg(second, second.content_region.x + 3, second.content_region.y)
        disabled_cells = _toggle_cells(second)
        assert label_bg is not None
        assert [bg for _, _, bg, _ in disabled_cells] == [label_bg] * 3
        assert all(fg is not None and fg != label_bg for _, fg, _, _ in disabled_cells)


@pytest.mark.parametrize("theme", ["acprof-graphite", "acprof-light"])
async def test_profiler_option_checkbox_interior_matches_background_across_states(tmp_path, theme):
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(90, 36)) as pilot:
        app.theme = theme
        profiling = app.query_one("#profiling-options", Collapsible)
        profiling.collapsed = False
        await pilot.pause()

        torch_box = app.query_one("#torch-profiler", StatusCheckbox)
        ncu_box = app.query_one("#ncu-profiler", StatusCheckbox)
        massif_box = app.query_one("#massif-profiler", StatusCheckbox)
        nsys_box = app.query_one("#nsys-profiler", StatusCheckbox)

        ncu_box.value = True
        massif_box.disabled = True
        nsys_box.value = True
        nsys_box.disabled = True
        torch_box.focus()
        await pilot.pause()

        for checkbox, expected_mark in (
            (torch_box, " "),
            (ncu_box, "✓"),
            (massif_box, " "),
            (nsys_box, "✓"),
        ):
            padding_bg = _rendered_bg(checkbox, checkbox.content_region.x - 1, checkbox.content_region.y)
            cells = _toggle_cells(checkbox)
            assert padding_bg is not None
            assert "".join(char for char, _, _, _ in cells) == f"[{expected_mark}]"
            assert [bg for _, _, bg, _ in cells] == [padding_bg] * 3
            assert all(fg is not None and fg != padding_bg for _, fg, _, _ in cells)
            assert not any(underline for _, _, _, underline in cells)
