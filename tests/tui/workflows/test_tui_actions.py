"""Workflow actions preserve input events and reject unfinished navigation."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, TabbedContent, TabPane

from tests.support.tui_actions import fill_form, open_page


class ActionsApp(App):
    def __init__(self, *, navigate=True):
        super().__init__()
        self.navigate = navigate
        self.navigation = []
        self.edits = []

    def compose(self) -> ComposeResult:
        yield Button("Form", id="nav-form-tab", name="form-tab")
        yield Button("Other", id="nav-other-tab", name="other-tab")
        with TabbedContent(id="main-tabs"):
            with TabPane("Form", id="form-tab"):
                yield Input("old first", id="first")
                yield Input("old second", id="second")
            with TabPane("Other", id="other-tab"):
                yield Input("retained", id="other")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        page_id = event.button.name
        assert page_id is not None
        self.navigation.append(page_id)
        if self.navigate:
            self.query_one("#main-tabs", TabbedContent).active = page_id

    def on_input_changed(self, event: Input.Changed) -> None:
        self.edits.append((event.input.id, event.value))


@pytest.mark.parametrize("size", [(80, 24), (120, 30)])
async def test_open_page_uses_navigation_and_keeps_an_already_selected_page(size: tuple[int, int]):
    app = ActionsApp()
    async with app.run_test(size=size) as pilot:
        app.query_one("#nav-other-tab", Button).active_effect_duration = 0
        await open_page(pilot, "other-tab")
        assert app.query_one("#main-tabs", TabbedContent).active == "other-tab"
        assert app.query_one("#other").region.area > 0
        await open_page(pilot, "other-tab")
        assert app.navigation == ["other-tab"]


async def test_open_page_reports_the_page_that_never_became_active():
    app = ActionsApp(navigate=False)
    async with app.run_test() as pilot:
        with pytest.raises(AssertionError, match="(?s)other-tab.*active=form-tab"):
            await open_page(pilot, "other-tab", timeout=0.1)


@pytest.mark.parametrize("size", [(80, 24), (120, 30)])
async def test_fill_form_replaces_unicode_and_clears_through_focused_inputs(size: tuple[int, int]):
    app = ActionsApp()
    async with app.run_test(size=size) as pilot:
        await fill_form(pilot, {"#first": "模型 [draft]/结果", "#second": "new value"})
        assert app.query_one("#first", Input).value == "模型 [draft]/结果"
        second = app.query_one("#second", Input)
        assert second.value == "new value"
        assert app.focused is second
        assert ("first", "模型 [draft]/结果") in app.edits
        assert ("second", "new value") in app.edits
        await fill_form(pilot, {"#second": ""})
        assert second.value == ""
        assert ("second", "") in app.edits
        assert app.focused is second


async def test_fill_form_rejects_a_disabled_input_even_if_its_value_already_matches():
    app = ActionsApp()
    async with app.run_test() as pilot:
        field = app.query_one("#second", Input)
        field.disabled = True
        with pytest.raises(AssertionError, match="#second.*disabled"):
            await fill_form(pilot, {"#second": "old second"})
        assert field.value == "old second"
