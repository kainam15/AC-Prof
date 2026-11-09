"""Structured multi-select cards for complete inference-profile decisions."""
from __future__ import annotations

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Checkbox, Static

from acprof.tasks.presentation import SelectionGroup, build_task_selection
from acprof.tui.models.task_choice_view import selection_summary, task_selection_view
from acprof.tui.models.task_labels import Translator, group_label
from acprof.tui.ui.checkbox import StatusCheckbox
from acprof.tui.ui.structured import FactGrid, StatusBanner, StructuredWidget
from acprof.tui.ui.view_models import ViewSection


class TaskChoice(StructuredWidget):
    """Keep selection targets separate from inspectable configuration facts."""

    def __init__(
        self, group: SelectionGroup, differences: ViewSection, tr: Translator, *, selected: bool = False,
    ) -> None:
        super().__init__()
        self.group = group
        self.differences = differences
        self.translate = tr
        self.initial_selected = selected

    def compose(self) -> ComposeResult:
        with Horizontal(classes="task-choice-header"):
            yield StatusCheckbox(value=self.initial_selected, compact=True)
            yield Static("", classes="task-choice-description", markup=False)
        yield FactGrid(self.differences, classes="task-differences")

    def on_mount(self) -> None:
        self.refresh_language()

    def refresh_language(self) -> None:
        label = Text(group_label(self.group, self.translate), style="bold")
        self.query_one(".task-choice-description", Static).update(label)
        self.query_one(Checkbox).tooltip = label

    def on_click(self, event: events.Click) -> None:
        # Disclosure, value inspection and copying must never change selection.
        header = self.query_one(".task-choice-header", Horizontal)
        label = self.query_one(".task-choice-description", Static)
        if event.widget is header or event.widget is label:
            checkbox = self.query_one(Checkbox)
            checkbox.focus()
            checkbox.toggle()
            event.stop()


class TaskChoices(StructuredWidget):
    """Only selected stable group IDs cross back into preparation orchestration."""

    BINDINGS = [("up", "previous_choice", ""), ("down", "next_choice", "")]

    def __init__(self, profiles: list[dict], *, translate: Translator, selected: list[str], **kwargs):
        super().__init__(**kwargs)
        self.presentation = build_task_selection(profiles)
        self.view = task_selection_view(self.presentation)
        self.translate = translate
        self.initial_selected = tuple(selected)

    def compose(self) -> ComposeResult:
        yield StatusBanner(selection_summary(self.view, self.initial_selected), classes="task-selection-summary")
        yield FactGrid(self.view.common, classes="task-common")
        for group, projected in zip(self.presentation.groups, self.view.groups, strict=True):
            yield TaskChoice(
                group, projected.differences, self.translate, selected=group.task_id in self.initial_selected,
            )

    def on_mount(self) -> None:
        self.refresh_language()

    def refresh_language(self) -> None:
        self.query_one(".task-selection-summary", StatusBanner).update_summary(
            selection_summary(self.view, tuple(self.selected)),
        )

    @on(Checkbox.Changed)
    def selection_changed(self) -> None:
        self.refresh_language()

    @property
    def selected(self) -> list[str]:
        return [choice.group.task_id for choice in self.query(TaskChoice) if choice.query_one(Checkbox).value]

    def select_all(self) -> None:
        for checkbox in self.query(Checkbox):
            checkbox.value = True

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in {"previous_choice", "next_choice"}:
            focused = self.app.focused
            return isinstance(focused, Checkbox) and focused in self.query(Checkbox)
        return True

    def move_choice(self, step: int) -> None:
        boxes = list(self.query(Checkbox))
        if self.app.focused in boxes:
            index = boxes.index(self.app.focused)
            boxes[max(0, min(len(boxes) - 1, index + step))].focus()

    def action_previous_choice(self) -> None:
        self.move_choice(-1)

    def action_next_choice(self) -> None:
        self.move_choice(1)
