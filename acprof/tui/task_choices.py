"""Wrapping multi-select cards for complete inference-profile decisions."""
from __future__ import annotations

import json

from rich.text import Text
from textual import events
from textual.containers import Horizontal, Vertical
from textual.widgets import Checkbox, Collapsible, Static

from acprof.task_presentation import SelectionGroup, build_task_selection
from acprof.tui.i18n import message_id
from acprof.tui.task_labels import Translator, format_parameters, group_label


class TaskChoice(Horizontal):
    DEFAULT_CSS = """
    TaskChoice { height: auto; margin-top: 1; }
    TaskChoice Checkbox { width: 5; height: 1; padding: 0; }
    TaskChoice .task-choice-description { width: 1fr; height: auto; }
    TaskChoice:focus-within { background: $boost; }
    """

    def __init__(self, group: SelectionGroup, tr: Translator, *, selected: bool = False):
        super().__init__()
        self.group = group
        self.tr = tr
        self.initial_selected = selected

    def compose(self):
        yield Checkbox(value=self.initial_selected, compact=True)
        label = Text()
        label.append(group_label(self.group, self.tr), style="bold")
        difference = format_parameters(self.group.differences, self.tr) or self.tr(message_id("ui.2201"))
        label.append("\n" + self.tr(message_id("ui.2202", difference)))
        label.append("\n" + self.tr(message_id("ui.2203", self.group.representative["task_id"])), style="dim")
        yield Static(label, classes="task-choice-description")

    def on_click(self, event: events.Click) -> None:
        if not isinstance(event.widget, Checkbox):
            checkbox = self.query_one(Checkbox)
            checkbox.focus()
            checkbox.toggle()
            event.stop()


class TaskChoices(Vertical):
    """Only selected stable group IDs cross back into preparation orchestration."""

    BINDINGS = [("up", "previous_choice", ""), ("down", "next_choice", "")]
    DEFAULT_CSS = """
    TaskChoices { height: auto; }
    TaskChoices > Static { height: auto; }
    TaskChoices > Collapsible { padding: 0; margin-top: 1; }
    TaskChoices > Collapsible > Contents { padding: 0 1; }
    """

    def __init__(self, profiles: list[dict], *, translate: Translator, selected: list[str], **kwargs):
        super().__init__(**kwargs)
        self.presentation = build_task_selection(profiles)
        self.tr = translate
        self.initial_selected = selected

    def compose(self):
        view = self.presentation
        yield Static(self.tr(message_id("ui.2204", view.semantic_count, len(view.groups))), markup=False)
        common = format_parameters(view.common, self.tr) or self.tr(message_id("ui.2205"))
        yield Static(self.tr(message_id("ui.2206", common)), classes="task-common", markup=False)
        for group in view.groups:
            yield TaskChoice(group, self.tr, selected=group.task_id in self.initial_selected)
        with Collapsible(title=self.tr(message_id("ui.2207")), collapsed=True, classes="task-declarations"):
            yield Static(self.tr(message_id("ui.2208")), markup=False)
            yield Static(json.dumps([group.profile for group in view.groups],
                                    ensure_ascii=False, indent=2, sort_keys=True), markup=False)

    @property
    def selected(self) -> list[str]:
        return [choice.group.task_id for choice in self.query(TaskChoice) if choice.query_one(Checkbox).value]

    def select_all(self) -> None:
        for checkbox in self.query(Checkbox):
            checkbox.value = True

    def move_choice(self, step: int) -> None:
        boxes = list(self.query(Checkbox))
        if self.app.focused in boxes:
            index = boxes.index(self.app.focused)
            boxes[max(0, min(len(boxes) - 1, index + step))].focus()

    def action_previous_choice(self) -> None:
        self.move_choice(-1)

    def action_next_choice(self) -> None:
        self.move_choice(1)
