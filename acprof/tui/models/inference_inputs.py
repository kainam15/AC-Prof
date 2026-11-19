"""Parameter intent controls backed by the shared declaration protocol."""
from __future__ import annotations

from typing import Any

from textual import on
from textual.containers import Horizontal, Vertical
from textual.widgets import Input as TextualInput, Select, Static

from acprof.extensions.parameters import ParameterRequest, ParameterSpec, validate_parameter_request
from acprof.extensions.schema import declaration_from_dict
from acprof.tui.i18n import message_id


class ParameterRequestInput(Vertical):
    """Keep Auto, unset and explicit values separate, including explicit null."""

    DEFAULT_CSS = """
    ParameterRequestInput { height: auto; }
    ParameterRequestInput > Horizontal { height: 3; }
    ParameterRequestInput .parameter-mode { width: 23; }
    ParameterRequestInput .parameter-value { width: 1fr; }
    ParameterRequestInput .parameter-reason { height: auto; color: $text-muted; }
    """

    def __init__(self, question: dict, *, identifier: str, translate, classes: str = ""):
        super().__init__(id=identifier, classes=classes)
        self.question = question
        self.translate = translate
        self.initial_request = question.get("request", {"mode": "unset"})
        self.effective_enabled = question.get("enabled", True)
        self._baseline_value: object = None
        self.spec = declaration_from_dict(
            ParameterSpec, question["parameter_spec"], question["path"],
        ) if "parameter_spec" in question else None

    def compose(self):
        from acprof.tui.models.review_inputs import review_input

        mode = self.initial_request["mode"]
        if mode == "explicit" and self.initial_request.get("value") is None:
            mode = "null"
        choices = [
            (self.translate(message_id("models.inference.mode_explicit")), "explicit"),
            (self.translate(message_id("models.inference.mode_auto")), "auto"),
            (self.translate(message_id("models.inference.mode_unset")), "unset"),
        ]
        if self.spec is not None and self.spec.nullable:
            choices.append((self.translate(message_id("models.inference.mode_null")), "null"))
        if mode == "null" and not any(value == mode for _, value in choices):
            mode = "explicit"  # Keep the invalid value visible for validation.
        question = self.value_question()
        with Horizontal():
            yield Select(choices, value=mode, allow_blank=False, id=f"{self.id}-mode",
                         classes="parameter-mode")
            yield review_input(question, identifier=f"{self.id}-value-0",
                               translate=self.translate, classes="parameter-value")
        yield Static("", id=f"{self.id}-reason", classes="parameter-reason", markup=False)

    def value_question(self) -> dict:
        question = {key: value for key, value in self.question.items() if key != "request"}
        if self.initial_request["mode"] == "explicit":
            question["value"] = self.initial_request.get("value")
        elif "resolved_value" in question:
            question["value"] = question["resolved_value"]
        spec = question.get("parameter_spec", {})
        question.setdefault("kind", spec.get("kind", "string"))
        question["typed_options"] = question["kind"] in {"enum", "boolean"}
        if question["kind"] == "enum":
            question["options"] = list(spec.get("choices", question.get("options", [])))
        elif question["kind"] == "boolean":
            question["options"] = [False, True]
        return question

    def on_mount(self) -> None:
        self._baseline_value = self._current_control_value()
        self.set_dependency_state(
            self.effective_enabled, self.question.get("disabled_reason", ""),
        )

    def _current_control_value(self) -> object:
        control = self.query_one(f"#{self.id}-value-0")
        return getattr(control, "value", None)

    @on(Select.Changed)
    def mode_or_value_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == f"{self.id}-mode":
            if event.value == "auto":
                self._baseline_value = self._current_control_value()
            self._sync_value_control()
        elif event.select.id == f"{self.id}-value-0" and event.value != self._baseline_value:
            self._promote_auto_to_explicit()

    @on(TextualInput.Changed)
    def value_input_changed(self, event: TextualInput.Changed) -> None:
        if event.input.id == f"{self.id}-value-0" and event.value != self._baseline_value:
            self._promote_auto_to_explicit()

    def _promote_auto_to_explicit(self) -> None:
        if not self.is_mounted:
            return
        mode_select = self.query_one(f"#{self.id}-mode", Select)
        if mode_select.value == "auto":
            with self.prevent(Select.Changed):
                mode_select.value = "explicit"

    def _sync_value_control(self) -> None:
        if not self.is_mounted:
            return
        mode = self.query_one(f"#{self.id}-mode", Select).value
        self.query_one(f"#{self.id}-value-0").disabled = (
            mode not in {"explicit", "auto"} or not self.effective_enabled
        )

    def set_dependency_state(self, enabled: bool, reason: str) -> None:
        self.effective_enabled = enabled
        if not self.is_mounted:
            return
        notice = self.query_one(f"#{self.id}-reason", Static)
        notice.update(self.translate(reason) if reason else "")
        notice.display = bool(reason)
        self._sync_value_control()

    def answer(self, *, focus_errors: bool = True) -> dict:
        from acprof.tui.models.review_inputs import review_answers

        mode = self.query_one(f"#{self.id}-mode", Select).value
        if mode in {"auto", "unset"}:
            return {"mode": mode}
        if mode == "null":
            value = None
        else:
            values = review_answers(self, [self.value_question()], prefix=f"{self.id}-value",
                                    focus_errors=focus_errors)
            value = values[self.question["path"]]
        try:
            request = ParameterRequest("explicit", value)
            if self.spec is not None:
                validate_parameter_request(self.spec, request)
        except (TypeError, ValueError) as exc:
            if focus_errors:
                target = self.query_one(f"#{self.id}-mode" if mode == "null" else f"#{self.id}-value-0")
                target.focus()
                target.scroll_visible(animate=False, immediate=True)
            raise ValueError(f"{self.question['path']}: {exc}") from exc
        return {"mode": "explicit", "value": value}

    def dependency_value(self) -> tuple[bool, Any]:
        """Use an unchanged resolved value only while its request is unchanged."""
        try:
            current = self.answer(focus_errors=False)
        except (TypeError, ValueError):
            return False, None
        if current["mode"] == "explicit":
            return True, current["value"]
        if current == self.initial_request and "resolved_value" in self.question:
            return True, self.question["resolved_value"]
        return False, None


def update_parameter_dependencies(widgets: list[ParameterRequestInput]) -> None:
    """Evaluate declared conditions without model-specific UI rules or mutation."""
    by_path = {widget.question["path"]: widget for widget in widgets}
    observed = {}
    for path, widget in by_path.items():
        known, value = widget.dependency_value()
        if known:
            observed[path] = value
    for widget in widgets:
        question, spec = widget.question, widget.spec
        available = question.get("available", True) and (spec is None or spec.enabled)
        if not available:
            widget.set_dependency_state(False, question.get("disabled_reason") or
                                        message_id("models.inference.unavailable"))
            continue
        if spec is None or not spec.requires and not spec.conflicts:
            widget.set_dependency_state(question.get("enabled", True),
                                        question.get("disabled_reason", ""))
            continue
        paths = question.get("dependency_paths", {})
        values = {name: observed[path] for name, path in paths.items() if path in observed}
        reasons = [condition.reason or message_id("models.inference.dependency_pending")
                   for condition in spec.requires if not condition.matches(values)]
        reasons.extend(condition.reason or message_id("models.inference.dependency_pending")
                       for condition in spec.conflicts if condition.matches(values))
        widget.set_dependency_state(not reasons, "; ".join(widget.translate(reason) for reason in reasons))
