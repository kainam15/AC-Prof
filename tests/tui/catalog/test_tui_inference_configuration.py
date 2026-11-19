"""Declared inference controls retain intent and confirm exact resolved content."""
import pytest
from textual.widgets import Button, Collapsible, Select, Static

from acprof.experiments.configuration import RunConfig
from acprof.extensions.parameters import (
    ParameterCondition,
    ParameterConsumer,
    ParameterSpec,
    parameter_spec_to_dict,
)
from acprof.tui.models.inference_inputs import ParameterRequestInput
from acprof.tui.run.preparation import PreparationScreen
from acprof.tui.ui.input import BarCursorInput as Input
from tests.support.tui_fixtures import AcprofTui


def question(name, kind, request, **spec_values):
    spec = ParameterSpec(name, kind, consumers=(ParameterConsumer("predict", name),), **spec_values)
    return {
        "path": f"tasks.group.v1.{name}", "task_id": "group.v1", "scope": "inference",
        "parameter_spec": parameter_spec_to_dict(spec), "kind": kind,
        "request": request, "available": True, "enabled": True,
    }


def draft_event():
    sampling = question("do_sample", "boolean", {"mode": "explicit", "value": False})
    temperature = question(
        "temperature", "float", {"mode": "explicit", "value": 0.7}, minimum=0,
        requires=(ParameterCondition("do_sample", (True,), "Sampling must be enabled"),),
    )
    temperature.update({
        "enabled": False, "disabled_reason": "Sampling must be enabled",
        "dependency_paths": {"do_sample": "tasks.group.v1.do_sample"},
    })
    language = question("language", "string", {"mode": "explicit", "value": "en"}, nullable=True)
    choices = question("schedule", "enum", {"mode": "explicit", "value": "fast"},
                       choices=("fast", "accurate"))
    return {"stage": "resolution", "status": "waiting", "request": {
        "id": 7, "kind": "review", "resolved": False, "purpose": "inference_configuration",
        "questions": [sampling, temperature, language, choices,
                      question("typed_choice", "enum", {"mode": "explicit", "value": 1}, choices=(1, True))],
        "parameter_advanced": [question("stops", "list", {"mode": "unset"}, item_kind="string")],
        "fields": {"Model": "fixture/model", "Revision": "a" * 40},
    }}


async def test_modes_dependencies_typed_values_and_explicit_defaults_survive_submit(tmp_path):
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    replies = []
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#ui-language", Select).value = "en"
        screen = PreparationScreen(draft_event(), respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        temperature = screen.query_one("#preparation-answer-1", ParameterRequestInput)
        value = screen.query_one("#preparation-answer-1-value-0", Input)
        assert value.disabled
        assert "Sampling" in str(screen.query_one("#preparation-answer-1-reason", Static).content)
        assert temperature.answer() == {"mode": "explicit", "value": 0.7}
        screen.query_one("#preparation-answer-0-value-0", Select).value = "choice:1"
        await pilot.pause()
        assert not value.disabled
        screen.query_one("#preparation-answer-2-mode", Select).value = "null"
        screen.query_one("#preparation-answer-3-mode", Select).value = "auto"
        typed = screen.query_one("#preparation-answer-4-value-0", Select)
        typed.value = "choice:1"
        screen.query_one(Collapsible).collapsed = False
        await pilot.pause()
        screen.query_one("#preparation-parameter-0-mode", Select).value = "explicit"
        stops = screen.query_one("#preparation-parameter-0-value-0", Input)
        stops.value = '["END", "完成"]'
        await pilot.pause()
        for size in ((80, 24), (120, 30)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            for button in screen.query("#preparation-actions Button"):
                assert button.region.area > 0
                assert screen.region.contains_region(button.region)
                assert app.get_widget_at(*button.region.center)[0] is button
        assert await pilot.click("#preparation-apply")
        assert replies == [{"action": "answer", "answers": {
            "tasks.group.v1.do_sample": {"mode": "explicit", "value": True},
            "tasks.group.v1.temperature": {"mode": "explicit", "value": 0.7},
            "tasks.group.v1.language": {"mode": "explicit", "value": None},
            "tasks.group.v1.schedule": {"mode": "auto"},
            "tasks.group.v1.typed_choice": {"mode": "explicit", "value": True},
            "tasks.group.v1.stops": {"mode": "explicit", "value": ["END", "完成"]},
        }}]


async def test_invalid_field_remains_editable_and_unset_is_not_auto(tmp_path):
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    replies = []
    event = draft_event()
    event["request"]["questions"] = [question(
        "count", "integer", {"mode": "explicit", "value": 4}, minimum=1,
    )]
    event["request"]["parameter_advanced"] = []
    async with app.run_test(size=(80, 24)) as pilot:
        screen = PreparationScreen(event, respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        field = screen.query_one("#preparation-answer-0-value-0", Input)
        field.focus()
        await pilot.press("ctrl+a", "backspace")
        assert field.value == ""
        await pilot.click("#preparation-apply")
        await pilot.pause()
        assert not replies
        assert app.focused is field
        assert "tasks.group.v1.count" in str(screen.query_one("#preparation-error", Static).content)
        await pilot.press("4")
        assert field.value == "4"
        screen.query_one("#preparation-answer-0-mode", Select).value = "unset"
        await pilot.pause()
        assert field.disabled
        await pilot.click("#preparation-apply")
        assert replies == [{"action": "answer", "answers": {
            "tasks.group.v1.count": {"mode": "unset"},
        }}]


async def test_final_confirmation_is_bound_to_digest_and_has_a_separate_edit_action(tmp_path):
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    replies = []
    digest = "c" * 64
    event = {"stage": "resolution", "status": "waiting", "request": {
        "id": 9, "kind": "review", "purpose": "inference_confirmation", "resolved": True,
        "questions": [], "confirmation_digest": digest,
        "fields": {"Model": "fixture/model", "SHA-256": digest},
        "summary": "dtype=float32; source=explicit; validation=not_run",
    }}
    async with app.run_test(size=(80, 24)) as pilot:
        screen = PreparationScreen(event, respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        assert not screen.query(ParameterRequestInput)
        assert await pilot.click("#preparation-edit")
        assert replies == [{"action": "edit"}]
        screen.update_event(event)
        await pilot.pause()
        assert await pilot.click("#preparation-continue")
        assert replies[-1] == {"action": "confirm", "confirmation_digest": digest}
        assert len(replies) == 2


@pytest.mark.parametrize("preset", ["auto", "benchmark", "author"])
@pytest.mark.parametrize("language", ["en", "zh"])
async def test_preset_requires_an_explicit_apply_action(tmp_path, preset, language):
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    replies = []
    event = draft_event()
    event["request"]["presets"] = [
        {"id": "custom", "label": "Custom"}, {"id": "auto", "label": "Auto"},
        {"id": "benchmark", "label": "AC-Prof Benchmark v1"}, {"id": "author", "label": "Author task preset"},
    ]
    event["request"]["selected_preset"] = "custom"
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#ui-language", Select).value = language
        screen = PreparationScreen(event, respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#inference-preset", Select).value = preset
        await pilot.pause()
        assert replies == []
        assert screen.query_one("#preparation-answer-0", ParameterRequestInput).answer() == {
            "mode": "explicit", "value": False,
        }
        button = screen.query_one("#preparation-preset", Button)
        for size in ((120, 30), (80, 24)):
            await pilot.resize_terminal(*size)
            button.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert screen.region.contains_region(button.region)
            assert app.get_widget_at(*button.region.center)[0] is button
        assert await pilot.click("#preparation-preset")
        await pilot.pause()
        assert replies == [{"action": "answer", "answers": {"$preset": preset}}]


async def test_preparation_step_bar_auto_direct_edit_and_quick_preset_pills(tmp_path):
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    replies = []
    event = draft_event()
    auto_temp = question("temperature", "float", {"mode": "auto"}, minimum=0)
    auto_temp["resolved_value"] = 0.7
    event["request"]["questions"] = [auto_temp]
    event["request"]["parameter_advanced"] = []
    event["request"]["presets"] = [
        {"id": "custom", "label": "Custom"},
        {"id": "auto", "label": "Auto"},
        {"id": "benchmark", "label": "AC-Prof Benchmark v1"},
    ]
    event["request"]["selected_preset"] = "auto"
    async with app.run_test(size=(120, 30)) as pilot:
        screen = PreparationScreen(event, respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()

        # Step indicator renders clearly across the 4-stage wizard.
        steps = str(screen.query_one("#preparation-steps", Static).render())
        assert "③ 推理配置" in steps

        # Auto-mode field stays directly editable and promotes mode to explicit upon edit.
        mode = screen.query_one("#preparation-answer-0-mode", Select)
        value = screen.query_one("#preparation-answer-0-value-0", Input)
        assert mode.value == "auto"
        assert not value.disabled
        value.value = "0.9"
        await pilot.pause()
        assert mode.value == "explicit"

        # One-click quick preset pill sends preset selection directly.
        assert await pilot.click("#preparation-quick-preset-benchmark")
        await pilot.pause()
        assert replies == [{"action": "answer", "answers": {"$preset": "benchmark"}}]
