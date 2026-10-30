"""Textual preparation screen exposes typed, optional model generation controls."""
import pytest
from textual.containers import VerticalScroll
from textual.widgets import Button, Collapsible, Select, Static

from acprof.experiments.configuration import RunConfig
from acprof.host.detect import TaskInfo
from acprof.tasks.parameters import parameter_review
from acprof.tasks.workloads import parse_task
from acprof.tui.run.preparation import PreparationScreen
from acprof.tui.ui.input import BarCursorInput as Input
from tests.support.tui_fixtures import AcprofTui


def parameter_event():
    model = TaskInfo("fixture/model", "text-generation", "nlp",
                     "transformers_model", "transformers", "f" * 40, "hub_api")
    workload = parse_task("text-generation", {}, encoder_decoder=False, source="fixture")
    review = parameter_review(model, [workload])
    return {"stage": "resolution", "status": "waiting", "request": {
        "id": 9, "kind": "review", "resolved": False, "purpose": "workload_parameters",
        "fields": {"Model": "fixture/model"}, **review,
    }}


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_parameter_review_edits_typed_controls_and_keeps_advanced_collapsible(tmp_path, language):
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    replies = []
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#ui-language", Select).value = language
        await pilot.pause()
        screen = PreparationScreen(parameter_event(), respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        advanced = screen.query_one(Collapsible)
        assert advanced.collapsed
        screen.query_one("#preparation-answer-0", Input).value = "96"
        screen.query_one("#preparation-answer-1", Select).value = True
        advanced.collapsed = False
        await pilot.pause()
        screen.query_one("#preparation-parameter-0", Input).value = "0.6"
        button = screen.query_one("#preparation-apply", Button)
        button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        await pilot.click("#preparation-apply")
        assert len(replies) == 1
        answers = replies[0]["answers"]
        assert answers["task_0.max_new_tokens"] == 96
        assert answers["task_0.do_sample"] is True
        assert answers["task_0.temperature"] == 0.6
        assert answers["task_0.top_p"] == 1.0
        assert answers["task_0.num_beams"] == 1


async def test_recommended_defaults_are_one_click_and_cancel_is_explicit(tmp_path):
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    replies = []
    async with app.run_test(size=(80, 24)) as pilot:
        screen = PreparationScreen(parameter_event(), respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        assert not screen.query("#preparation-continue")
        defaults = screen.query_one("#preparation-defaults", Button)
        assert defaults.label
        await pilot.click("#preparation-defaults")
        expected = {question["path"]: question["value"] for question in
                    parameter_event()["request"]["questions"] + parameter_event()["request"]["parameter_advanced"]}
        assert replies == [{"action": "answer", "answers": expected}]


@pytest.mark.parametrize("language", ["zh", "en"])
@pytest.mark.parametrize(("identifier", "answer_path", "replacement", "expected"), [
    ("preparation-answer-0", "task_0.max_new_tokens", "96", 96),
    ("preparation-parameter-0", "task_0.temperature", "0.6", 0.6),
])
async def test_invalid_typed_value_shows_error_and_can_be_corrected(
    tmp_path, language, identifier, answer_path, replacement, expected,
):
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    replies = []
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#ui-language", Select).value = language
        await pilot.pause()
        screen = PreparationScreen(parameter_event(), respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        if identifier.startswith("preparation-parameter"):
            advanced = screen.query_one(Collapsible)
            advanced.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert await pilot.click(advanced.query_one("CollapsibleTitle"))
            await pilot.pause()
        field = screen.query_one("#" + identifier, Input)
        field.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(field)
        await pilot.press("ctrl+a", "backspace")
        assert field.value == ""
        assert await pilot.click("#preparation-apply")
        await pilot.pause()
        assert replies == []
        assert app.focused is field
        error = screen.query_one("#preparation-error", Static)
        assert answer_path in str(error.content)
        for size in ((80, 24), (120, 30)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            assert screen.region.contains_region(error.region)
            assert screen.query_one(VerticalScroll).region.bottom <= error.region.y
            assert error.region.bottom <= screen.query_one("#preparation-actions").region.y
            assert app.get_widget_at(*error.region.center)[0] is error
            rendered = "\n".join(strip.text for strip in screen._compositor.render_strips())
            assert answer_path in rendered
            for button in screen.query("#preparation-actions Button"):
                assert button.region.area > 0
                assert screen.region.contains_region(button.region)
                assert app.get_widget_at(*button.region.center)[0] is button
        await pilot.press("ctrl+a", *replacement)
        assert field.value == replacement
        assert await pilot.click("#preparation-apply")
        await pilot.pause()
        assert len(replies) == 1
        assert replies[0]["action"] == "answer"
        assert replies[0]["answers"][answer_path] == expected
