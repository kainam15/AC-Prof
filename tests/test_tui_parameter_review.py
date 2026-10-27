"""Textual preparation screen exposes typed, optional model generation controls."""
import pytest
from textual.widgets import Button, Collapsible, Select, Static
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.host.detect import TaskInfo
from acprof.task_parameters import parameter_review
from acprof.task_workloads import parse_task
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.preparation import PreparationScreen


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


async def test_invalid_typed_value_stays_on_screen(tmp_path):
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    replies = []
    async with app.run_test(size=(80, 24)) as pilot:
        screen = PreparationScreen(parameter_event(), respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#preparation-answer-0", Input).value = ""
        await pilot.click("#preparation-apply")
        assert replies == []
        assert "task_0.max_new_tokens" in str(screen.query_one("#preparation-error", Static).render())
