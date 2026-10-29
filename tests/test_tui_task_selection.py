"""Task choice is a wrapping, explicit, cancellable preparation decision."""
from functools import partial

import pytest
from rich.cells import cell_len
from task_selection_fixtures import t5_profiles, t5_task
from textual.containers import VerticalScroll
from textual.widgets import Button, Checkbox, Select, Static
from tui_fixtures import AcprofTui, checkbox_text

from acprof.experiment import RunConfig
from acprof.inference_groups import group_inference_tasks
from acprof.task_presentation import ConfigField, build_task_selection
from acprof.task_workloads import parse_task
from acprof.tui.i18n import translate
from acprof.tui.preparation import PreparationScreen
from acprof.tui.task_choices import TaskChoice, TaskChoices
from acprof.tui.task_labels import format_parameter, group_label


def task_event(profiles=None):
    profiles = t5_profiles() if profiles is None else profiles
    return {"stage": "resolution", "status": "waiting", "request": {
        "id": 1, "kind": "review", "resolved": False,
        "fields": {"Model": "google-t5/t5-small", "Revision": t5_task().model_revision},
        "questions": [{"path": "tasks", "kind": "multi", "value": [],
                       "options": [item["task_id"] for item in profiles], "profiles": profiles}]}}


def long_profiles():
    profiles = group_inference_tasks([
        parse_task("summarization", {"num_beams": beam}, encoder_decoder=True, source="fixture")
        for beam in range(1, 7)])
    for index, profile in enumerate(profiles):
        profile["execution_profile"]["signature"]["future_parameter"] = {
            "nested": ["中文 [literal] <markup> " * 5 + str(index), index]}
    return profiles


@pytest.mark.parametrize("language", ["zh", "en"])
@pytest.mark.parametrize("size", [(80, 24), (120, 30)])
async def test_multiselect_requires_confirmation_and_supports_all(tmp_path, language, size):
    app = AcprofTui(RunConfig.smoke("google-t5/t5-small"), settings_path=tmp_path / "settings.json")
    replies = []
    async with app.run_test(size=size) as pilot:
        app.query_one("#ui-language", Select).value = language
        await pilot.pause()
        screen = PreparationScreen(task_event(), respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        selection = screen.query_one(TaskChoices)
        assert selection.selected == []
        for card in selection.query(TaskChoice):
            checkbox = card.query_one(Checkbox)
            description = card.query_one(".task-choice-description", Static)
            assert checkbox.region.width == 5
            assert description.region.x == checkbox.region.right
            assert description.region.width >= 55
            assert "X" not in checkbox_text(checkbox)
            assert "✓" not in checkbox_text(checkbox)
        await pilot.click("#preparation-apply")
        assert replies == []
        assert screen.query_one("#preparation-error", Static).content
        assert isinstance(app.focused, Checkbox)
        assert screen.query_one("#preparation-error").region.bottom <= screen.query_one("#preparation-actions").region.y
        await pilot.press("space")
        await pilot.pause()
        assert isinstance(app.focused, Checkbox)
        assert checkbox_text(app.focused).count("✓") == 1
        assert not screen.query_one("#preparation-error").display
        assert selection.selected == [task_event()["request"]["questions"][0]["options"][0]]
        await pilot.press("down", "space")
        assert len(selection.selected) == 2
        await pilot.press("up", "space")
        await pilot.pause()
        assert isinstance(app.focused, Checkbox)
        assert "✓" not in checkbox_text(app.focused)
        assert len(selection.selected) == 1
        for selector in ("#preparation-select-all", "#preparation-apply", "#preparation-cancel"):
            button = screen.query_one(selector, Button)
            assert button.region.bottom <= size[1]
            assert cell_len(str(button.label)) <= button.content_region.width
            assert app.get_widget_at(button.region.x + 1, button.region.y + 1)[0] is button
        await pilot.click("#preparation-select-all")
        assert len(selection.selected) == 2
        await pilot.click("#preparation-apply")
        await pilot.click("#preparation-apply")
        assert replies == [{"action": "answer", "answers": {
            "tasks": task_event()["request"]["questions"][0]["options"]}}]


@pytest.mark.parametrize("action", ["escape", "button"])
async def test_cancel_does_not_return_task_selection(tmp_path, action):
    app = AcprofTui(RunConfig.smoke("google-t5/t5-small"), settings_path=tmp_path / "settings.json")
    replies = []
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(PreparationScreen(task_event(), respond=replies.append))
        await pilot.pause()
        await pilot.click("#preparation-select-all")
        if action == "escape":
            await pilot.press("escape")
        else:
            await pilot.click("#preparation-cancel")
        assert replies == [{"action": "cancel"}]


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_long_unknown_parameters_details_and_resize_are_reachable(tmp_path, language):
    profiles = long_profiles()
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    replies = []
    async with app.run_test(size=(120, 30)) as pilot:
        app.query_one("#ui-language", Select).value = language
        await pilot.pause()
        screen = PreparationScreen(task_event(profiles), respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        selection = screen.query_one(TaskChoices)
        cards = list(selection.query(TaskChoice))
        for card in cards:
            text = card.query_one(".task-choice-description", Static)
            assert card.region.height >= 3
            assert text.region.right <= screen.query_one(VerticalScroll).content_region.right
            assert "future_parameter.nested[1]" in str(text.content)
        for previous, current in zip(cards, cards[1:], strict=False):
            assert previous.region.bottom <= current.region.y
        cards[0].query_one(".task-choice-description", Static).scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        await pilot.click(cards[0].query_one(".task-choice-description", Static))
        assert len(selection.selected) == 1
        for _ in cards[1:]:
            await pilot.press("down", "space")
        assert len(selection.selected) == 6
        assert app.focused is cards[-1].query_one(Checkbox)
        assert not screen.query(".task-declarations")
        assert screen.query_one(".task-evidence-note", Static).content
        scroll = screen.query_one(VerticalScroll)
        scroll.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        for selector in ("#preparation-select-all", "#preparation-apply", "#preparation-cancel"):
            button = screen.query_one(selector, Button)
            assert app.get_widget_at(button.region.x + 1, button.region.y + 1)[0] is button
        await pilot.click("#preparation-apply")
        assert replies[0]["answers"]["tasks"] == [profile["task_id"] for profile in profiles]


@pytest.mark.parametrize("language", ["zh", "en"])
def test_names_and_parameters_use_localized_templates_without_translating_raw_values(language):
    tr = partial(translate, language=language)
    view = build_task_selection(t5_profiles())
    labels = {group.purposes[0].kind: group_label(group, tr) for group in view.groups}
    assert labels["summarization"] == ("文本摘要" if language == "zh" else "Summarization")
    assert labels["translation"] == ("文本翻译 · 3 种语言方向" if language == "zh" else "Translation · 3 language directions")
    assert format_parameter(ConfigField(("generation", "no_repeat_ngram_size"), 0), tr) == (
        "禁止重复 n-gram = 0" if language == "zh" else "No-repeat n-gram = 0")
    assert format_parameter(ConfigField(("new.flag",), False), tr) == '["new.flag"] = false'
    assert format_parameter(ConfigField(("new.flag",), 0), tr) == '["new.flag"] = 0'
    assert format_parameter(ConfigField(("new.flag",), "开启 [bold]"), tr) == '["new.flag"] = "开启 [bold]"'
    mixed = build_task_selection(t5_profiles(generation_override={"no_repeat_ngram_size": 0,
                                                                "length_penalty": 1.0})).groups[0]
    assert group_label(mixed, tr) == labels["summarization"] + " / " + labels["translation"]
    member = mixed.profile["semantic_tasks"][0]
    member.update(kind="future-kind", task_id="future-task")
    mixed.profile["execution_profile"]["representative_task_id"] = "future-task"
    unknown = build_task_selection([mixed.profile]).groups[0]
    assert "future-task" in group_label(unknown, tr)
