"""Task choice is a structured, explicit, cancellable preparation decision."""
import json
from functools import partial

import pytest
from rich.cells import cell_len
from textual.containers import VerticalScroll
from textual.widgets import Button, Checkbox, Collapsible, Select, Static

from acprof.experiments.configuration import RunConfig
from acprof.tasks.groups import group_inference_tasks
from acprof.tasks.presentation import build_task_selection
from acprof.tasks.workloads import parse_task
from acprof.tui.i18n import translate
from acprof.tui.models.task_choices import TaskChoice, TaskChoices
from acprof.tui.models.task_labels import group_label
from acprof.tui.run.preparation import PreparationScreen
from acprof.tui.ui.structured import FactDetailScreen, FactGrid, FactValue, StatusBanner
from tests.support.task_selection_fixtures import t5_profiles, t5_task
from tests.support.tui_fixtures import AcprofTui, checkbox_text


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
        cards = list(selection.query(TaskChoice))
        assert selection.selected == []
        assert app.focused is cards[0].query_one(Checkbox)
        summary = selection.query_one(".task-selection-summary", StatusBanner).query_one(".status-headline", Static)
        assert str(summary.content) == ("- 4 种用途归为 2 组 · 待选择" if language == "zh"
                                        else "- 4 use cases in 2 groups · choose tasks")
        common = selection.query_one(".task-common", FactGrid)
        assert common.query_one(".fact-more", Collapsible).collapsed
        for card in cards:
            checkbox = card.query_one(Checkbox)
            description = card.query_one(".task-choice-description", Static)
            differences = card.query_one(".task-differences", FactGrid)
            assert app.get_widget_at(*checkbox.region.center)[0] is checkbox
            assert description.region.x == checkbox.region.right
            assert description.region.width > 0
            assert description.region.right <= card.region.right
            assert differences.region.right <= screen.query_one(VerticalScroll).content_region.right
            assert differences.query_one(".fact-more", Collapsible).collapsed
            assert "X" not in checkbox_text(checkbox)
            assert "✓" not in checkbox_text(checkbox)
            assert "…" not in checkbox.render_line(0).text
        await pilot.click("#preparation-apply")
        assert replies == []
        assert screen.query_one("#preparation-error", Static).content
        assert app.focused is cards[0].query_one(Checkbox)
        assert screen.query_one("#preparation-error").region.bottom <= screen.query_one("#preparation-actions").region.y
        await pilot.press("space")
        await pilot.pause()
        assert checkbox_text(app.focused).count("✓") == 1
        selected_line = app.focused.render_line(0).text
        assert selected_line.count("✓") == 1
        assert "…" not in selected_line
        assert not screen.query_one("#preparation-error").display
        assert selection.selected == [task_event()["request"]["questions"][0]["options"][0]]
        assert str(summary.content).endswith("已选 1 组" if language == "zh" else "selected 1")
        await pilot.press("down", "space")
        assert len(selection.selected) == 2
        assert str(summary.content).endswith("已选 2 组" if language == "zh" else "selected 2")
        await pilot.press("up", "space")
        await pilot.pause()
        assert isinstance(app.focused, Checkbox)
        assert "✓" not in checkbox_text(app.focused)
        assert len(selection.selected) == 1
        assert str(summary.content).endswith("已选 1 组" if language == "zh" else "selected 1")
        for selector in ("#preparation-select-all", "#preparation-apply", "#preparation-cancel"):
            button = screen.query_one(selector, Button)
            assert button.region.bottom <= size[1]
            assert cell_len(str(button.label)) <= button.content_region.width
            assert app.get_widget_at(button.region.x + 1, button.region.y + 1)[0] is button
        await pilot.click("#preparation-select-all")
        assert len(selection.selected) == 2
        assert str(summary.content).endswith("已选 2 组" if language == "zh" else "selected 2")
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
            grid = card.query_one(".task-differences", FactGrid)
            assert card.region.height >= 3
            assert text.region.right <= screen.query_one(VerticalScroll).content_region.right
            values = list(grid.query_one(".fact-grid").query(FactValue))
            raw = next(value for value in values if value.row.label == "future_parameter.nested[0]")
            assert "中文 [literal] <markup>" in str(raw.tooltip)
            assert "…" in raw.render().plain
            assert any(value.row.label == "future_parameter.nested[1]" for value in grid.query(FactValue))
            assert grid.query_one(".fact-more", Collapsible).collapsed
        for previous, current in zip(cards, cards[1:], strict=False):
            assert previous.region.bottom <= current.region.y
        cards[0].query_one(".task-choice-description", Static).scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(cards[0].query_one(".task-choice-description", Static))
        assert len(selection.selected) == 1
        for _ in cards[1:]:
            await pilot.press("down", "space")
        assert len(selection.selected) == 6
        assert app.focused is cards[-1].query_one(Checkbox)
        scroll = screen.query_one(VerticalScroll)
        scroll.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        for selector in ("#preparation-select-all", "#preparation-apply", "#preparation-cancel"):
            button = screen.query_one(selector, Button)
            assert app.get_widget_at(button.region.x + 1, button.region.y + 1)[0] is button
        await pilot.click("#preparation-apply")
        assert replies[0]["answers"]["tasks"] == [profile["task_id"] for profile in profiles]


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_details_and_copy_never_toggle_or_submit_a_task(tmp_path, language):
    profiles = long_profiles()
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    replies = []
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#ui-language", Select).value = language
        await pilot.pause()
        screen = PreparationScreen(task_event(profiles), respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        selection = screen.query_one(TaskChoices)
        card = selection.query(TaskChoice).first()
        await pilot.press("space")
        selected = [card.group.task_id]
        assert selection.selected == selected

        common_details = selection.query_one(".task-common .fact-more", Collapsible)
        common_title = common_details.query_one("CollapsibleTitle")
        common_title.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(common_title)
        assert not common_details.collapsed
        assert selection.selected == selected
        await pilot.press("enter")
        assert common_details.collapsed

        details = card.query_one(".task-differences .fact-more", Collapsible)
        title = details.query_one("CollapsibleTitle")
        title.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(title)
        assert not details.collapsed
        assert selection.selected == selected
        await pilot.press("down", "up")
        assert app.focused is title
        await pilot.press("tab")
        assert app.focused is details.query(FactValue).first()
        await pilot.press("c")
        assert app.clipboard == card.group.task_id
        values = card.query_one(".task-differences .fact-grid").query(FactValue)
        raw = next(value for value in values if value.row.label == "future_parameter.nested[0]")
        raw.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(raw)
        assert app.focused is raw
        assert selection.selected == selected
        await pilot.press("down", "up")
        assert app.focused is raw
        original = next(profile for profile in profiles if profile["task_id"] == card.group.task_id)
        expected = json.dumps(original["execution_profile"]["signature"]["future_parameter"]["nested"][0],
                              ensure_ascii=False)
        await pilot.press("c")
        assert app.clipboard == expected
        assert selection.selected == selected
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, FactDetailScreen)
        assert expected in [str(item.content) for item in app.screen.query("#fact-detail-body Static")]
        app.copy_to_clipboard("sentinel")
        assert await pilot.click("#fact-copy")
        assert app.clipboard == expected
        assert selection.selected == selected
        assert replies == []
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is screen
        assert app.focused is raw
        assert selection.selected == selected
        assert replies == []


async def test_initial_selection_and_title_click_keep_stable_group_ids(tmp_path):
    event = task_event()
    options = event["request"]["questions"][0]["options"]
    event["request"]["questions"][0]["value"] = [options[-1], "stale-group"]
    app = AcprofTui(RunConfig.smoke("google-t5/t5-small"), settings_path=tmp_path / "settings.json")
    replies = []
    async with app.run_test(size=(80, 24)) as pilot:
        screen = PreparationScreen(event, respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        selection = screen.query_one(TaskChoices)
        cards = list(selection.query(TaskChoice))
        assert selection.selected == [options[-1]]
        assert app.focused is cards[0].query_one(Checkbox)
        title = cards[-1].query_one(".task-choice-description", Static)
        title.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(title)
        assert selection.selected == []
        assert app.focused is cards[-1].query_one(Checkbox)
        await pilot.press("down")
        assert app.focused is cards[-1].query_one(Checkbox)
        await pilot.press("up", "space", "up")
        assert app.focused is cards[0].query_one(Checkbox)
        assert selection.selected == [options[0]]
        context = screen.query_one("#preparation-fields", FactGrid)
        context_section = context.query_one(".fact-section", Collapsible)
        assert context_section.collapsed
        context_title = context_section.query_one("CollapsibleTitle")
        context_title.focus()
        context_title.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        await pilot.press("enter", "tab")
        assert not context_section.collapsed
        assert isinstance(app.focused, FactValue)
        assert app.focused.row.value == event["request"]["fields"]["Model"]
        await pilot.press("tab", "c")
        assert isinstance(app.focused, FactValue)
        assert app.focused.row.value == event["request"]["fields"]["Revision"]
        assert app.clipboard == event["request"]["fields"]["Revision"]
        assert selection.selected == [options[0]]
        screen.query_one("#preparation-apply", Button).focus()
        await pilot.press("enter")
        assert replies == [{"action": "answer", "answers": {"tasks": [options[0]]}}]


@pytest.mark.parametrize("language", ["zh", "en"])
def test_task_names_use_localized_purposes_and_preserve_unknown_ids(language):
    tr = partial(translate, language=language)
    view = build_task_selection(t5_profiles())
    labels = {group.purposes[0].kind: group_label(group, tr) for group in view.groups}
    assert labels["summarization"] == ("文本摘要" if language == "zh" else "Summarization")
    assert labels["translation"] == ("文本翻译 · 3 种语言方向" if language == "zh" else "Translation · 3 language directions")
    mixed = build_task_selection(t5_profiles(generation_override={"no_repeat_ngram_size": 0,
                                                                "length_penalty": 1.0})).groups[0]
    assert group_label(mixed, tr) == labels["summarization"] + " / " + labels["translation"]
    member = mixed.profile["semantic_tasks"][0]
    member.update(kind="future-kind", task_id="future-task")
    mixed.profile["execution_profile"]["representative_task_id"] = "future-task"
    unknown = build_task_selection([mixed.profile]).groups[0]
    assert "future-task" in group_label(unknown, tr)
