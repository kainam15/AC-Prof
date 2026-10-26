"""Task choice is an explicit, cancellable preparation decision."""
import pytest
from textual.widgets import Button, Select, SelectionList
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.tui.preparation import PreparationScreen


def task_event():
    return {"stage": "resolution", "status": "waiting", "request": {
        "id": 1, "kind": "review", "resolved": False,
        "fields": {"Model": "google-t5/t5-small"},
        "questions": [{"path": "tasks", "kind": "multi", "value": [],
                       "options": ["translation_en_to_de", "translation_en_to_fr", "translation_en_to_ro", "summarization"],
                       "labels": {"translation_en_to_de": "英语 → 德语", "translation_en_to_fr": "英语 → 法语",
                                  "translation_en_to_ro": "英语 → 罗马尼亚语", "summarization": "文本摘要"},
                       "reason": "请选择要采集的任务，可多选或全选。"}]}}


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_multiselect_requires_confirmation_and_supports_all(tmp_path, language):
    app = AcprofTui(RunConfig.smoke("google-t5/t5-small"), settings_path=tmp_path / "settings.json")
    replies = []
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one("#ui-language", Select).value = language
        await pilot.pause()
        screen = PreparationScreen(task_event(), respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        selection = screen.query_one(SelectionList)
        assert selection.selected == []
        await pilot.click("#preparation-apply")
        assert replies == []
        selection.focus()
        await pilot.press("space")
        assert selection.selected == ["translation_en_to_de"]
        all_button = screen.query_one("#preparation-select-all", Button)
        all_button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        await pilot.click("#preparation-select-all")
        assert len(selection.selected) == 4
        await pilot.click("#preparation-apply")
        assert len(replies) == 1
        assert replies[0]["answers"]["tasks"] == task_event()["request"]["questions"][0]["options"]


async def test_escape_cancels_without_task_selection(tmp_path):
    app = AcprofTui(RunConfig.smoke("google-t5/t5-small"), settings_path=tmp_path / "settings.json")
    replies = []
    async with app.run_test(size=(80, 24)) as pilot:
        screen = PreparationScreen(task_event(), respond=replies.append)
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.press("escape")
        assert replies == [{"action": "cancel"}]
