"""Task selections survive the editable TUI form and its public launch actions."""
from dataclasses import replace
from pathlib import Path

import pytest
from textual.widgets import ContentSwitcher, Input, Label, Static

from acprof.experiments.configuration import RunConfig
from acprof.tui.settings.preferences import TuiSettings, UiPreferences, load_settings, save_settings
from acprof.tui.ui.dialogs import ConfirmActionScreen
from tests.support.tui_actions import fill_form
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition


@pytest.mark.parametrize(("size", "language"), [((80, 24), "zh"), ((120, 30), "en")])
async def test_saved_selection_reaches_run_preview_and_probe_confirmation(tmp_path, size, language):
    path = tmp_path / "tui.json"
    project = Path(__file__).resolve().parents[3]
    initial = replace(RunConfig.smoke("google-t5/t5-small"), task="text2text-generation", tasks="summarization")
    save_settings(path, TuiSettings(run_defaults=initial, ui=UiPreferences(language=language)), project)
    app = AcprofTui(settings_path=path)
    selected = "translation_en_to_fr,translation_en_to_de"
    async with app.run_test(size=size) as pilot:
        assert app.configuration.collect_config().tasks == "summarization"
        assert await pilot.click("#run-mode-advanced")
        await wait_for_condition(pilot, lambda: app.query_one("#experiment-pages", ContentSwitcher).current ==
                                 "advanced-form", description="advanced form", timeout=3)
        field = app.query_one("#tasks", Input)
        assert field.value == "summarization"
        labels = {str(label.content) for label in app.query_one("#advanced-form").query(Label)}
        assert ("工作负载 (--tasks)" if language == "zh" else "Workloads (--tasks)") in labels
        assert ("模型接口 (--task)" if language == "zh" else "Interface (--task)") in labels
        await fill_form(pilot, {"#tasks": "translation_en_to_fr, translation_en_to_de"})
        preview = app.query_one("#command-preview", Static)
        await wait_for_condition(pilot, lambda: f"--tasks {selected}" in str(preview.content),
                                 description="run command with both selected tasks", timeout=5)
        assert "--task text2text-generation" in str(preview.content)
        app.configuration.save_run_default()
        saved, warning = load_settings(path, project)
        assert not warning
        assert saved.run_defaults.tasks == selected

        assert await pilot.click("#probe-largest")
        await wait_for_condition(pilot, lambda: isinstance(app.screen, ConfirmActionScreen),
                                 description="minimum-memory probe confirmation", timeout=3)
        pending = app.experiments.pending_launch
        assert pending.kind == "probe"
        assert pending.config.tasks == selected
        assert pending.command[pending.command.index("--tasks") + 1] == selected
        assert pending.command[pending.command.index("--task") + 1] == "text2text-generation"
        await pilot.press("escape")
        await wait_for_condition(pilot, lambda: app.experiments.pending_launch is None,
                                 description="probe cancelled before launch", timeout=3)
        assert not app.is_busy()

    reopened = AcprofTui(settings_path=path)
    async with reopened.run_test(size=size) as pilot:
        assert reopened.query_one("#tasks", Input).value == selected
        assert reopened.configuration.collect_config().tasks == selected
        reopened.configuration.open_config_field("tasks")
        await wait_for_condition(pilot, lambda: reopened.focused is reopened.query_one("#tasks", Input),
                                 description="focus the workload selector", timeout=3)
        await fill_form(pilot, {"#tasks": "not/a/task"})
        await wait_for_condition(pilot, lambda: reopened.query_one("#tasks-error", Static).display,
                                 description="show the invalid task beside the field", timeout=5)
        await fill_form(pilot, {"#tasks": ""})
        await wait_for_condition(pilot, lambda: not reopened.query_one("#tasks-error", Static).display and
                                 "--tasks" not in str(reopened.query_one("#command-preview", Static).content),
                                 description="restore automatic task resolution", timeout=5)
        assert reopened.configuration.collect_config().task == "text2text-generation"


@pytest.mark.parametrize(("saved_model", "expected"), [
    ("google-t5/t5-small", ""),
    ("", "summarization"),
    ("example/new", "summarization"),
])
def test_recent_model_respects_the_scope_of_saved_workloads(tmp_path, saved_model, expected):
    path = tmp_path / "tui.json"
    defaults = replace(RunConfig.smoke(saved_model), tasks="summarization")
    save_settings(path, TuiSettings(run_defaults=defaults, last_model="example/new"), tmp_path)
    config = AcprofTui(settings_path=path).configuration.initial_config
    assert config.model == "example/new"
    assert config.tasks == expected
    assert load_settings(path, tmp_path)[0].run_defaults == defaults
