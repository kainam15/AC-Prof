"""Runtime selection stays in the basic form and reuses configuration preflight."""
from dataclasses import replace
from unittest.mock import Mock

import pytest
from textual.widgets import Button, ContentSwitcher, Select, Static
from textual.widgets._select import SelectOverlay

from acprof.experiments.configuration import RunConfig
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.run.plan_panel import RunPlanPanel
from acprof.tui.run.progress import ProgressSnapshot
from acprof.tui.run.resource_view import resource_state
from acprof.tui.settings.preferences import TuiSettings, save_settings
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition


async def choose_runtime(app, pilot, runtime):
    control = app.query_one('#run-form #runtime', Select)
    control.scroll_visible(animate=False, immediate=True)
    control.focus()
    await pilot.pause()
    assert control.region.width > 0
    assert control.region.bottom <= app.query_one('#run-form').content_region.bottom
    await pilot.press('enter', 'home', *(['down'] * ('docker', 'process').index(runtime)), 'enter')
    await wait_for_condition(
        pilot, lambda: f'--runtime {runtime}' in str(app.query_one('#command-preview', Static).content),
        description=f'{runtime} command preview', timeout=3,
    )
    assert control.value == runtime


def complete_check(app, checks):
    token = object()
    app.preflight.check_request = token
    app.preflight.check_config = app.preflight.preflight_config()
    app.preflight.show_quick_check(checks, '', token, environment_checks=checks)


@pytest.mark.parametrize(('language', 'size', 'saved_runtime'), [
    ('zh', (80, 24), 'docker'), ('en', (120, 30), 'process'), ('zh', (150, 45), 'process'),
])
async def test_basic_runtime_selection_navigation_and_restart(tmp_path, monkeypatch, language, size, saved_runtime):
    path = tmp_path / 'settings.json'
    config = replace(RunConfig.smoke('fixture/model'), cpus='1,2', mems='2,4', gpus='off,on', cpuset_cpus='0-3')
    app = AcprofTui(config, settings_path=path)
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size) as pilot:
        check = Mock()
        monkeypatch.setattr(app.preflight, 'execute_quick_check', check)
        control = app.query_one('#run-form #runtime', Select)
        assert app.configuration.collect_config() == replace(config, runtime='docker')
        assert control.query_one(SelectOverlay).option_count == 2
        assert tuple(value for _, value in control._options) == ('docker', 'process')
        assert not app.query('#runtime-status')
        assert not app.query('#reset-runtime')
        complete_check(app, [PreflightCheck('Runtime', 'ok', 'Docker', runtime_kind='docker')])
        assert control.value == 'docker'
        assert app.configuration.collect_config().runtime == 'docker'
        assert len(app.query('#runtime')) == 1
        assert not app.query('#advanced-form #runtime')
        controls = list(app.query('#run-form Select'))
        assert controls.index(app.query_one('#model-source')) < controls.index(control)
        assert not app.query('#profiling-mode')
        for runtime in ('process', 'docker'):
            await choose_runtime(app, pilot, runtime)
            view = app.query_one(RunPlanPanel).view
            assert view is not None
            assert view.section.rows[0].value == runtime.title()
            assert app.configuration.collect_config() == replace(config, runtime=runtime)
            assert app.query_one('#start-run', Button).disabled
            assert not app.preflight.config_matches()
            assert await pilot.click('#run-mode-advanced')
            await pilot.pause()
            assert app.query_one('#experiment-pages', ContentSwitcher).current == 'advanced-form'
            assert not control.is_on_screen
            assert await pilot.click('#run-mode-basic')
            await pilot.pause()
            assert app.query_one('#experiment-pages', ContentSwitcher).current == 'run-form'
            assert app.query_one('#runtime', Select).value == runtime
            await wait_for_condition(
                pilot, lambda: app.preflight.check_running,
                description=f'automatic {runtime} preflight', timeout=3,
            )
            checked, token = check.call_args.args
            assert checked.runtime == runtime
            assert not app.preflight.check_inventory
            app.preflight.show_quick_check(
                [PreflightCheck('Runtime', 'ok', runtime.title(), runtime_kind=runtime)], '', token,
            )
            assert app.preflight.config_matches()
            assert not app.query_one('#start-run', Button).disabled
        await choose_runtime(app, pilot, saved_runtime)
        await pilot.resize_terminal(80, 24)
        assert app.configuration.collect_config().runtime == saved_runtime
        app.configuration.preferences = replace(app.configuration.preferences, language='en' if language == 'zh' else 'zh')
        app.configuration.apply_ui_preferences()
        await pilot.pause()
        assert app.configuration.collect_config().runtime == saved_runtime
        await pilot.click('#run-mode-advanced')
        save = app.query_one('#save-run-default', Button)
        save.scroll_visible(animate=False, immediate=True)
        save.focus()
        await pilot.press('enter')
        await wait_for_condition(pilot, path.is_file, description='saved runtime configuration', timeout=3)

    restored = AcprofTui(settings_path=path)
    async with restored.run_test(size=(80, 24)):
        assert restored.query_one('#run-form #runtime', Select).value == saved_runtime
        assert restored.configuration.collect_config() == replace(config, runtime=saved_runtime)


@pytest.mark.parametrize('status', ['ok', 'fail'])
async def test_docker_status_uses_current_preflight_and_becomes_stale(tmp_path, monkeypatch, status):
    config = replace(RunConfig.smoke('fixture/model'), runtime='docker')
    app = AcprofTui(config, settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(120, 30)) as pilot:
        complete_check(app, [PreflightCheck('Docker', status, 'fixture daemon', check_id='docker')])
        assert not app.query('#runtime-status')
        item = next(item for item in app.preflight.status.items if item.check.check_id == 'docker')
        assert item.group == ('success' if status == 'ok' else 'error')
        assert app.query_one('#start-run', Button).disabled == (status == 'fail')
        monkeypatch.setattr(app.preflight, 'execute_quick_check', Mock())
        await choose_runtime(app, pilot, 'process')
        await wait_for_condition(pilot, lambda: app.preflight.check_running,
                                 description='automatic Process preflight', timeout=3)
        assert not app.preflight.config_matches()
        assert app.preflight.selected_runtime == ''
        assert resource_state(app.preflight) == 'checking'
        assert app.query_one('#start-run', Button).disabled


async def test_runtime_edit_rejects_late_preflight_result(tmp_path, monkeypatch):
    app = AcprofTui(RunConfig.smoke('fixture/model'), settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(120, 30)) as pilot:
        complete_check(app, [PreflightCheck('Runtime', 'ok', 'Docker', runtime_kind='docker')])
        assert app.preflight.selected_runtime == 'docker'
        monkeypatch.setattr(app.preflight, 'execute_quick_check', Mock())
        app.preflight.quick_check()
        token = app.preflight.check_request
        await choose_runtime(app, pilot, 'process')
        app.preflight.show_quick_check(
            [PreflightCheck('Runtime', 'ok', 'Docker', runtime_kind='docker')], '', token,
        )
        assert app.preflight.selected_runtime == ''
        assert not app.preflight.config_matches()
        assert app.query_one('#start-run', Button).disabled
        assert app.query_one(RunPlanPanel).view.section.rows[0].value == 'Process'
        assert '--runtime process' in str(app.query_one('#command-preview', Static).content)
        assert not app.query('#process-resource-open')
        complete_check(app, [PreflightCheck('Runtime', 'ok', 'Process', runtime_kind='process')])
        assert not app.query_one('#start-run', Button).disabled
        # Legacy Auto values are converted when applying a form configuration.
        app.configuration.apply_config(replace(app.configuration.collect_config(), runtime='auto'))
        assert app.configuration.collect_config().runtime == 'docker'
        assert app.query_one('#runtime', Select).value == 'docker'
        assert app.preflight.selected_runtime == ''
        complete_check(app, [PreflightCheck('Runtime', 'ok', 'Docker', runtime_kind='docker')])
        assert not app.query('#runtime-status')
        assert app.query_one(RunPlanPanel).view.section.rows[0].value == 'Docker'
        assert '--runtime docker' in str(app.query_one('#command-preview', Static).content)


async def test_preflight_does_not_replace_an_explicit_runtime(tmp_path, monkeypatch):
    app = AcprofTui(RunConfig.smoke('fixture/model'), settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(120, 30)) as pilot:
        monkeypatch.setattr(app.preflight, 'execute_quick_check', Mock())
        control = app.query_one('#runtime', Select)
        assert control.query_one(SelectOverlay).option_count == 2
        for checked_runtime in ('docker', 'process', 'docker'):
            complete_check(app, [PreflightCheck('Runtime', 'ok', checked_runtime, runtime_kind=checked_runtime)])
            await pilot.pause()
            assert control.value == 'docker'
            assert app.configuration.collect_config().runtime == 'docker'
            assert '--runtime docker' in str(app.query_one('#command-preview', Static).content)
        control.scroll_visible(animate=False, immediate=True)
        control.focus()
        await pilot.pause()
        await pilot.press('enter', 'escape')
        assert app.configuration.collect_config().runtime == 'docker'
        await choose_runtime(app, pilot, 'process')
        assert app.configuration.collect_config().runtime == 'process'
        complete_check(app, [PreflightCheck('Runtime', 'ok', 'Process', runtime_kind='process')])
        assert control.value == 'process'
        app.execution.session.snapshot = ProgressSnapshot(measurement_active=True)
        app.set_busy(True)
        assert control.disabled
        assert app.configuration.collect_config().runtime == 'process'
        app.execution.session.snapshot = ProgressSnapshot()
        app.set_busy(False)
        assert not control.disabled


async def test_preflight_does_not_retarget_an_open_runtime_menu(tmp_path, monkeypatch):
    app = AcprofTui(RunConfig.smoke('fixture/model'), settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        monkeypatch.setattr(app.preflight, 'execute_quick_check', Mock())
        app.preflight.checked_config = None
        app.preflight.refresh_preflight_state()
        control = app.query_one('#runtime', Select)
        assert control.value == 'docker'
        control.scroll_visible(animate=False, immediate=True)
        control.focus()
        await pilot.pause()
        await pilot.press('enter', 'home', 'down')
        overlay = control.query_one(SelectOverlay)
        assert overlay.highlighted == 1
        complete_check(app, [PreflightCheck('Runtime', 'ok', 'Process', runtime_kind='process')])
        await pilot.pause()
        assert overlay.highlighted == 1
        await pilot.press('enter')
        await wait_for_condition(pilot, lambda: app.configuration.collect_config().runtime == 'process',
                                 description='commit the option the user highlighted', timeout=3)
        assert control.value == 'process'


async def test_legacy_saved_auto_uses_docker_without_rewriting_settings(tmp_path):
    path = tmp_path / 'settings.json'
    app = AcprofTui(settings_path=path)
    save_settings(path, TuiSettings(run_defaults=RunConfig.smoke('fixture/model')), app.project_dir)
    restored = AcprofTui(settings_path=path)
    async with restored.run_test(size=(80, 24)):
        assert restored.query_one('#runtime', Select).value == 'docker'
        assert restored.configuration.collect_config().runtime == 'docker'
        assert restored.configuration.saved_settings.run_defaults.runtime == 'auto'
