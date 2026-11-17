"""Runtime selection stays in the basic form and reuses configuration preflight."""
from dataclasses import replace
from unittest.mock import Mock

import pytest
from textual.widgets import Button, ContentSwitcher, Select, Static

from acprof.experiments.configuration import RunConfig
from acprof.tui.i18n import message_id
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.run.plan_panel import RunPlanPanel
from acprof.tui.run.resource_view import resource_state
from acprof.tui.ui.structured import StatusBanner
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition


async def choose_runtime(app, pilot, runtime):
    control = app.query_one('#run-form #runtime', Select)
    control.scroll_visible(animate=False, immediate=True)
    control.focus()
    await pilot.pause()
    assert control.region.width > 0
    assert control.region.bottom <= app.query_one('#run-form').content_region.bottom
    await pilot.press('enter', 'home', *(['down'] * ('auto', 'docker', 'process').index(runtime)), 'enter')
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
    ('zh', (80, 24), 'auto'), ('en', (120, 30), 'docker'), ('zh', (150, 45), 'process'),
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
        assert control.value == 'auto'
        assert len(app.query('#runtime')) == 1
        assert not app.query('#advanced-form #runtime')
        controls = list(app.query('#run-form Select'))
        assert controls.index(app.query_one('#model-source')) < controls.index(control)
        assert controls.index(control) < controls.index(app.query_one('#profiling-mode'))
        for runtime in ('docker', 'process', 'auto'):
            await choose_runtime(app, pilot, runtime)
            view = app.query_one(RunPlanPanel).view
            assert view is not None
            assert view.section.rows[0].value == (
                message_id('run.runtime.automatic') if runtime == 'auto' else runtime.title()
            )
            assert app.configuration.collect_config() == replace(config, runtime=runtime)
            assert app.query_one('#start-run', Button).disabled
            assert not app.preflight.config_matches()
            hint = app.query_one('#runtime-status .status-headline', Static).visual.plain
            assert ('cgroup' if runtime == 'process' else 'Docker' if runtime == 'docker' else 'Auto') in hint
            toggle = app.query_one('#open-run-settings', Button)
            assert await pilot.click(toggle)
            await pilot.pause()
            assert app.query_one('#experiment-pages', ContentSwitcher).current == 'advanced-form'
            assert not control.is_on_screen
            assert await pilot.click(toggle)
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
            resolved = 'process' if runtime == 'process' else 'docker'
            app.preflight.show_quick_check(
                [PreflightCheck('Runtime', 'ok', resolved, runtime_kind=resolved)], '', token,
            )
            assert app.preflight.config_matches()
            assert not app.query_one('#start-run', Button).disabled
        await choose_runtime(app, pilot, saved_runtime)
        await pilot.resize_terminal(80, 24)
        await choose_runtime(app, pilot, saved_runtime)
        await pilot.click('#open-run-settings')
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
        banner = app.query_one('#runtime-status', StatusBanner)
        assert banner.summary.level == ('success' if status == 'ok' else 'error')
        assert 'Docker' in app.tr(banner.summary.headline)
        assert app.query_one('#start-run', Button).disabled == (status == 'fail')
        monkeypatch.setattr(app.preflight, 'execute_quick_check', Mock())
        await choose_runtime(app, pilot, 'process')
        await wait_for_condition(pilot, lambda: app.preflight.check_running,
                                 description='automatic Process preflight', timeout=3)
        assert not app.preflight.config_matches()
        assert app.preflight.selected_runtime == ''
        assert resource_state(app.preflight) == 'checking'
        assert app.query_one('#start-run', Button).disabled


async def test_runtime_edit_rejects_late_auto_resolution(tmp_path, monkeypatch):
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
        await choose_runtime(app, pilot, 'auto')
        assert app.preflight.selected_runtime == ''
        assert 'Auto' in app.tr(app.query_one('#runtime-status', StatusBanner).summary.headline)
        complete_check(app, [PreflightCheck('Runtime', 'ok', 'Process', runtime_kind='process')])
        assert not app.query_one('#runtime-status', StatusBanner).display
        assert app.query_one(RunPlanPanel).view.section.rows[0].value == 'Process'
        assert '--runtime auto' in str(app.query_one('#command-preview', Static).content)
