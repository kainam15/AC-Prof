"""Issue inspection is passive; navigation and permission changes are explicit."""
from dataclasses import replace

import pytest
from textual.widgets import Button, ContentSwitcher, Input, TabbedContent

from acprof.experiments.configuration import RunConfig
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.settings.environment import EnvironmentSettingsScreen
from acprof.tui.ui.dialogs import ConfirmActionScreen
from acprof.tui.ui.structured import ActionButtons
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition
from tests.tui.workflows.test_tui_process_permissions import denied_check


def process_app(tmp_path):
    return AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='1', mems='1'),
                     settings_path=tmp_path / 'settings.json')


async def test_blocked_admission_preserves_page_focus_and_scroll(tmp_path):
    app = process_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        app.preflight.checks = (denied_check(),)
        app.preflight.refresh_preflight_state()
        field = app.query_one('#input-scales', Input)
        field.focus()
        field.scroll_visible(immediate=True)
        await pilot.pause()
        page = app.query_one('#run-form')
        position = page.scroll_offset
        assert not app.preflight.allow_collection()
        await pilot.pause()
        assert app.query_one('#experiment-pages', ContentSwitcher).current == 'run-form'
        assert app.focused is field
        assert page.scroll_offset == position
        assert app.query_one('#start-run', Button).disabled


async def test_selecting_issue_only_inspects_then_explicit_action_opens_diagnostic(tmp_path):
    app = process_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        app.preflight.checks = (denied_check(),)
        app.preflight.refresh_preflight_state()
        base = app.screen
        app.preflight.panel.show(pin=True, group='error')
        await wait_for_condition(pilot, lambda: bool(app.query('#preflight-signal-process_resources')),
                                 timeout=3, description='resource issue rendered')
        assert await pilot.click('#preflight-signal-process_resources')
        assert app.screen is base
        assert app.query_one('#experiment-pages', ContentSwitcher).current == 'run-form'
        action = list(app.query_one('#preflight-callout').query_one(ActionButtons).query(Button))[1]
        action.focus()
        await pilot.press('enter')
        await wait_for_condition(pilot, lambda: app.screen is not base,
                                 timeout=3, description='dedicated Process diagnostic')
        assert app.screen.id == 'process-resource-screen'
        assert app.query_one('#experiment-pages', ContentSwitcher).current == 'run-form'
        await pilot.press('escape')
        assert app.screen is base


@pytest.mark.parametrize(('identifier', 'tool'), [('cpu_instructions', 'perf'), ('capture_permissions', 'tcpdump')])
async def test_permission_issue_focuses_exact_target(tmp_path, monkeypatch, identifier, tool):
    monkeypatch.setattr('acprof.tui.app.PROJECT_DIR', tmp_path)
    app = AcprofTui(RunConfig.smoke('fixture/model'), settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        row = PreflightCheck(tool, 'fail', 'permission denied', check_id=identifier)
        app.preflight.checks = (row,)
        app.preflight.refresh_preflight_state()
        app.preflight.panel.open_issue(row)
        await wait_for_condition(pilot, lambda: isinstance(app.screen, EnvironmentSettingsScreen),
                                 timeout=3, description='permissions screen')
        await wait_for_condition(pilot, lambda: bool(app.screen.query('#permission-' + tool))
                                 and app.focused is app.screen.query_one('#permission-' + tool),
                                 timeout=3, description='specific permission focus')
        assert app.screen.query_one('#environment-tabs', TabbedContent).active == 'permissions-tab'
        assert app.preflight.status.counts['error'] == 1


async def test_unmapped_error_does_not_open_unrelated_permissions(tmp_path, monkeypatch):
    monkeypatch.setattr('acprof.tui.app.PROJECT_DIR', tmp_path)
    app = process_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        row = PreflightCheck('Unknown', 'fail', 'fixture', check_id='unknown_check')
        app.preflight.checks = (row,)
        app.preflight.panel.show(pin=True, group='error')
        await pilot.pause()
        base = app.screen
        app.preflight.panel.open_issue(row)
        await pilot.pause()
        assert app.screen is base
        assert app.preflight.panel.display


@pytest.mark.parametrize('field_id', ['cpus', 'mems'])
async def test_resource_configuration_error_focuses_field_and_clears_on_repair(tmp_path, field_id):
    app = process_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        control = app.query_one('#' + field_id, Input)
        control.value = 'invalid'
        await wait_for_condition(pilot, lambda: not app.configuration.validation_pending
                                 and app.preflight.status.counts['error'] == 1,
                                 timeout=3, description='resource configuration blocker')
        item = next(item for item in app.preflight.status.blockers if item.group == 'error')
        app.activate_tab('settings-tab')
        app.preflight.panel.open_issue(item.check)
        await wait_for_condition(pilot, lambda: app.focused is control,
                                 timeout=3, description='exact resource field focused')
        assert app.query_one('#experiment-pages', ContentSwitcher).current == 'run-form'
        control.value = '1'
        await wait_for_condition(pilot, lambda: not app.configuration.validation_pending
                                 and app.preflight.status.counts['error'] == 0,
                                 timeout=3, description='repaired resource removed from issue list')


async def test_inherited_limits_require_confirmation_and_cancel_preserves_values(tmp_path):
    app = process_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        app.configuration.use_process_inherited_resources()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmActionScreen)
        assert app.query_one('#cpus', Input).value == '1'
        assert app.query_one('#mems', Input).value == '1'
        await pilot.press('escape')
        assert app.query_one('#cpus', Input).value == '1'
        app.configuration.use_process_inherited_resources()
        await pilot.pause()
        assert await pilot.click('#confirm-yes')
        await wait_for_condition(pilot, lambda: app.query_one('#cpus', Input).value == 'auto',
                                 timeout=3, description='confirmed inherited limits')
        assert app.query_one('#mems', Input).value == 'auto'


async def test_resolved_auto_runtime_has_no_duplicate_banner(tmp_path):
    app = AcprofTui(RunConfig.smoke('fixture/model'), settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)):
        app.preflight.checks = (PreflightCheck('Runtime', 'ok', 'Docker', runtime_kind='docker', check_id='runtime'),)
        app.preflight.refresh_preflight_state()
        assert not app.query('#runtime-status')
        assert not app.query_one('#preflight-run-reason').display


@pytest.mark.parametrize('result_status', ['ok', 'fail'])
async def test_permission_recheck_updates_global_evidence(tmp_path, monkeypatch, result_status):
    monkeypatch.setattr('acprof.tui.app.PROJECT_DIR', tmp_path)
    row = PreflightCheck('perf', 'fail', 'permission denied', check_id='cpu_instructions')
    result = replace(row, status=result_status, detail='rechecked: ' + result_status)
    monkeypatch.setattr('acprof.tui.settings.environment.quick_preflight', lambda *a, **kw: [result])
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='docker', profiling_mode='full'),
                    settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        app.preflight.checks = (row,)
        app.configuration.open_environment_settings(permissions=True)
        await wait_for_condition(pilot, lambda: bool(app.screen.query('#check-environment-permissions')),
                                 timeout=3, description='permission actions mounted')
        screen = app.screen
        screen.query_one('#check-environment-permissions').focus()
        await pilot.press('enter')
        await wait_for_condition(pilot, lambda: 'rechecked:' in str(screen.query_one('#permission-report').content),
                                 timeout=3, description='permission report complete')
        item = next(item for item in app.preflight.status.items if item.check.check_id == 'cpu_instructions')
        assert item.check.detail == result.detail
        assert item.check.status == result_status
        assert app.preflight.status.can_start == (result_status == 'ok')


@pytest.mark.parametrize('method', ['keyboard', 'mouse'])
async def test_field_navigation_returns_to_original_page_focus_and_scroll(tmp_path, method):
    app = process_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one('#model', Input).value = ''
        await wait_for_condition(pilot, lambda: app.preflight.status.counts['error'] == 1,
                                 timeout=3, description='invalid model')
        app.configuration.show_run_form('advanced-form')
        app.activate_tab('settings-tab')
        body = app.query_one('#settings-body')
        body.scroll_end(animate=False, immediate=True)
        origin = app.query_one('#save-ui-settings')
        origin.focus(scroll_visible=False)
        await pilot.pause()
        position = body.scroll_offset
        app.preflight.panel.show(pin=True, group='error')
        await wait_for_condition(pilot, lambda: bool(app.query('#preflight-signal-config_model')),
                                 timeout=3, description='model issue rendered')
        app.query_one('#preflight-signal-config_model').focus()
        await pilot.press('enter')
        button = list(app.query_one('#preflight-callout').query_one(ActionButtons).query(Button))[1]
        button.focus()
        await pilot.press('enter')
        await wait_for_condition(pilot, lambda: app.focused is app.query_one('#model'),
                                 timeout=3, description='model focused')
        assert app.query_one('#status-return').display
        if method == 'keyboard':
            await pilot.press('alt+left')
        else:
            assert await pilot.click('#status-return')
        await wait_for_condition(pilot, lambda: app.focused is origin,
                                 timeout=3, description='origin focus restored')
        assert app.query_one('#main-tabs', TabbedContent).active == 'settings-tab'
        assert app.query_one('#experiment-pages', ContentSwitcher).current == 'advanced-form'
        assert str(app.query_one('#open-run-settings', Button).label) == app.tr('返回基本配置')
        assert body.scroll_offset == position
        assert not app.query_one('#status-return').display


@pytest.mark.parametrize(('identifier', 'screen_id', 'callout', 'details'), [
    ('docker', 'docker-repair-screen', '#docker-summary', '#docker-technical'),
    ('cpu_energy', 'issue-diagnostic-screen', '#issue-callout', '#issue-callout .callout-details'),
], ids=['docker', 'cpu_energy'])
async def test_diagnostics_show_specific_guidance_without_permission_setup(tmp_path, identifier, screen_id, callout, details):
    from textual.widgets import Collapsible

    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='docker', profiling_mode='full'),
                    settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        row = PreflightCheck(identifier, 'fail', 'fixture diagnostic', check_id=identifier)
        app.preflight.checks = (row,)
        app.preflight.panel.open_issue(row)
        await wait_for_condition(pilot, lambda: app.screen.id == screen_id and bool(app.screen.query(callout)),
                                 timeout=3, description='specific diagnostic mounted')
        assert not app.screen.query('#permission-perf')
        assert app.screen.query_one(details, Collapsible).collapsed
        assert app.screen.query_one(callout + ' .callout-next').content


async def test_permission_check_exception_keeps_required_capability_unverified(tmp_path, monkeypatch):
    monkeypatch.setattr('acprof.tui.app.PROJECT_DIR', tmp_path)
    def failed(*args, **kwargs):
        raise OSError('fixture permission probe failed')
    monkeypatch.setattr('acprof.tui.settings.environment.quick_preflight', failed)
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='docker', profiling_mode='full'),
                    settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        app.preflight.checks = (PreflightCheck('perf', 'ok', 'old evidence', check_id='cpu_instructions'),)
        app.configuration.open_environment_settings(permissions=True)
        await wait_for_condition(pilot, lambda: bool(app.screen.query('#check-environment-permissions')),
                                 timeout=3, description='permission actions')
        screen = app.screen
        screen.query_one('#check-environment-permissions').focus()
        await pilot.press('enter')
        await wait_for_condition(pilot, lambda: 'fixture permission probe failed' in str(screen.query_one('#permission-report').content),
                                 timeout=3, description='actual failure reason')
        assert not app.preflight.status.can_start
        assert app.preflight.status.counts['pending'] > 0
        assert not any(item.check.status == 'ok' for item in app.preflight.status.items
                       if item.check.check_id == 'cpu_instructions')
