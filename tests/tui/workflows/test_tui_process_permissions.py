"""Process permission recovery uses cached preflight, not a second checker."""
from dataclasses import replace

import pytest
from textual.widgets import Button, Collapsible, ContentSwitcher, Input, Static

from acprof.experiments.configuration import RunConfig
from acprof.tui.run.diagnostics import PreflightCheck
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition


async def open_resource_issue(app, pilot):
    from acprof.tui.ui.structured import ActionButtons
    app.preflight.panel.show(pin=True, group='error')
    await wait_for_condition(pilot, lambda: bool(app.query('#preflight-signal-process_resources')),
                             timeout=3, description='Process issue')
    app.query_one('#preflight-signal-process_resources').focus()
    await pilot.press('enter')
    button = list(app.query_one('#preflight-callout').query_one(ActionButtons).query(Button))[1]
    button.focus()
    await pilot.press('enter')
    await wait_for_condition(pilot, lambda: bool(app.screen.query('#issue-callout')),
                             timeout=3, description='Process diagnostic mounted')


async def test_process_permissions_use_global_entry_without_inline_panel(tmp_path):
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='1', mems='1'),
                    settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        app.preflight.checks = (denied_check(),)
        app.preflight.refresh_preflight_state()
        assert not app.query('#process-resource-help, #process-resource-open')
        await open_resource_issue(app, pilot)
        assert app.screen.id == 'process-resource-screen'
        assert app.query_one('#experiment-pages', ContentSwitcher).current == 'run-form'
        assert app.query_one('#start-run', Button).disabled
        assert app.query_one('#cpus', Input).value == '1'
        assert app.query_one('#preflight-run-reason', Button).display


def denied_check():
    from acprof.host.runtimes.cgroups import ResourceDiagnostic
    diagnostic = ResourceDiagnostic(code='permission_denied', version='v2', current='/fixture/session.scope',
                                    required=('cpu', 'memory'), detail='PermissionError: fixture denied',
                                    guidance=('Delegate=yes',))
    return PreflightCheck('Process resources', 'fail', diagnostic.summary, check_id='process_resources',
                          resource_diagnostic=diagnostic)


@pytest.mark.parametrize(('code', 'action_text', 'next_text'), [
    ('permission_denied', '授权步骤', '委派'),
    ('delegation_missing', '授权步骤', '委派'),
    ('parent_limit', '上级配额', '上级'),
    ('controller_missing', '系统支持', '控制器'),
    ('unsupported', '系统支持', 'cgroup v2'),
    ('configuration_error', '详细原因', '路径'),
    ('verification_failed', '详细原因', '归属'),
])
async def test_process_recovery_guidance_follows_failure_cause(tmp_path, code, action_text, next_text):
    from acprof.tui.ui.structured import ActionButtons, Callout
    base = denied_check()
    diagnostic = replace(base.resource_diagnostic, code=code)
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='1', mems='1'),
                    settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(120, 30)) as pilot:
        app.preflight.checks = (replace(base, resource_diagnostic=diagnostic,
                                       detail=diagnostic.summary),)
        app.preflight.refresh_preflight_state()
        await open_resource_issue(app, pilot)
        callout = app.screen.query_one('#issue-callout', Callout)
        buttons = list(callout.query_one(ActionButtons).query(Button))
        assert buttons[0].variant == 'primary'
        assert action_text in str(buttons[0].label)
        assert next_text in app.tr(callout.content.next_step)
        assert '重新检查' in str(buttons[1].label)
        assert app.screen.query_one('#issue-inherited-fallback', Collapsible).collapsed
        assert app.query_one('#cpus', Input).value == '1'


async def test_process_permission_details_can_be_copied_without_changing_quotas(tmp_path, monkeypatch):
    from acprof.tui.ui.structured import ActionButtons, FactDetailScreen
    base = denied_check()
    diagnostic = replace(base.resource_diagnostic, guidance=('由管理员手动执行', 'sudo busctl call fixture-command'))
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='1', mems='1'),
                    settings_path=tmp_path / 'settings.json')
    copied = []
    monkeypatch.setattr(app, 'copy_to_clipboard', copied.append)
    async with app.run_test(size=(120, 30)) as pilot:
        app.preflight.checks = (replace(base, resource_diagnostic=diagnostic),)
        app.preflight.refresh_preflight_state()
        await open_resource_issue(app, pilot)
        action = list(app.screen.query_one('#issue-callout').query_one(ActionButtons).query(Button))[0]
        action.scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert await pilot.click(action)
        await pilot.pause()
        assert isinstance(app.screen, FactDetailScreen)
        assert await pilot.click('#fact-copy')
        assert 'sudo busctl call fixture-command' in copied[0]
        assert app.query_one('#cpus', Input).value == app.query_one('#mems', Input).value == '1'


@pytest.mark.parametrize('size', [(80, 24), (120, 30), (150, 45)])
@pytest.mark.parametrize('language', ['zh', 'en'])
async def test_details_keyboard_inherit_and_resize(tmp_path, size, language):
    from acprof.tui.ui.structured import ActionButtons, Callout, FactDetailScreen
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='1', mems='1'),
                    settings_path=tmp_path / 'settings.json')
    app.configuration.preferences = replace(app.configuration.preferences, language=language,
                                            theme='acprof-light' if language == 'en' else 'acprof-graphite')
    async with app.run_test(size=size) as pilot:
        app.preflight.checks = (denied_check(),)
        app.preflight.refresh_preflight_state()
        assert app.query_one('#cpus-error', Static).display
        assert not app.query_one('#mems-error', Static).display
        await open_resource_issue(app, pilot)
        panel = app.screen.query_one('#issue-callout', Callout)
        buttons = list(panel.query_one(ActionButtons).query(Button))
        details = buttons[0]
        assert details.variant == 'primary'
        assert ('授权步骤' if language == 'zh' else 'authorization steps') in str(details.label)
        details.scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert details.region.width > 0
        assert details.region.height > 0
        assert await pilot.click(details)
        await pilot.pause()
        assert isinstance(app.screen, FactDetailScreen)
        assert 'Delegate=yes' in app.screen.row.value
        assert 'PermissionError' in app.screen.row.value
        await pilot.press('escape')
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert details.region.bottom <= app.screen.size.height
        fallback = app.screen.query_one('#issue-inherited-fallback', Collapsible)
        assert fallback.collapsed
        fallback.scroll_visible(immediate=True, force=True)
        await pilot.pause()
        fallback.query_one('CollapsibleTitle').focus()
        await pilot.press('enter')
        await pilot.pause()
        assert not fallback.collapsed
        inherit = app.screen.query_one('#process-inherited-resources', Button)
        assert any(
            ('不再承诺数值配额' if language == 'zh' else 'without a numeric quota') in str(hint.content)
            for hint in app.screen.query('#issue-inherit-note')
        )
        inherit.scroll_visible(immediate=True, force=True)
        await pilot.pause()
        inherit.focus()
        await pilot.press('enter')
        await pilot.pause()
        assert app.query_one('#cpus', Input).value == '1'
        assert await pilot.click('#confirm-yes')
        await wait_for_condition(pilot, lambda: not app.query_one('#cpus-error', Static).display,
                                 timeout=3, description='confirmed inheritance validated')
        assert app.query_one('#cpus', Input).value == 'auto'
        assert app.query_one('#mems', Input).value == 'auto'
        assert not app.query_one('#cpus-error', Static).display


async def test_real_worker_retry_rechecks_and_keeps_numeric_quotas(tmp_path, monkeypatch):
    import threading

    from acprof.tui.app import AcprofTui as ProductionTui
    from acprof.tui.run.resource_view import resource_state
    from acprof.tui.ui.structured import ActionButtons
    from tests.support.tui_waits import wait_for_condition

    ready = threading.Event()
    attempts = []
    def check(config, **kwargs):
        attempts.append(config)
        if len(attempts) == 1:
            return [denied_check()]
        if not ready.wait(10):
            raise TimeoutError('test did not release recheck')
        return [replace(denied_check(), status='ok', detail='verified',
                        resource_diagnostic=replace(denied_check().resource_diagnostic, code='ready'))]
    monkeypatch.setattr('acprof.tui.run.preflight_controller.quick_preflight', check)
    app = ProductionTui(replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='1', mems='1'),
                        settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(120, 30)) as pilot:
        try:
            await wait_for_condition(pilot, lambda: app.preflight.check_completed, description='initial failure', timeout=10)
            assert resource_state(app.preflight) == 'authorization'
            await open_resource_issue(app, pilot)
            retry = list(app.screen.query_one('#issue-callout').query_one(ActionButtons).query(Button))[1]
            retry.scroll_visible(immediate=True, force=True)
            await pilot.pause()
            assert await pilot.click(retry)
            await wait_for_condition(pilot, lambda: resource_state(app.preflight) == 'repairing', description='repair attempt', timeout=10)
            assert app.query_one('#start-run', Button).disabled
            assert app.screen.query_one('#process-inherited-resources', Button).disabled
        finally:
            ready.set()
        await wait_for_condition(pilot, lambda: not app.preflight.check_running and resource_state(app.preflight) == 'ready', description='grant and recheck', timeout=10)
        assert len(attempts) == 2
        assert app.query_one('#cpus', Input).value == app.query_one('#mems', Input).value == '1'
        assert not app.preflight.preflight_run_reason()


async def test_unselected_inventory_failures_remain_advisory_and_do_not_block_process(tmp_path):
    from acprof.tui.ui.environment import display_checks
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='auto', mems='auto'),
                    settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        token = object()
        app.preflight.check_request = token
        app.preflight.check_config = app.preflight.preflight_config()
        inventory = [PreflightCheck(name, 'fail', 'unavailable', capability_status='error', check_id=name)
                     for name in ('docker', 'ncu', 'nsys', 'massif', 'torch')]
        app.preflight.show_quick_check([PreflightCheck('Runtime', 'ok', 'Process', runtime_kind='process', check_id='runtime')], '', token, environment_checks=inventory)
        await pilot.pause()
        assert all(row.status == 'fail' for row in app.preflight.environment_checks)
        assert all(row.status == 'warn' for row in display_checks(app.preflight)
                   if row.check_id in {'docker', 'ncu', 'nsys', 'massif', 'torch'})
        assert not app.preflight.preflight_run_reason()
        assert not app.query_one('#start-run', Button).disabled


async def test_measurement_rejects_recheck_and_inherited_action(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from acprof.tui.run.progress import ProgressSnapshot
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='1', mems='1'),
                    settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)):
        worker = Mock()
        monkeypatch.setattr(app.preflight, 'execute_quick_check', worker)
        app.execution.session.snapshot = ProgressSnapshot(measurement_active=True)
        app.preflight.quick_check()
        app.configuration.use_process_inherited_resources()
        worker.assert_not_called()
        assert app.query_one('#cpus', Input).value == app.query_one('#mems', Input).value == '1'
