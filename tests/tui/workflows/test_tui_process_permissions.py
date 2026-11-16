"""Process permission recovery uses cached preflight, not a second checker."""
from dataclasses import replace

import pytest
from textual.widgets import Button, ContentSwitcher, Input, Static

from acprof.experiments.configuration import RunConfig
from acprof.tui.run.diagnostics import PreflightCheck
from tests.support.tui_fixtures import AcprofTui


async def test_process_permissions_have_inline_recovery_entry(tmp_path):
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='1', mems='1'),
                    settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        app.preflight.checks = (PreflightCheck('Process resources', 'fail', 'fixture denied', check_id='process_resources'),)
        app.preflight.refresh_preflight_state()
        button = app.query_one('#process-resource-open', Button)
        assert button.display
        button.focus()
        await pilot.press('enter')
        await pilot.pause()
        assert app.query_one('#experiment-pages', ContentSwitcher).current == 'advanced-form'
        assert app.query_one('#process-resource-help').display
        assert app.query_one('#start-run', Button).disabled
        assert app.query_one('#cpus', Input).value == '1'
        assert not app.query_one('#preflight-run-reason', Static).display


def denied_check():
    from acprof.host.runtimes.cgroups import ResourceDiagnostic
    diagnostic = ResourceDiagnostic(code='permission_denied', version='v2', current='/fixture/session.scope',
                                    required=('cpu', 'memory'), detail='PermissionError: fixture denied',
                                    guidance=('Delegate=yes',))
    return PreflightCheck('Process resources', 'fail', diagnostic.summary, check_id='process_resources',
                          resource_diagnostic=diagnostic)


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
        entry = app.query_one('#process-resource-open', Button)
        assert ('待授权' if language == 'zh' else 'Authorization') in str(entry.label)
        entry.focus()
        await pilot.press('enter')
        await pilot.pause()
        panel = app.query_one('#process-resource-callout', Callout)
        buttons = list(panel.query_one(ActionButtons).query(Button))
        details = buttons[1]
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
        assert details.region.bottom <= app.query_one('#advanced-form').content_region.bottom
        inherit = app.query_one('#process-inherited-resources', Button)
        assert any(
            ('不再承诺数值配额' if language == 'zh' else 'without a numeric quota') in str(hint.content)
            for hint in app.query_one('#advanced-form').query('.page-hint')
        )
        inherit.scroll_visible(immediate=True, force=True)
        await pilot.pause()
        inherit.focus()
        await pilot.press('enter')
        await pilot.pause()
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
            app.query_one('#process-resource-open', Button).focus()
            await pilot.press('enter')
            await pilot.pause()
            retry = app.query_one('#process-resource-callout').query_one(ActionButtons).query(Button).first()
            retry.scroll_visible(immediate=True, force=True)
            await pilot.pause()
            assert await pilot.click(retry)
            await wait_for_condition(pilot, lambda: resource_state(app.preflight) == 'repairing', description='repair attempt', timeout=10)
            assert app.query_one('#start-run', Button).disabled
            assert app.query_one('#process-inherited-resources', Button).disabled
        finally:
            ready.set()
        await wait_for_condition(pilot, lambda: not app.preflight.check_running and resource_state(app.preflight) == 'ready', description='grant and recheck', timeout=10)
        assert len(attempts) == 2
        assert app.query_one('#cpus', Input).value == app.query_one('#mems', Input).value == '1'
        assert not app.preflight.preflight_run_reason()


async def test_unselected_inventory_failures_are_not_requested_and_do_not_block_process(tmp_path):
    from acprof.tui.ui.environment import display_checks
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='auto', mems='auto'),
                    settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        token = object()
        app.preflight.check_request = token
        app.preflight.check_config = app.preflight.preflight_config()
        inventory = [PreflightCheck(name, 'fail', 'unavailable', capability_status='error', check_id=name)
                     for name in ('docker', 'ncu', 'nsys', 'massif', 'torch')]
        app.preflight.show_quick_check([], '', token, environment_checks=inventory)
        await pilot.pause()
        assert all(row.status == 'fail' for row in app.preflight.environment_checks)
        assert all(row.status == 'not_requested' for row in display_checks(app.preflight)
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
