"""The real header, issue navigation and local hints share one status model."""
from dataclasses import replace

import pytest
from textual.widgets import Button, ContentSwitcher, Input, Static

from acprof.experiments.configuration import RunConfig
from acprof.tui.ui.environment import EnvironmentIndicator
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition
from tests.tui.workflows.test_tui_process_permissions import denied_check


@pytest.mark.parametrize(('size', 'language'), [((80, 24), 'zh'), ((120, 30), 'en'), ((150, 45), 'zh')])
async def test_global_blockers_navigate_with_mouse_and_keyboard(tmp_path, size, language):
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='1', mems='1'),
                    settings_path=tmp_path / 'settings.json')
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size) as pilot:
        app.preflight.checks = (denied_check(),)
        app.query_one('#model', Input).value = ''
        await wait_for_condition(pilot, lambda: app.preflight.status.counts['error'] == 2,
                                 timeout=3, description='configuration and quota blockers')
        assert app.query_one('#environment-error', EnvironmentIndicator).content == '× 2'
        assert app.query_one('#start-run', Button).disabled
        assert app.query_one('#model-error', Static).display
        assert app.query_one('#cpus-error', Static).display
        app.query_one('#environment-error').focus()
        await pilot.press('enter')
        await wait_for_condition(pilot, lambda: bool(app.query('#preflight-signal-config_model')),
                                 timeout=3, description='global error rows')
        row = app.query_one('#preflight-signal-config_model')
        row.scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert await pilot.click(row)
        await wait_for_condition(pilot, lambda: app.focused is app.query_one('#model'),
                                 timeout=3, description='jump to invalid model')
        assert not app.preflight.panel.display
        app.query_one('#environment-error').focus()
        await pilot.press('enter')
        await pilot.pause()
        app.query_one('#preflight-signal-process_resources').focus()
        await pilot.press('enter')
        await wait_for_condition(pilot, lambda: app.query_one('#experiment-pages', ContentSwitcher).current == 'advanced-form',
                                 timeout=3, description='jump to permission recovery')
        assert app.query_one('#process-resource-help').display
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert app.focused is not None
        assert app.focused.region.bottom <= app.screen.size.height


async def test_header_pending_then_success_and_unrequested_are_separate(tmp_path):
    from acprof.tui.run.diagnostics import PreflightCheck
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='auto', mems='auto'),
                    settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        app.preflight.environment_checks = (PreflightCheck('NCU', 'fail', 'not installed', check_id='ncu'),)
        app.preflight.checks = (replace(denied_check(), status='ok', detail='CPU=auto, memory=auto; inherited',
                                       resource_diagnostic=replace(denied_check().resource_diagnostic, code='inherited')),)
        app.preflight.refresh_preflight_state()
        assert app.query_one('#environment-error', EnvironmentIndicator).content == '× 0'
        assert app.query_one('#environment-not_requested', EnvironmentIndicator).content == '- 1'
        assert not app.query_one('#start-run', Button).disabled
        assert 'auto' in app.preflight.status.items[0].check.detail or any(
            'auto' in item.check.detail for item in app.preflight.status.items)
        app.preflight.check_running = True
        app.preflight.refresh_preflight_state()
        await pilot.pause()
        assert app.query_one('#start-run', Button).disabled
        assert app.query_one('#environment-pending').display
        assert app.query_one('#environment-success', EnvironmentIndicator).content == '✓ 1'


async def test_plan_quality_warning_refreshes_without_environment_probe(tmp_path, monkeypatch):
    from unittest.mock import Mock
    app = AcprofTui(RunConfig.smoke('fixture/model'), settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        check = Mock()
        monkeypatch.setattr(app.preflight, 'execute_quick_check', check)
        assert app.query_one('#environment-warning', EnvironmentIndicator).content == '! 1'
        assert not app.query_one('#start-run', Button).disabled
        app.query_one('#repeat', Input).value = '3'
        await wait_for_condition(pilot, lambda: app.query_one('#environment-warning', EnvironmentIndicator).content == '! 0',
                                 timeout=3, description='resolved sample-count warning')
        assert not app.query_one('#start-run', Button).disabled
        check.assert_not_called()


async def test_required_collector_opens_permissions_tab(tmp_path, monkeypatch):
    from textual.widgets import TabbedContent

    from acprof.tui.run.diagnostics import PreflightCheck
    from acprof.tui.settings.environment import EnvironmentSettingsScreen

    monkeypatch.setattr('acprof.tui.app.PROJECT_DIR', tmp_path)
    app = AcprofTui(RunConfig.smoke('fixture/model'), settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        app.preflight.checks = (PreflightCheck('perf', 'fail', 'permission denied', check_id='cpu_instructions'),)
        app.preflight.refresh_preflight_state()
        app.query_one('#environment-error').focus()
        await pilot.press('enter')
        await wait_for_condition(pilot, lambda: bool(app.query('#preflight-signal-cpu_instructions')),
                                 timeout=3, description='required perf issue')
        app.query_one('#preflight-signal-cpu_instructions').focus()
        await pilot.press('enter')
        await wait_for_condition(pilot, lambda: isinstance(app.screen, EnvironmentSettingsScreen),
                                 timeout=3, description='permission recovery screen')
        assert app.screen.query_one('#environment-tabs', TabbedContent).active == 'permissions-tab'
        assert app.screen.query_one('#permission-perf').region.width > 0
