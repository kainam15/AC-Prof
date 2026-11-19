"""The real header, issue navigation and local hints share one status model."""
from dataclasses import replace

import pytest
from textual.widgets import Button, ContentSwitcher, Input, Static

from acprof.experiments.configuration import RunConfig
from acprof.tui.ui.environment import EnvironmentIndicator
from acprof.tui.ui.structured import ActionButtons
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition
from tests.tui.workflows.test_tui_process_permissions import denied_check


async def locate_selected(app, pilot):
    button = list(app.query_one('#preflight-callout').query_one(ActionButtons).query(Button))[1]
    button.scroll_visible(immediate=True, force=True)
    button.focus()
    await pilot.press('enter')


@pytest.mark.parametrize(('size', 'language'), [((80, 24), 'zh'), ((120, 30), 'en'), ((150, 45), 'zh')])
async def test_global_blockers_navigate_with_mouse_and_keyboard(tmp_path, size, language):
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='1', mems='1'),
                    settings_path=tmp_path / 'settings.json')
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size) as pilot:
        app.preflight.checks = (denied_check(),)
        app.query_one('#model', Input).value = ''
        await wait_for_condition(pilot, lambda: app.preflight.status.counts['error'] == 1,
                                 timeout=3, description='quota blocker with pending model')
        assert not app.query_one('#model-error', Static).display
        app.experiments.request_run()  # An explicit start request must still validate strictly.
        await wait_for_condition(pilot, lambda: app.preflight.status.counts['error'] == 2,
                                 timeout=3, description='configuration and quota blockers')
        assert app.query_one('#environment-error', EnvironmentIndicator).content == '× 2'
        assert app.query_one('#start-run', Button).disabled
        assert app.query_one('#model-error', Static).display
        assert app.query_one('#cpus-error', Static).display
        await pilot.pause()  # Complete the strict-error field focus before opening diagnostics.
        app.query_one('#environment-error').focus()
        await pilot.press('enter')
        await wait_for_condition(pilot, lambda: bool(app.query('#preflight-signal-config_model')),
                                 diagnostics=lambda: f"state={app.preflight.status.state}; rows={[(x.check.check_id, x.group) for x in app.preflight.status.items]}; panel={app.preflight.panel.display}",
                                 timeout=3, description='global error rows')
        row = app.query_one('#preflight-signal-config_model')
        row.scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert await pilot.click(row)
        await locate_selected(app, pilot)
        await wait_for_condition(pilot, lambda: app.focused is app.query_one('#model'),
                                 timeout=3, description='jump to invalid model')
        assert not app.preflight.panel.display
        app.query_one('#environment-error').focus()
        await pilot.press('enter')
        await pilot.pause()
        app.query_one('#preflight-signal-process_resources').focus()
        await pilot.press('enter')
        await locate_selected(app, pilot)
        await wait_for_condition(pilot, lambda: app.screen.id == 'process-resource-screen',
                                 timeout=3, description='dedicated permission recovery')
        assert not app.query('#process-resource-help')
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert app.focused is not None
        assert app.focused.region.bottom <= app.screen.size.height


async def test_header_separates_pending_requirements_and_optional_warnings(tmp_path):
    from acprof.tui.run.diagnostics import PreflightCheck
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='auto', mems='auto'),
                    settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        app.preflight.environment_checks = (PreflightCheck('NCU', 'fail', 'not installed', check_id='ncu'),)
        app.preflight.checks = (replace(denied_check(), status='ok', detail='CPU=auto, memory=auto; inherited',
                                       resource_diagnostic=replace(denied_check().resource_diagnostic, code='inherited')),)
        app.preflight.refresh_preflight_state()
        assert app.query_one('#environment-error', EnvironmentIndicator).content == '× 0'
        assert app.query_one('#environment-warning', EnvironmentIndicator).content == '! 1'
        assert any(item.check.check_id == 'ncu' and item.group == 'not_requested' for item in app.preflight.status.items)
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
        await wait_for_condition(pilot, lambda: not app.configuration.validation_pending
                                 and app.query_one('#environment-warning', EnvironmentIndicator).content == '! 0',
                                 timeout=3, description='resolved sample-count warning')
        assert not app.query_one('#start-run', Button).disabled
        check.assert_not_called()


@pytest.mark.parametrize(('language', 'size', 'label'), [
    ('zh', (80, 24), 'Nsight Compute · 未安装'), ('en', (120, 30), 'Nsight Compute · Not installed'),
])
async def test_optional_profiler_inspect_enable_and_recheck(tmp_path, monkeypatch, language, size, label):
    from unittest.mock import Mock

    from acprof.tui.app import AcprofTui as ProductionTui
    from acprof.tui.i18n import message_id
    from acprof.tui.run.diagnostics import PreflightCheck
    from tests.support.tui_fixtures import expand_check_source

    missing = PreflightCheck('ncu', 'fail', message_id('common.state.not_installed'),
                             'unavailable', check_id='ncu', reason_code='not_installed')
    repaired = PreflightCheck('ncu', 'ok', 'fixture version', 'available', check_id='ncu')
    scan = Mock(side_effect=[(missing,), (repaired,)])
    monkeypatch.setattr('acprof.tui.run.preflight_controller.scan_environment', scan)
    monkeypatch.setattr('acprof.tui.run.preflight_controller.quick_preflight', Mock(return_value=[
        PreflightCheck('Runtime', 'ok', 'Docker', runtime_kind='docker', check_id='runtime'),
    ]))
    config = replace(RunConfig.smoke('fixture/model'), runtime='docker', gpus='on', repeat=3)
    app = ProductionTui(config, settings_path=tmp_path / 'settings.json')
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size) as pilot:
        await wait_for_condition(pilot, lambda: app.preflight.check_completed, description='startup scan', timeout=5)
        assert app.query_one('#environment-warning').content == '! 0'
        assert await pilot.click('#environment-not_requested')
        await expand_check_source(app, pilot, 'ncu')
        signal = app.query_one('#preflight-signal-ncu')
        signal.scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert await pilot.click(signal)
        assert signal.render().plain == '! ' + label
        callout = app.query_one('#preflight-callout')
        assert app.tr(callout.content.problem) == '! ' + label
        assert 'gpu_executed_mflop_per_request_ncu' in app.tr(callout.content.impact)
        assert 'ncu --version' in app.tr(callout.content.next_step)
        await locate_selected(app, pilot)
        await wait_for_condition(pilot, lambda: app.focused is app.query_one('#ncu-profiler'),
                                 description='Nsight Compute checkbox focused', timeout=3)
        await pilot.press('space')
        await wait_for_condition(pilot, lambda: app.preflight.config_matches()
                                 and app.query_one('#environment-warning').content == '! 1',
                                 description='selected profiler counted from cached inventory', timeout=5)
        indicator = app.query_one('#environment-warning')
        await pilot.pause()
        region = indicator.content_region
        strip = app.screen._compositor.render_strips()[region.y].crop(region.x, region.right)
        assert strip.text.strip() == '! 1', 'header must show only status count'
        assert label not in strip.text
        await pilot.resize_terminal(52, 24)
        await pilot.pause()
        assert app.query_one('#quit-app').region.right <= 52
        assert label not in str(indicator.tooltip)
        await pilot.resize_terminal(*size)
        await pilot.pause()
        assert scan.call_count == 1
        assert not app.query_one('#start-run', Button).disabled
        assert await pilot.click('#environment-warning')
        await pilot.pause()
        retry = app.preflight.panel.retry_button
        retry.scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert await pilot.click(retry)
        await wait_for_condition(pilot, lambda: app.preflight.environment_checks == (repaired,)
                                 and app.query_one('#environment-warning').content == '! 0',
                                 description='repaired profiler removed from warnings', timeout=5)
        assert scan.call_count == 2
        assert not app.query_one('#start-run', Button).disabled
        assert indicator.render().plain == '! 0'


async def test_optional_collector_opens_permissions_tab(tmp_path, monkeypatch):
    from textual.widgets import TabbedContent

    from acprof.tui.run.diagnostics import PreflightCheck
    from acprof.tui.settings.environment import EnvironmentSettingsScreen

    monkeypatch.setattr('acprof.tui.app.PROJECT_DIR', tmp_path)
    app = AcprofTui(RunConfig.smoke('fixture/model'), settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        app.preflight.checks = (PreflightCheck('perf', 'fail', 'permission denied', check_id='cpu_instructions'),)
        app.preflight.refresh_preflight_state()
        assert not app.query_one('#start-run', Button).disabled
        app.query_one('#environment-warning').focus()
        await pilot.press('enter')
        await wait_for_condition(pilot, lambda: bool(app.query('#preflight-signal-cpu_instructions')),
                                 timeout=3, description='optional perf permission warning')
        app.query_one('#preflight-signal-cpu_instructions').focus()
        await pilot.press('enter')
        await locate_selected(app, pilot)
        await wait_for_condition(pilot, lambda: isinstance(app.screen, EnvironmentSettingsScreen),
                                 timeout=3, description='permission recovery screen')
        assert app.screen.query_one('#environment-tabs', TabbedContent).active == 'permissions-tab'
        assert app.screen.query_one('#permission-perf').region.width > 0


async def test_process_profiler_blocker_navigates_to_basic_runtime_control(tmp_path):
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='process', profiling_mode='full', compute_profile_tool='torch'),
                    settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        assert app.query_one('#environment-error', EnvironmentIndicator).content == '× 1'
        assert app.query_one('#start-run', Button).disabled
        assert app.query_one('#runtime-error', Static).display
        app.query_one('#environment-error').focus()
        await pilot.press('enter')
        await wait_for_condition(pilot, lambda: bool(app.query('#preflight-signal-config_runtime')),
                                 timeout=3, description='unsupported Process profiler issue')
        app.query_one('#preflight-signal-config_runtime').focus()
        await pilot.press('enter')
        await locate_selected(app, pilot)
        await wait_for_condition(pilot, lambda: app.focused is app.query_one('#runtime'),
                                 timeout=3, description='basic runtime control')
        assert app.query_one('#experiment-pages', ContentSwitcher).current == 'run-form'
        assert app.query_one('#runtime').is_on_screen



async def test_collector_issue_updates_cached_status_after_preflight(tmp_path):
    from acprof.events.progress import encode_event, parse_event
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        await wait_for_condition(pilot, lambda: app.preflight.check_completed, timeout=3, description="fixture preflight")
        event = parse_event(encode_event("collector_issue", collector="cpu_instructions", detail="perf attach denied",
                                         fields=["cpu_instructions_total"]))
        app.preflight.collection_issue(event)
        issue = next(item for item in app.preflight.status.items if item.check.check_id == "cpu_instructions")
        assert issue.group == "warning"
        assert issue.target == "permissions:perf"
        assert "perf attach denied" in issue.check.detail
        assert "cpu_instructions_total" in issue.check.detail
        assert not app.query_one("#start-run", Button).disabled
