"""Global Docker warnings lead to copy-only recovery with live cached updates."""
from dataclasses import replace
from unittest.mock import Mock

import pytest
from textual.widgets import Button, Collapsible

from acprof.experiments.configuration import RunConfig
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.run.docker_diagnostics import DockerDiagnostic, DockerHost
from tests.support.tui_fixtures import AcprofTui, expand_check_source
from tests.support.tui_waits import wait_for_condition


def docker_check(code='cli_missing'):
    diagnostic = DockerDiagnostic(code, DockerHost('Linux', 'ubuntu', 'Ubuntu 24.04', 'noble', 'native_linux', True),
                                  context='default', endpoint='unix:///var/run/docker.sock',
                                  detail='permission denied' if code == 'socket_permission' else '')
    return PreflightCheck('Docker', 'ok' if code == 'ready' else 'fail', diagnostic.summary,
                          check_id='docker', docker_diagnostic=diagnostic)


async def open_docker(app, pilot, *, keyboard=False):
    app.preflight.environment_checks = (docker_check(),)
    app.preflight.refresh_preflight_state()
    app.query_one('#environment-not_requested').focus()
    await pilot.press('enter')
    await wait_for_condition(pilot, lambda: bool(app.query('#preflight-section-docker')),
                             description='unrequested Docker source visible', timeout=3)
    await expand_check_source(app, pilot, 'docker')
    signal = app.query_one('#preflight-signal-docker')
    signal.scroll_visible(immediate=True, force=True)
    await pilot.pause()
    if keyboard:
        signal.focus()
        await pilot.press('enter')
    else:
        assert await pilot.click(signal)
    await wait_for_condition(pilot, lambda: app.screen.id == 'docker-repair-screen',
                             description='Docker repair guide opened by warning', timeout=3)
    await pilot.pause()


@pytest.mark.parametrize(('language', 'size'), [('zh', (80, 24)), ('en', (120, 30)), ('zh', (150, 45))])
async def test_guide_copy_recheck_failure_then_ready(tmp_path, monkeypatch, language, size):
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='auto', mems='auto'),
                    settings_path=tmp_path / 'settings.json')
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size) as pilot:
        checks = Mock()
        monkeypatch.setattr(app.preflight, 'execute_quick_check', checks)
        await open_docker(app, pilot, keyboard=language == 'en')
        screen = app.screen
        checks.assert_not_called()
        assert not app.query_one('#start-run', Button).disabled
        assert screen.query_one('#docker-technical', Collapsible).collapsed
        copy = screen.query_one('#docker-copy-repository', Button)
        copy.scroll_visible(immediate=True, force=True)
        await pilot.pause()
        clipboard = Mock()
        monkeypatch.setattr(app, 'copy_to_clipboard', clipboard)
        assert await pilot.click(copy)
        await wait_for_condition(pilot, lambda: clipboard.called, description='repository commands copied', timeout=2)
        assert 'https://download.docker.com/linux/ubuntu' in clipboard.call_args.args[0]
        checks.assert_not_called()
        retry = screen.query_one('#docker-recheck', Button)
        assert retry.region.width > 0
        assert retry.region.bottom <= size[1]
        assert await pilot.click(retry)
        await wait_for_condition(pilot, lambda: checks.called, description='explicit recheck starts', timeout=2)
        assert retry.disabled
        assert 'checking' in str(screen.query_one('#docker-status').content).lower() or '检查' in str(screen.query_one('#docker-status').content)
        config, token = checks.call_args.args
        app.preflight.show_quick_check(app.preflight.checks, '', token, True, (docker_check('socket_permission'),), '')
        await wait_for_condition(pilot, lambda: bool(screen.query('#docker-copy-group')),
                                 description='permission repair replaces installation', timeout=3)
        assert 'root' in str(screen.query_one('#docker-note-group').content)
        assert not retry.disabled
        assert await pilot.click(retry)
        await wait_for_condition(pilot, lambda: checks.call_count == 2, description='second recheck starts', timeout=2)
        _, token = checks.call_args.args
        app.preflight.show_quick_check(app.preflight.checks, '', token, True, (docker_check('ready'),), '')
        await pilot.pause()
        assert any(item.check.check_id == 'docker' and item.group == 'success' for item in app.preflight.status.items)
        assert not screen.query('#docker-copy-group')
        assert not app.query_one('#start-run', Button).disabled
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert retry.region.bottom <= 24
        await pilot.press('escape')
        await wait_for_condition(pilot, lambda: len(app.screen_stack) == 1, description='guide closes', timeout=2)


async def test_measurement_rejects_docker_recheck_even_from_an_open_guide(tmp_path, monkeypatch):
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='process'), settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=(80, 24)) as pilot:
        await open_docker(app, pilot)
        checks = Mock()
        monkeypatch.setattr(app.preflight, 'execute_quick_check', checks)
        app.execution.session.snapshot = replace(app.execution.session.snapshot, measurement_active=True)
        app.screen.update_cached(app.preflight.status)
        assert app.screen.query_one('#docker-recheck', Button).disabled
        app.preflight.quick_check()
        checks.assert_not_called()
        app.execution.session.snapshot = replace(app.execution.session.snapshot, measurement_active=False)
        app.screen.update_cached(app.preflight.status)
        assert not app.screen.query_one('#docker-recheck', Button).disabled
