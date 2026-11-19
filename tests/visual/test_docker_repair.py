"""Reviewed Docker recovery scenes reuse the actual global-status dialog."""
from dataclasses import replace

import pytest
from textual.widgets import Collapsible

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from acprof.tui.ui.structured import Callout
from tests.support.tui_waits import wait_for_condition
from tests.tui.workflows.test_tui_docker_repair import docker_check

pytestmark = pytest.mark.visual


@pytest.mark.parametrize(('language', 'size', 'scene'), [
    ('zh', (80, 24), 'install'), ('en', (80, 24), 'install'),
    ('en', (120, 30), 'permission'), ('zh', (80, 24), 'permission'),
    ('zh', (120, 30), 'wsl'), ('en', (120, 30), 'desktop'),
    ('zh', (150, 45), 'checking'), ('en', (80, 24), 'ready'),
])
def test_docker_repair(snap_compare, svg_operations, tmp_path, language, size, scene):
    config = replace(RunConfig.smoke('fixture/model'), runtime='docker' if scene == 'desktop' else 'process',
                     cpus='auto', mems='auto')
    app = AcprofTui(config, settings_path=tmp_path / 'settings.json')
    app.configuration.preferences = replace(app.configuration.preferences, language=language,
                                            theme='acprof-light' if language == 'en' else 'acprof-graphite')

    async def prepare(pilot):
        await svg_operations.wait_for_preflight()
        await pilot.pause()
        app.set_input_cursor_blink_enabled(False)
        code = {'permission': 'socket_permission', 'wsl': 'daemon_unreachable',
                'desktop': 'desktop_unsupported', 'ready': 'ready'}.get(scene, 'cli_missing')
        check = docker_check(code)
        diagnostic = check.docker_diagnostic
        if scene == 'wsl':
            diagnostic = replace(diagnostic, host=replace(diagnostic.host, environment='wsl2', systemd=False))
        elif scene == 'desktop':
            diagnostic = replace(diagnostic, context='desktop-linux', endpoint='unix:///home/user/.docker/desktop/docker.sock', product='Docker Desktop')
        app.preflight.environment_checks = (replace(check, docker_diagnostic=diagnostic),)
        if scene == 'desktop':
            app.preflight.checks = app.preflight.environment_checks
        app.preflight.refresh_preflight_state()
        row = next(item.check for item in app.preflight.status.items if item.check.check_id == 'docker')
        if scene == 'desktop':
            assert row.status == 'fail'
            assert not app.preflight.status.can_start
        app.preflight.panel.open_issue(row)
        await wait_for_condition(pilot, lambda: app.screen.id == 'docker-repair-screen' and app.screen.is_mounted,
                                 description='Docker guide mounted', timeout=3)
        if scene == 'checking':
            app.preflight.check_running = True
            app.preflight.refresh_preflight_state()
        await pilot.pause()
        assert app.screen.query_one('#docker-summary', Callout).content.problem == diagnostic.summary
        assert app.screen.query_one('#docker-technical', Collapsible).collapsed
        if scene == 'permission':
            app.screen.query_one('#docker-note-group').scroll_visible(immediate=True, force=True, top=True)
            await pilot.pause()
        app.clear_notifications()
    assert snap_compare(app, terminal_size=size, run_before=prepare)
