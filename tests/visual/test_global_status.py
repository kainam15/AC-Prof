"""Current configuration blockers, stale checks and inherited resource semantics."""
from dataclasses import replace

import pytest
from textual.widgets import Input

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from acprof.tui.run.diagnostics import PreflightCheck
from tests.tui.workflows.test_tui_process_permissions import denied_check

pytestmark = pytest.mark.visual


@pytest.mark.parametrize(('language', 'size', 'scene'), [
    ('zh', (80, 24), 'blocked'), ('en', (120, 30), 'blocked'),
    ('zh', (150, 45), 'checking'), ('zh', (80, 24), 'inherited'),
])
def test_global_status(snap_compare, svg_operations, tmp_path, language, size, scene):
    config = replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='1', mems='1')
    if scene == 'inherited':
        config = replace(config, cpus='auto', mems='auto')
    app = AcprofTui(config, settings_path=tmp_path / 'settings.json')
    app.configuration.preferences = replace(app.configuration.preferences, language=language,
                                            theme='acprof-light' if language == 'en' else 'acprof-graphite')

    async def prepare(pilot):
        await pilot.pause()
        app.set_input_cursor_blink_enabled(False)
        app.preflight.environment_checks = (
            PreflightCheck('uv', 'ok', '/fixture/bin/uv', check_id='uv'),
            PreflightCheck('Docker', 'fail', 'not installed', check_id='docker'),
            PreflightCheck('NCU', 'fail', 'not installed', check_id='ncu'),
        )
        app.preflight.checks = (denied_check(),)
        if scene == 'blocked':
            app.query_one('#model', Input).value = ''
            app.configuration.refresh_command_preview(notify=False)
        elif scene == 'checking':
            app.preflight.check_running = True
        else:
            app.preflight.checks = (replace(denied_check(), status='ok',
                                           detail='CPU=auto, memory=auto; inherited, sharing=unknown',
                                           resource_diagnostic=replace(denied_check().resource_diagnostic, code='inherited')),)
        app.preflight.refresh_preflight_state()
        app.preflight.panel.show(pin=True, group='error' if scene == 'blocked' else 'pending' if scene == 'checking' else 'success')
        await pilot.pause()
        if scene == 'inherited':
            app.query_one('#preflight-signal-process_resources').focus()
        app.clear_notifications()
    assert snap_compare(app, terminal_size=size, run_before=prepare)
