"""Source names, partial NVML evidence and grouped capture dependencies."""
from dataclasses import replace
from pathlib import Path

import pytest

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from acprof.tui.i18n import message_id
from acprof.tui.run.diagnostics import PreflightCheck
from tests.support.tui_fixtures import expand_check_source

pytestmark = pytest.mark.visual


@pytest.mark.parametrize(('language', 'size'), [('zh', (80, 24)), ('en', (120, 30)), ('zh', (52, 24))])
def test_source_groups(snap_compare, svg_operations, tmp_path, monkeypatch, language, size):
    monkeypatch.setattr('acprof.tui.app.PROJECT_DIR', Path('/workspace'))
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='docker', gpus='on', repeat=3),
                    settings_path=tmp_path / 'settings.json')
    app.configuration.preferences = replace(app.configuration.preferences, language=language)

    async def prepare(pilot):
        await svg_operations.wait_for_preflight()
        app.set_input_cursor_blink_enabled(False)
        app.preflight.environment_checks = (
            PreflightCheck('GPU', 'ok', 'fixture device', check_id='gpu_device'),
            PreflightCheck('Memory', 'ok', 'fixture memory', check_id='gpu_memory'),
            PreflightCheck('Utilization', 'ok', 'fixture utilization', check_id='gpu_utilization'),
            PreflightCheck('Power', 'fail', 'fixture unsupported power counter', 'unsupported', check_id='gpu_power'),
            PreflightCheck('tcpdump', 'fail', 'fixture missing tool', check_id='tcpdump', reason_code='not_installed'),
            PreflightCheck('Permissions', 'fail', 'fixture missing tool', 'unavailable', check_id='capture_permissions',
                           source='tcpdump', reason_code='dependency_unavailable'),
        )
        app.preflight.refresh_preflight_state()
        app.preflight.panel.show(pin=True, group='warning')
        await pilot.pause()
        assert app.preflight.status.counts['warning'] == 2
        assert not app.preflight.panel.query_one('#preflight-section-nvml').collapsed
        app.clear_notifications()
    assert snap_compare(app, terminal_size=size, run_before=prepare)


@pytest.mark.parametrize(('language', 'size', 'identifier'), [
    ('zh', (80, 24), 'ncu'), ('en', (120, 30), 'cpu_energy'),
])
def test_capability_reason(snap_compare, svg_operations, tmp_path, monkeypatch, language, size, identifier):
    monkeypatch.setattr('acprof.tui.app.PROJECT_DIR', Path('/workspace'))
    config = replace(RunConfig.smoke('fixture/model'), runtime='process', repeat=3)
    if identifier == 'cpu_energy':
        config = replace(config, runtime='docker')
    app = AcprofTui(config, settings_path=tmp_path / 'settings.json')
    app.configuration.preferences = replace(app.configuration.preferences, language=language)

    async def prepare(pilot):
        await svg_operations.wait_for_preflight()
        app.set_input_cursor_blink_enabled(False)
        app.preflight.environment_checks = (PreflightCheck('ncu', 'fail', message_id('common.state.not_installed'),
                                                           check_id='ncu', reason_code='not_installed'),)
        if identifier == 'cpu_energy':
            app.preflight.checks = (PreflightCheck('RAPL', 'fail', 'permission_denied: cannot read energy_uj',
                                                   'permission_denied', check_id='cpu_energy'),)
        app.preflight.refresh_preflight_state()
        app.preflight.panel.show(pin=True, group='warning' if identifier == 'cpu_energy' else 'not_requested')
        await pilot.pause()
        await expand_check_source(app, pilot, identifier)
        app.query_one('#preflight-signal-' + identifier).focus()
        app.clear_notifications()
    assert snap_compare(app, terminal_size=size, run_before=prepare)
