"""Runtime controls are usable in a small terminal without host/Docker probes."""
from dataclasses import replace
from pathlib import Path

import pytest
from textual.widgets import Button, ContentSwitcher, Select, TabbedContent
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig, build_run_command
from acprof.tui.diagnostics import quick_preflight


@pytest.mark.parametrize('size', [(80, 24), (120, 40)])
async def test_process_inherited_button_builds_runnable_basic_command(tmp_path, size):
    config = replace(RunConfig.smoke('fixture/model'), runtime='process')
    app = AcprofTui(config, settings_path=tmp_path / 'settings.json')
    async with app.run_test(size=size) as pilot:
        app.query_one('#main-tabs', TabbedContent).active = 'run-tab'
        app.query_one('#experiment-pages', ContentSwitcher).current = 'advanced-form'
        await pilot.pause()
        button = app.query_one('#process-inherited-resources', Button)
        button.scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert button.display
        assert button.region.height > 0
        assert await pilot.click(button)
        await pilot.pause()
        actual = app.configuration.collect_config()
        assert actual.runtime == 'process'
        assert actual.profiling_mode == 'basic'
        assert actual.cpus == actual.mems == 'auto'
        assert not actual.prune_startup_oom
        assert app.query_one('#runtime', Select).value == 'process'
        assert app.query_one('#probe-largest', Button).disabled
        command = build_run_command(actual, project_dir=Path.cwd())
        assert command[command.index('--runtime') + 1] == 'process'
        assert command[command.index('--cpus') + 1] == 'auto'


def test_process_preflight_keeps_optional_permission_failures_nonfatal(tmp_path, monkeypatch):
    from acprof.host.runtimes import capabilities, resources
    monkeypatch.setattr(resources, 'preflight_resources', lambda *args: None)
    monkeypatch.setattr('acprof.host.runtimes.process_control.require_process_supervision', lambda: 'available')
    monkeypatch.setattr(capabilities, 'probe_collection_plan', lambda *a, **kw: {'probes': {
        'process_cpu': {'status': 'permission_denied', 'detail': 'fixture procfs denied'},
        'process_memory': {'status': 'available', 'detail': 'fixture RSS'},
        'cpu_energy': {'status': 'unsupported', 'detail': 'no RAPL'},
        'gpu_memory': {'status': 'not_requested', 'detail': 'CPU-only experiment'},
    }})
    def which(name):
        assert name != 'docker'
        return '/fixture/uv' if name == 'uv' else None
    monkeypatch.setattr('acprof.tui.diagnostics.shutil.which', which)
    config = replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='auto', mems='auto')
    checks = quick_preflight(config, project_dir=tmp_path,
                             command_runner=lambda *a, **kw: pytest.fail('Process preflight must not invoke Docker'))
    assert not any(check.status == 'fail' for check in checks)
    assert next(check for check in checks if check.label == 'process_cpu').capability_status == 'permission_denied'
    assert next(check for check in checks if check.label == 'gpu_memory').status == 'not_requested'


def test_process_doctor_never_claims_native_benchmark():
    from acprof.host.doctor import DoctorCheck, report_dict
    report = report_dict([DoctorCheck('runtime', 'available', 'Process', runtime_kind='process')],
                         profiling_mode='basic', gpus='off')
    assert report['ready']
    assert not report['native_benchmark']


def test_unavailable_process_ownership_blocks_preflight_before_resource_checks(tmp_path, monkeypatch):
    from acprof.host.runtimes import capabilities, process_control, resources
    monkeypatch.setattr(capabilities, 'probe_collection_plan', lambda *a, **kw: {'probes': {}})
    def denied():
        raise RuntimeError('pidfd denied by seccomp')
    monkeypatch.setattr(process_control, 'require_process_supervision', denied)
    monkeypatch.setattr(resources, 'preflight_resources', lambda *a: pytest.fail('ownership must be checked first'))
    config = replace(RunConfig.smoke('fixture/model'), runtime='process', cpus='auto', mems='auto')
    checks = quick_preflight(config, project_dir=tmp_path)
    assert any(check.status == 'fail' and 'pidfd denied' in str(check.detail) for check in checks)
