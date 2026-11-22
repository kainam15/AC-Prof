"""Docker repair classification uses bounded read-only commands and real routing."""
import os
import subprocess
from dataclasses import replace
from unittest.mock import Mock

import pytest

from acprof.experiments.configuration import RunConfig
from acprof.platform import Environment
from acprof.tui.run.diagnostics import PreflightCheck, quick_preflight
from tests.tui.workflows.test_global_status import controller


@pytest.fixture
def docker_probe(monkeypatch):
    monkeypatch.setattr('acprof.host.runtimes.capabilities.probe_collection_plan', lambda *a, **kw: {'probes': {}})
    monkeypatch.setattr('acprof.tui.run.diagnostics.detect_environment', lambda: Environment('native_linux', system='Linux'))
    monkeypatch.setattr('shutil.which', lambda name: '/usr/bin/' + name)
    monkeypatch.delenv('DOCKER_CONTEXT', raising=False)
    monkeypatch.delenv('DOCKER_HOST', raising=False)
    calls = []

    def run(*, context='default', endpoint='unix:///var/run/docker.sock', error='', inspect_error='', runtime='docker'):
        def command(args, **kwargs):
            calls.append(tuple(args))
            assert 0 < kwargs['timeout'] <= 15
            if args[1:3] == ('context', 'show'):
                return subprocess.CompletedProcess(args, 0, context, '')
            if args[1:3] == ('context', 'inspect'):
                return subprocess.CompletedProcess(args, bool(inspect_error), endpoint if not inspect_error else '', inspect_error)
            assert args[1] == 'info', 'repairs must never execute from a probe'
            return subprocess.CompletedProcess(args, bool(error), 'fixture|Linux' if not error else '', error)
        checks = quick_preflight(replace(RunConfig.smoke('fixture/model'), runtime=runtime), command_runner=command)
        return next(row for row in checks if row.check_id == 'docker')
    return run, calls


@pytest.mark.parametrize(('error', 'code'), [
    ('Cannot connect to the Docker daemon at unix:///var/run/docker.sock', 'daemon_unreachable'),
    ('permission denied while trying to connect to the Docker daemon socket', 'socket_permission'),
    ('', 'ready'),
])
def test_docker_failure_has_specific_repair_reason(docker_probe, error, code):
    run, _ = docker_probe
    row = run(error=error)
    diagnostic = getattr(row, 'docker_diagnostic', None)
    assert diagnostic is not None, 'Docker checks need structured installation and repair evidence'
    assert diagnostic.code == code
    assert row.status == ('ok' if code == 'ready' else 'fail')
    assert error in diagnostic.detail


def test_context_failure_does_not_fall_through_to_an_unrelated_daemon(docker_probe):
    run, calls = docker_probe
    row = run(inspect_error='context "missing" does not exist')
    assert getattr(row, 'docker_diagnostic', None) is not None
    assert row.docker_diagnostic.code == 'context_unavailable'
    assert not any(args[1] == 'info' for args in calls)


@pytest.mark.parametrize(('mode', 'group'), [('best_effort', 'not_requested'), ('basic', 'warning')])
def test_process_keeps_missing_docker_available_for_inspection(mode, group):
    value = controller(current=replace(RunConfig.smoke('fixture/model'), runtime='process', profiling_mode=mode),
                       inventory=[PreflightCheck('Docker', 'fail', 'CLI missing', check_id='docker', reason_code='not_installed')])
    item = next(item for item in value.status.items if item.check.check_id == 'docker')
    assert item.group == group
    assert item.check.reason_code == 'not_installed'
    assert value.status.can_start


def test_auto_fallback_replaces_old_inventory_with_fresh_docker_diagnosis():
    from tests.tui.workflows.test_tui_docker_repair import docker_check
    value = controller(current=RunConfig.smoke('fixture/model'), inventory=[docker_check('ready')], checks=[
        docker_check('socket_permission'),
        PreflightCheck('Runtime', 'ok', 'Process', runtime_kind='process', check_id='runtime'),
    ])
    item = next(item for item in value.status.items if item.check.check_id == 'docker')
    assert item.check.docker_diagnostic.code == 'socket_permission'
    assert item.group == 'warning'
    assert not item.required
    assert value.status.can_start


def test_auto_fallback_preserves_the_actionable_docker_reason(docker_probe, monkeypatch):
    monkeypatch.setattr('acprof.tui.run.diagnostics._process_checks', lambda _, *, project_dir: [
        PreflightCheck('Runtime', 'ok', 'Process', runtime_kind='process', check_id='runtime')])
    run, _ = docker_probe
    row = run(error='Cannot connect to the Docker daemon', runtime='auto')
    assert row.status == 'warn'
    assert row.docker_diagnostic.code == 'daemon_unreachable'


def test_explicit_context_overrides_host_without_mutating_environment(docker_probe, monkeypatch):
    monkeypatch.setenv('DOCKER_CONTEXT', 'production')
    monkeypatch.setenv('DOCKER_HOST', 'unix:///var/run/docker.sock')
    before = dict(os.environ)
    run, calls = docker_probe
    row = run(context='production', endpoint='ssh://remote.example')
    assert row.docker_diagnostic.code == 'context_unsupported'
    assert row.docker_diagnostic.endpoint == 'ssh://remote.example'
    assert not any(args[1] == 'info' for args in calls)
    assert dict(os.environ) == before


def test_missing_cli_does_not_execute_commands(monkeypatch):
    monkeypatch.setattr('acprof.host.runtimes.capabilities.probe_collection_plan', lambda *a, **kw: {'probes': {}})
    monkeypatch.setattr('shutil.which', lambda _: None)
    monkeypatch.setattr('acprof.tui.run.docker_diagnostics._cli_outside_path', lambda: '')
    command = Mock(side_effect=AssertionError('CLI is unavailable'))
    row = next(row for row in quick_preflight(
        replace(RunConfig.smoke('fixture/model'), runtime='docker'), command_runner=command,
    ) if row.check_id == 'docker')
    assert row.status == 'fail'
    assert getattr(row, 'docker_diagnostic', None) is not None
    assert row.docker_diagnostic.code == 'cli_missing'
    command.assert_not_called()


def test_installation_is_detected_on_next_check_without_restarting(docker_probe, monkeypatch):
    run, _ = docker_probe
    monkeypatch.setattr('acprof.tui.run.docker_diagnostics._cli_outside_path', lambda: '')
    monkeypatch.setattr('shutil.which', lambda _: None)
    assert run().docker_diagnostic.code == 'cli_missing'
    monkeypatch.setattr('shutil.which', lambda name: '/usr/bin/' + name)
    assert run().docker_diagnostic.code == 'ready'


def test_daemon_api_error_does_not_suggest_stopped_service(docker_probe):
    run, _ = docker_probe
    row = run(error='client version 1.53 is too new. Maximum supported API version is 1.41')
    assert row.docker_diagnostic.code == 'check_failed'
    assert 'client version' in row.docker_diagnostic.detail


@pytest.mark.parametrize(('context', 'endpoint', 'code'), [
    ('desktop-linux', 'unix:///home/user/.docker/desktop/docker.sock', 'desktop_unsupported'),
    ('rootless', 'unix:///run/user/1000/docker.sock', 'context_unsupported'),
    ('remote', 'tcp://example.test:2376', 'context_unsupported'),
])
def test_nonlocal_context_does_not_issue_info(docker_probe, context, endpoint, code):
    run, calls = docker_probe
    row = run(context=context, endpoint=endpoint)
    assert row.docker_diagnostic.code == code
    assert row.status == 'fail'
    assert not any(args[1] == 'info' for args in calls)


def test_host_override_ignores_saved_desktop_context(docker_probe, monkeypatch):
    monkeypatch.setenv('DOCKER_HOST', 'unix:///var/run/docker.sock')
    run, calls = docker_probe
    row = run(context='desktop-linux', endpoint='unix:///home/user/.docker/desktop/docker.sock')
    assert row.docker_diagnostic.code == 'ready'
    assert row.docker_diagnostic.selection_source == 'DOCKER_HOST'
    assert len(calls) == 1
    assert calls[0][1] == 'info'


@pytest.mark.parametrize(('stage', 'code'), [('context', 'context_unavailable'), ('info', 'daemon_unreachable')])
def test_timeout_keeps_stage_and_error(docker_probe, stage, code):
    from acprof.tui.run.docker_diagnostics import inspect_docker
    def command(args, **kwargs):
        if args[1] == stage:
            raise subprocess.TimeoutExpired(args, kwargs['timeout'])
        return subprocess.CompletedProcess(args, 0, 'default' if args[2] == 'show' else 'unix:///var/run/docker.sock', '')
    result = inspect_docker(command, Environment('native_linux', system='Linux'))
    assert result.code == code
    assert 'TimeoutExpired' in result.detail


def test_desktop_server_on_native_socket_is_not_engine(docker_probe):
    from acprof.tui.run.docker_diagnostics import inspect_docker
    outputs = ('default', 'unix:///var/run/docker.sock', 'docker-desktop|Linux')
    command = Mock(side_effect=[subprocess.CompletedProcess([], 0, text, '') for text in outputs])
    result = inspect_docker(command, Environment('native_linux', system='Linux'))
    assert result.code == 'desktop_unsupported'
    assert result.product == 'Docker Desktop'


def test_cli_outside_path_can_be_linked_without_overwriting(tmp_path, monkeypatch):
    from acprof.tui.run.docker_diagnostics import inspect_docker
    from acprof.tui.run.docker_guide import repair_steps
    cli = tmp_path / 'installed docker'
    cli.write_text('#!/bin/sh\nexit 0\n')
    cli.chmod(0o755)
    monkeypatch.setattr('shutil.which', lambda _: None)
    monkeypatch.setattr('acprof.tui.run.docker_diagnostics.CLI_LOCATIONS', (str(cli),))
    monkeypatch.setenv('PATH', '/usr/bin:/bin')
    command = Mock(side_effect=AssertionError('do not execute off-PATH binaries'))
    result = inspect_docker(command, Environment('native_linux', system='Linux'))
    assert result.code == 'cli_not_on_path'
    assert result.cli == str(cli)
    assert repair_steps(result)[0].command == f"sudo ln -s -- '{cli}' /usr/bin/docker"
    command.assert_not_called()


def test_context_permission_error_is_not_socket_permission(docker_probe):
    run, _ = docker_probe
    row = run(inspect_error='open /home/user/.docker/contexts/meta: permission denied')
    assert row.docker_diagnostic.code == 'context_unavailable'
    assert '.docker/contexts' in row.docker_diagnostic.detail


def test_explicit_docker_failure_blocks_and_checking_is_not_success():
    config = replace(RunConfig.smoke('fixture/model'), runtime='docker')
    failure = PreflightCheck('Docker', 'fail', 'daemon missing', check_id='docker')
    value = controller(current=config, checks=[failure], inventory=[failure])
    assert value.status.counts['error'] == 1
    assert not value.status.can_start
    value.check_running = True
    row = next(item for item in value.status.items if item.check.check_id == 'docker')
    assert row.check.status == 'checking'
    assert row.group == 'pending'


@pytest.mark.parametrize(('distribution', 'codename'), [('ubuntu', 'noble'), ('debian', 'bookworm')])
def test_install_guide_uses_exact_distribution_and_nonclobber_commands(distribution, codename):
    from acprof.tui.run.docker_diagnostics import DockerDiagnostic, DockerHost
    from acprof.tui.run.docker_guide import repair_steps
    result = DockerDiagnostic('cli_missing', DockerHost('Linux', distribution, distribution, codename, 'native_linux', True))
    steps = repair_steps(result)
    assert f'https://download.docker.com/linux/{distribution}/gpg' in steps[0].command
    assert f'Suites: {codename}' in steps[0].command
    assert 'set -C' in steps[0].command
    assert 'sudo apt-get install docker-ce ' in steps[1].command
    assert 'sudo systemctl start docker' in steps[2].command
    assert 'sudo' not in steps[-1].command
    assert not any('docker run' in step.command or 'context use' in step.command for step in steps)


def test_linux_derivative_does_not_receive_ubuntu_sources():
    from acprof.tui.run.docker_diagnostics import DockerDiagnostic, DockerHost
    from acprof.tui.run.docker_guide import repair_steps
    steps = repair_steps(DockerDiagnostic('cli_missing', DockerHost('Linux', 'linuxmint', 'Linux Mint', 'wilma')))
    assert steps[0].key == 'other_linux'
    assert not any('apt-get' in step.command for step in steps)


@pytest.mark.parametrize('system', ['Windows', 'Darwin'])
def test_unsupported_host_never_probes_docker(system):
    from acprof.tui.run.docker_diagnostics import inspect_docker
    command = Mock(side_effect=AssertionError('unsupported platform'))
    result = inspect_docker(command, Environment('unknown', system=system))
    assert result.code == 'unsupported_environment'
    command.assert_not_called()


def test_wsl_identity_and_service_guidance(monkeypatch):
    from acprof.tui.run.docker_diagnostics import DockerDiagnostic, docker_host
    from acprof.tui.run.docker_guide import repair_steps
    monkeypatch.setattr('platform.freedesktop_os_release', lambda: {
        'ID': 'ubuntu', 'PRETTY_NAME': 'Ubuntu 24.04', 'VERSION_CODENAME': 'noble'})
    host = replace(docker_host(Environment('wsl2', system='Linux')), systemd=False)
    assert host.environment == 'wsl2'
    assert host.distribution == 'ubuntu'
    assert repair_steps(DockerDiagnostic('daemon_unreachable', host))[0].command.startswith('sudo service docker start')
