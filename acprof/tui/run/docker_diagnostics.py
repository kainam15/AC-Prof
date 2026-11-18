"""Read-only Docker evidence for TUI repair guidance, never installation or repair."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from acprof.host.preflight import (
    docker_context_is_docker_desktop,
    docker_host_is_native_socket,
    docker_info_is_docker_desktop,
)
from acprof.platform import Environment
from acprof.tui.i18n import message_id

CLI_LOCATIONS = ('/usr/local/bin/docker', '/usr/bin/docker', '/bin/docker',
                 '/snap/bin/docker', '/opt/docker/bin/docker')


@dataclass(frozen=True, slots=True)
class DockerHost:
    system: str = ''
    distribution: str = ''
    name: str = ''
    codename: str = ''
    environment: str = ''
    systemd: bool = False


@dataclass(frozen=True, slots=True)
class DockerDiagnostic:
    code: str
    host: DockerHost
    cli: str = ''
    context: str = ''
    endpoint: str = ''
    product: str = 'unknown'
    selection_source: str = 'Docker config'
    detail: str = ''
    path_directory: str = ''

    @property
    def summary(self) -> str:
        return message_id('run.docker_repair.reason.' + self.code)

    @property
    def next_step(self) -> str:
        key = {'cli_missing': 'install', 'cli_not_on_path': 'path', 'daemon_unreachable': 'service',
               'socket_permission': 'socket', 'context_unavailable': 'context',
               'context_unsupported': 'context', 'desktop_unsupported': 'desktop',
               'ready': 'ready', 'unsupported_environment': 'unsupported'}.get(self.code, 'unknown')
        return message_id('run.docker_repair.next.' + key)


def docker_host(environment: Environment) -> DockerHost:
    """Read OS identity once in the existing preflight worker, never from a view."""
    system = environment.system or platform.system()
    release: dict[str, str] = {}
    if system == 'Linux':
        try:
            release = platform.freedesktop_os_release()
        except OSError:
            pass  # The unknown-distribution guide links to the official platform list.
    return DockerHost(system, release.get('ID', ''), release.get('PRETTY_NAME', system),
                      release.get('UBUNTU_CODENAME') or release.get('VERSION_CODENAME', ''),
                      environment.environment, Path('/run/systemd/system').is_dir())


def _cli_outside_path() -> str:
    candidates = (*CLI_LOCATIONS, str(Path.home() / '.docker/bin/docker'),
                  str(Path.home() / '.local/bin/docker'))
    return next((path for path in candidates if Path(path).is_file() and os.access(path, os.X_OK)), '')


def inspect_docker(command_runner: Callable[..., subprocess.CompletedProcess[str]],
                   environment: Environment) -> DockerDiagnostic:
    """Resolve the effective endpoint before attempting a bounded local info read.

    Delegate context resolution to the CLI and preserve its environment. An
    invalid context never falls back to default; remote endpoints are not probed.
    """
    host = docker_host(environment)
    evidence = DockerDiagnostic('cli_missing', host)
    if host.system != 'Linux' or (environment.wsl and environment.environment != 'wsl2'):
        return replace(evidence, code='unsupported_environment')
    cli = shutil.which('docker')
    if not cli:
        discovered = _cli_outside_path()
        directory = next((path for path in ('/usr/local/bin', '/usr/bin', '/bin')
                          if path in os.environ.get('PATH', '').split(os.pathsep)), '')
        return replace(evidence, code='cli_not_on_path' if discovered else 'cli_missing',
                       cli=discovered, path_directory=directory)
    evidence = replace(evidence, cli=cli)
    stage = 'context'
    try:
        explicit_context = os.environ.get('DOCKER_CONTEXT', '').strip()
        explicit_host = os.environ.get('DOCKER_HOST', '').strip()
        if explicit_context:
            evidence = replace(evidence, context=explicit_context, selection_source='DOCKER_CONTEXT')
        elif explicit_host:
            evidence = replace(evidence, context='default', endpoint=explicit_host, selection_source='DOCKER_HOST')
        else:
            shown = command_runner((cli, 'context', 'show'), timeout=5.0)
            if shown.returncode != 0 or not shown.stdout.strip():
                return replace(evidence, code='context_unavailable', detail=(shown.stderr or shown.stdout).strip())
            evidence = replace(evidence, context=shown.stdout.strip())
        if not evidence.endpoint:
            inspected = command_runner((cli, 'context', 'inspect', evidence.context, '--format',
                                        '{{(index .Endpoints "docker").Host}}'), timeout=5.0)
            if inspected.returncode != 0 or not inspected.stdout.strip():
                return replace(evidence, code='context_unavailable', detail=(inspected.stderr or inspected.stdout).strip())
            evidence = replace(evidence, endpoint=inspected.stdout.strip())
        if docker_context_is_docker_desktop(evidence.context):
            return replace(evidence, code='desktop_unsupported', product='Docker Desktop')
        if not docker_host_is_native_socket(evidence.endpoint):
            return replace(evidence, code='context_unsupported')
        stage = 'daemon'
        info = command_runner((cli, 'info', '--format', '{{.Name}}|{{.OperatingSystem}}'), timeout=10.0)
        evidence = replace(evidence, detail=(info.stderr or info.stdout).strip())
        if info.returncode != 0:
            detail = evidence.detail.lower()
            code = ('socket_permission' if 'permission denied' in detail or 'access is denied' in detail
                    else 'daemon_unreachable' if any(text in detail for text in (
                        'cannot connect', 'connection refused', 'is the docker daemon running',
                        'timed out', 'timeout', 'no such file', 'connection reset', 'eof'))
                    else 'check_failed')
            return replace(evidence, code=code)
        if docker_info_is_docker_desktop('Name=' + info.stdout):
            return replace(evidence, code='desktop_unsupported', product='Docker Desktop')
        return replace(evidence, code='ready', product='Docker Engine')
    except (OSError, subprocess.TimeoutExpired) as exc:
        code = 'context_unavailable' if stage == 'context' else 'daemon_unreachable'
        if isinstance(exc, FileNotFoundError):
            code = 'cli_missing'
        return replace(evidence, code=code, detail=f'{type(exc).__name__}: {exc}')
