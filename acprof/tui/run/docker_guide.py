"""OS-specific, copy-only repair plans projected from cached Docker evidence."""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

from acprof.tui.i18n import message_id
from acprof.tui.run.docker_diagnostics import DockerDiagnostic

INSTALL_URL = 'https://docs.docker.com/engine/install/'
CONTEXT_URL = 'https://docs.docker.com/engine/manage-resources/contexts/'
ACCESS_URL = INSTALL_URL + 'linux-postinstall/'
VERIFY_COMMAND = "docker context show\ndocker context ls\ndocker info --format '{{.Name}}|{{.OperatingSystem}}'"


@dataclass(frozen=True, slots=True)
class DockerStep:
    key: str
    command: str = ''
    url: str = ''

    @property
    def title(self) -> str:
        return message_id('run.docker_repair.step.' + self.key)

    @property
    def note(self) -> str:
        return message_id('run.docker_repair.note.' + self.key)


def _repository_command(distribution: str, codename: str) -> str:
    # Only exact distro IDs and simple, detected release names reach this
    # template. noclobber refuses existing files, including symlinks. No remote
    # shell script, forced removal, service enablement, or context mutation.
    return f'''(
set -eu
for file in /etc/apt/keyrings/docker.asc /etc/apt/sources.list.d/docker.sources /etc/apt/sources.list.d/docker.list; do
  if [ -e "$file" ] || [ -L "$file" ]; then
    printf 'Existing Docker configuration; review first: %s\\n' "$file" >&2
    exit 1
  fi
done
sudo apt-get update
sudo apt-get install ca-certificates curl
acprof_docker_key=$(mktemp)
trap 'rm -f "$acprof_docker_key"' EXIT
curl -fsSL https://download.docker.com/linux/{distribution}/gpg -o "$acprof_docker_key"
sudo install -m 0755 -d /etc/apt/keyrings
sudo sh -c 'set -C; umask 022; cat > /etc/apt/keyrings/docker.asc' < "$acprof_docker_key"
sudo sh -c 'set -C; umask 022; cat > /etc/apt/sources.list.d/docker.sources' <<EOF
Types: deb
URIs: https://download.docker.com/linux/{distribution}
Suites: {codename}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt-get update
)'''


def repair_steps(diagnostic: DockerDiagnostic) -> tuple[DockerStep, ...]:
    """No host reads: opening, scrolling and copying cannot trigger checks."""
    code, host = diagnostic.code, diagnostic.host
    install_url = INSTALL_URL + host.distribution + '/' if host.distribution in {'ubuntu', 'debian'} else INSTALL_URL
    if code == 'unsupported_environment':
        return (DockerStep('unsupported', url=INSTALL_URL),)
    if code == 'cli_missing':
        if host.distribution in {'ubuntu', 'debian'} and re.fullmatch(r'[a-z][a-z0-9-]*', host.codename):
            steps = [DockerStep('repository', _repository_command(host.distribution, host.codename), install_url),
                     DockerStep('install', 'sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin', install_url)]
        else:
            steps = [DockerStep('other_linux', 'cat /etc/os-release', install_url)]
        steps.append(_service_step(diagnostic))
    elif code == 'cli_not_on_path':
        command = (f'sudo ln -s -- {shlex.quote(diagnostic.cli)} {shlex.quote(diagnostic.path_directory + "/docker")}'
                   if diagnostic.path_directory else f'export PATH={shlex.quote(diagnostic.cli.rsplit("/", 1)[0])}:"$PATH"')
        steps = [DockerStep('path' if diagnostic.path_directory else 'path_restart', command, install_url)]
    elif code == 'socket_permission':
        steps = [DockerStep('socket', 'id\nls -l /var/run/docker.sock\ngetent group docker', ACCESS_URL),
                 DockerStep('group', 'sudo groupadd -f docker\nsudo usermod -aG docker "$(id -un)"', ACCESS_URL)]
    elif code == 'daemon_unreachable':
        steps = [_service_step(diagnostic)]
    elif code in {'context_unavailable', 'context_unsupported', 'desktop_unsupported'}:
        steps = [DockerStep('context', 'docker context ls\ndocker context inspect default', CONTEXT_URL)]
        if diagnostic.selection_source in {'DOCKER_HOST', 'DOCKER_CONTEXT'}:
            steps.append(DockerStep('overrides', 'env -u DOCKER_HOST -u DOCKER_CONTEXT docker --context default info', CONTEXT_URL))
        else:
            steps.append(DockerStep('select', 'docker context use default', CONTEXT_URL))
        if code == 'desktop_unsupported':
            steps.append(DockerStep('engine', url=install_url))
    else:
        steps = []
    steps.append(DockerStep('verify', VERIFY_COMMAND))
    return tuple(steps)


def _service_step(diagnostic: DockerDiagnostic) -> DockerStep:
    distribution = diagnostic.host.distribution
    install_url = INSTALL_URL + distribution + '/' if distribution in {'ubuntu', 'debian'} else INSTALL_URL
    if diagnostic.host.systemd:
        return DockerStep('service', 'sudo systemctl start docker\nsystemctl --no-pager --full status docker', install_url)
    if diagnostic.host.distribution in {'ubuntu', 'debian'}:
        return DockerStep('service_legacy', 'sudo service docker start\nservice docker status', install_url)
    return DockerStep('service_other', url=INSTALL_URL)
