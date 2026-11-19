"""Cgroup discovery and delegation diagnostics shared by Process enforcement.

Only a writable delegated subtree may be changed. No privileged commands, unit
changes, migrations of other processes or system-wide controller writes run here.
"""
from __future__ import annotations

import errno
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from acprof.host.preflight import detect_cgroup_version
from acprof.messages import message

CGROUP_ROOT = Path('/sys/fs/cgroup')
PROC_CGROUP = Path('/proc/self/cgroup')
MOUNTINFO = Path('/proc/self/mountinfo')

REASONS = {
    'permission_denied': '当前用户没有所需的 cgroup 委派权限。',
    'controller_missing': '系统未提供本次配额需要的 CPU 或 Memory 控制器。',
    'delegation_missing': '所需控制器尚未委派到实验 cgroup。',
    'parent_limit': '请求的配额超过可见父级限制或 CPU 可用范围。',
    'unsupported': '当前系统不支持 Process 精确配额所需的 cgroup v2。',
    'verification_failed': '资源配额或进程归属验证失败，不能按该配额采集。',
    'configuration_error': 'Process 资源配置无效，请检查 cgroup 路径和配额。',
}


@dataclass(frozen=True)
class ResourceDiagnostic:
    code: str = 'ready'
    version: str = 'unknown'
    current: str = ''
    parent: str = ''
    controllers: tuple[str, ...] = ()
    enabled: tuple[str, ...] = ()
    required: tuple[str, ...] = ()
    writable: bool = False
    repaired: bool = False
    detail: str = ''
    systemd: str = ''
    guidance: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        if self.code == 'inherited':
            return message('CPU / 内存继承上级限制，不承诺数值配额或资源独占。')
        if self.code == 'ready':
            return message('Process 资源权限和配额验证已通过。')
        return message(REASONS.get(self.code, REASONS['configuration_error']))


class ProcessResourceError(ValueError):
    def __init__(self, diagnostic: ResourceDiagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(diagnostic.summary)


def mounts() -> tuple[tuple[Path, Path], ...]:
    def unescape(value: str) -> str:
        return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), value)
    found = []
    for line in MOUNTINFO.read_text().splitlines():
        before, after = line.split(' - ', 1)
        if after.split()[0] == 'cgroup2':
            fields = before.split()
            found.append((Path(unescape(fields[3])), Path(unescape(fields[4]))))
    return tuple(found)


def current_cgroup() -> Path | None:
    return process_cgroup(PROC_CGROUP)


def process_cgroup(membership: Path) -> Path | None:
    for line in membership.read_text().splitlines():
        if not line.startswith('0::'):
            continue
        relative = Path(line[3:])
        if '..' in relative.parts:
            return None
        for root, mount in mounts():
            if relative.is_relative_to(root):
                candidate = mount / relative.relative_to(root)
                if candidate.is_dir():
                    return candidate
    return None


def ancestors(parent: Path) -> Iterator[Path]:
    visible = [mount for _, mount in mounts() if parent.is_relative_to(mount)]
    if not visible:
        raise ValueError('ACPROF_CGROUP_ROOT must be on a cgroup v2 filesystem')
    mount = max(visible, key=lambda item: len(item.parts))
    for directory in (parent, *parent.parents):
        if not directory.is_relative_to(mount):
            break
        yield directory


def delegated(directory: Path) -> bool:
    # A writable mount root is not evidence of systemd delegation, even as root.
    return (directory not in {mount for _, mount in mounts()}
            and all(os.access(path, os.W_OK) for path in
                    (directory, directory / 'cgroup.procs', directory / 'cgroup.subtree_control')))


def inspect(cpu: int | None, memory: int | None, current: Path | None) -> ResourceDiagnostic:
    version = detect_cgroup_version(cgroup_root=str(CGROUP_ROOT), proc_self_cgroup_path=str(PROC_CGROUP))
    required = tuple(name for name, value in (('cpu', cpu), ('memory', memory)) if value is not None)
    result = ResourceDiagnostic(version=version, current=str(current or ''), required=required)
    if version != 'v2':
        return replace(result, code='unsupported')
    explicit = os.environ.get('ACPROF_CGROUP_ROOT')
    parent = Path(explicit).resolve() if explicit else current
    if parent is None:
        return replace(result, code='unsupported')
    parents = tuple(ancestors(parent))
    # Prefer existing delegation; never walk outside a configured root to repair it.
    if not explicit:
        candidates = []
        for candidate in parents:
            candidates.append(candidate)
            # Do not cross into the systemd user manager's independently owned tree.
            if candidate.name.endswith(('.service', '.scope')):
                break
        parent = next((p for p in candidates if delegated(p)), parent)
    result = replace(result, parent=str(parent), writable=delegated(parent))
    try:
        available = tuple((parent / 'cgroup.controllers').read_text().split())
        enabled = tuple((parent / 'cgroup.subtree_control').read_text().split())
    except OSError as exc:
        return failure(result, exc).diagnostic
    result = replace(result, controllers=available, enabled=enabled)
    supported = (tuple(ancestors(parent))[-1] / 'cgroup.controllers').read_text().split()
    if not set(required) <= set(supported):
        return replace(result, code='controller_missing')
    if not result.writable:
        return replace(result, code='permission_denied')
    if not set(required) <= set(available):
        return replace(result, code='delegation_missing')
    if not set(required) <= set(enabled) and (parent / 'cgroup.procs').read_text().strip() and not _can_move_self(parent):
        return replace(result, code='delegation_missing', detail='delegated parent contains other processes; no processes were moved')
    return result


def _can_move_self(parent: Path) -> bool:
    # The unit owns eventual cleanup of this controller leaf. Never relocate a
    # shell, supervisor, another application or an entire login session.
    return (Path('/run/systemd/system').is_dir()
            and parent.name.endswith(('.service', '.scope')) and current_cgroup() == parent
            and (parent / 'cgroup.procs').read_text().split() == [str(os.getpid())])


def enable_controllers(diagnostic: ResourceDiagnostic) -> ResourceDiagnostic:
    missing = set(diagnostic.required) - set(diagnostic.enabled)
    if not missing:
        return diagnostic
    parent = Path(diagnostic.parent)
    # Recheck immediately before repair; never disable controllers.
    if not delegated(parent):
        raise ProcessResourceError(replace(diagnostic, code='permission_denied'))
    if (parent / 'cgroup.procs').read_text().strip():
        if not _can_move_self(parent):
            raise ProcessResourceError(replace(diagnostic, code='delegation_missing'))
        controller = parent / f'acprof-controller-{os.getpid()}'
        controller.mkdir()  # Never take over an existing subgroup.
        from acprof.system.cleanup import cleanup_on_exit
        with cleanup_on_exit(controller.rmdir, description="Process controller migration", on_error_only=True):
            (controller / 'cgroup.procs').write_text(str(os.getpid()))
        # After migration, keep our occupied leaf even if enabling fails. The
        # unit reaps it on exit; retry can rediscover the same delegated parent.
        if str(os.getpid()) not in (controller / 'cgroup.procs').read_text().split():
            raise ProcessResourceError(replace(diagnostic, code='verification_failed', detail='controller membership mismatch'))
    (parent / 'cgroup.subtree_control').write_text(' '.join('+' + name for name in sorted(missing)))
    enabled = tuple((parent / 'cgroup.subtree_control').read_text().split())
    if not set(diagnostic.required) <= set(enabled):
        raise ProcessResourceError(replace(diagnostic, code='verification_failed', detail='controller enable readback mismatch'))
    return replace(diagnostic, enabled=enabled, repaired=True)


def failure(diagnostic: ResourceDiagnostic, exc: Exception, *, code: str = '') -> ProcessResourceError:
    if isinstance(exc, ProcessResourceError):
        return exc
    if not code:
        code = ('permission_denied' if isinstance(exc, PermissionError) or
                isinstance(exc, OSError) and exc.errno in {errno.EACCES, errno.EPERM, errno.EROFS}
                else 'verification_failed' if isinstance(exc, RuntimeError) else 'configuration_error')
    from acprof.system.cleanup import format_exception_notes
    return ProcessResourceError(replace(diagnostic, code=code, detail=f'{type(exc).__name__}: {format_exception_notes(exc)}'))


def with_guidance(diagnostic: ResourceDiagnostic) -> ResourceDiagnostic:
    """Read systemd state only after failure, outside any measurement window."""
    import shlex
    import shutil
    import subprocess

    from acprof.system.command import run_command

    steps = ['配额未生效；精确配额实验已阻断。完成配置后点击“重新检查权限”，无需重启 AC-Prof。']
    if diagnostic.code == 'parent_limit':
        steps.append('降低 CPU / 内存配额，或请管理员调整上级限额；子 cgroup 无法突破父级限额。')
    elif diagnostic.code == 'unsupported':
        steps.append('请管理员检查内核和 cgroup v2 挂载；cgroup v1 不支持本程序的精确配额。系统级切换可能需要重启主机。')
    else:
        steps.append('需要 CPU / Memory 控制器以及可写的委派目录、cgroup.procs 和 cgroup.subtree_control。')
    state = ''
    if Path('/run/systemd/system').is_dir() and shutil.which('systemctl'):
        unit = next((part for part in reversed(Path(diagnostic.current).parts)
                     if part.endswith(('.service', '.scope'))), '')
        if unit and diagnostic.code not in {'parent_limit', 'unsupported', 'controller_missing'}:
            user_service = f'user@{os.getuid()}.service' in Path(diagnostic.current).parts and unit != f'user@{os.getuid()}.service'
            command = ['systemctl', *(['--user'] if user_service else []), 'show', unit,
                       '--property=Delegate,DelegateControllers,ControlGroup']
            try:
                result = run_command(command, capture_output=True, text=True, timeout=3, check=False)
                state = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
            except (OSError, subprocess.TimeoutExpired) as exc:
                state = f'{type(exc).__name__}: {exc}'
            steps.append(shlex.join(command))
            steps.append('检查当前 unit 的 Delegate=yes 和 CPU / Memory 委派。SSH 无用户 D-Bus 时，请管理员从系统管理器配置；不要仅检查 user@ 服务。')
            if user_service:
                section = 'Service' if unit.endswith('.service') else 'Scope'
                steps.extend((f'systemctl --user set-property --runtime {shlex.quote(unit)} Delegate=yes',
                              f'systemctl --user edit {shlex.quote(unit)}',
                              f'[{section}]\nDelegate=yes' + ('\n# systemd >= 254: DelegateSubgroup=controller' if section == 'Service' else ''),
                              '持久 unit 配置仅对后续启动生效；当前进程可采用下方临时 scope 方案。'))
        if diagnostic.version == 'v2' and diagnostic.code not in {'parent_limit', 'controller_missing'}:
            pid, uid = os.getpid(), os.getuid()
            scope = f'acprof-permissions-{pid}.scope'
            steps.append('以下命令由管理员在另一终端手动执行：为当前 AC-Prof 进程创建临时委派 scope，只迁移该进程，不迁移其他会话进程。需要系统 D-Bus；父级限额仍须重新验证。')
            steps.append(
                "(\nset -eu\nsudo busctl call org.freedesktop.systemd1 /org/freedesktop/systemd1 "
                "org.freedesktop.systemd1.Manager StartTransientUnit 'ssa(sv)a(sa(sv))' "
                f"{scope} fail 3 PIDs au 1 {pid} Delegate b true Slice s system.slice 0\n"
                f"acprof_group=/sys/fs/cgroup$(systemctl show --value --property=ControlGroup {scope})\n"
                f'test "$acprof_group" = /sys/fs/cgroup/system.slice/{scope} || exit 1\n'
                'sudo mkdir "$acprof_group/controller"\n'
                f"printf '%s\\n' {pid} | sudo tee \"$acprof_group/controller/cgroup.procs\" >/dev/null\n"
                "printf '+cpu +memory\\n' | sudo tee \"$acprof_group/cgroup.subtree_control\" >/dev/null\n"
                f'sudo chown {uid} "$acprof_group" "$acprof_group/cgroup.procs" '
                '"$acprof_group/cgroup.subtree_control" "$acprof_group/controller/cgroup.procs"\n)'
            )
    else:
        steps.append('非 systemd 环境：请管理员提供 cgroup v2 委派子树。启动前可设置 ACPROF_CGROUP_ROOT；已有进程可由管理员迁入其 controller 子组后重新检查。')
    if os.environ.get('ACPROF_CGROUP_ROOT'):
        steps.append('当前设置了 ACPROF_CGROUP_ROOT：重新检查仍使用该目录。请管理员修复该目录及迁移路径权限；新 scope 不能覆盖显式目录。')
    steps.append('仅在接受继承上级限制时选择“使用继承限制”：CPU / 内存改为 auto，不再承诺数值配额或独占资源。')
    return replace(diagnostic, systemd=state, guidance=tuple(steps))
