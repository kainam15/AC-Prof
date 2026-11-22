"""Resource enforcement failures must be actionable and cannot claim verification."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from acprof.host.runtimes import resources


def test_permission_error_is_actionable_and_preserves_exact_quotas(hierarchy, monkeypatch):
    monkeypatch.setattr(Path, 'mkdir', Mock(side_effect=PermissionError(13, 'Permission denied')))
    with pytest.raises(resources.ProcessResourceError) as error:
        resources.ProcessLimits(1, None)
    assert 'Permission denied' not in str(error.value)
    assert '权限' in str(error.value)
    assert error.value.diagnostic.code == 'permission_denied'


def test_preflight_verifies_writes_and_membership_before_cleanup(monkeypatch):
    limit = Mock(diagnostic=resources.ResourceDiagnostic())
    monkeypatch.setattr(resources, 'ProcessLimits', Mock(return_value=limit))
    resources.preflight_resources([1], [2], '')
    limit.verify_probe.assert_called_once_with()
    limit.close.assert_called_once_with()


def test_failed_verification_cannot_leave_previous_verified_evidence(monkeypatch):
    monkeypatch.setattr(resources, 'inherited_resources', lambda: {'affinity': [0]})
    limit = resources.ProcessLimits(None, None)
    limit.verify(123)
    assert limit.evidence()['verified']
    limit.path = '/no-such-cgroup'
    with pytest.raises((RuntimeError, OSError, ValueError)):
        limit.verify(123)
    assert not limit.evidence()['verified']


@pytest.fixture
def hierarchy(tmp_path, monkeypatch, request):
    """A synthetic kernel hierarchy: mkdir creates virtual files; rmdir removes them."""
    from acprof.host.runtimes import cgroups
    root = tmp_path / 'cgroup'
    root.mkdir()
    parent = root / getattr(request, 'param', 'delegated')
    parent.mkdir()
    current = parent / 'controller'
    current.mkdir()
    for path in (root, parent, current):
        for name, text in {'cgroup.controllers': 'cpu memory', 'cgroup.subtree_control': 'cpu memory',
                           'cgroup.procs': '', 'cpu.max': 'max 100000', 'memory.max': 'max'}.items():
            (path / name).write_text(text)
    proc = tmp_path / 'proc-cgroup'
    proc.write_text(f'0::/{parent.name}/controller\n')
    mountinfo = tmp_path / 'mountinfo'
    mountinfo.write_text(f'37 28 0:31 / {root} rw - cgroup2 cgroup2 rw\n')
    monkeypatch.setattr(cgroups, 'CGROUP_ROOT', root)
    monkeypatch.setattr(cgroups, 'PROC_CGROUP', proc)
    monkeypatch.setattr(cgroups, 'MOUNTINFO', mountinfo)
    monkeypatch.setattr(resources.os, 'sched_getaffinity', lambda pid: {0, 1, 2, 3})
    monkeypatch.setenv('ACPROF_CGROUP_ROOT', str(parent))
    mkdir, rmdir, write = Path.mkdir, Path.rmdir, Path.write_text
    def create(path, *args, **kwargs):
        mkdir(path, *args, **kwargs)
        if path.parent == parent and path.name.startswith('acprof-'):
            for field, value in {'cpu.max': 'max 100000', 'memory.max': 'max', 'cgroup.procs': '',
                                 'memory.events': 'oom_kill 0'}.items():
                write(path / field, value)
    def remove(path, *args, **kwargs):
        if path.parent == parent and path.name.startswith('acprof-'):
            for file in path.iterdir():
                file.unlink()
        rmdir(path, *args, **kwargs)
    def write_control(path, text, *args, **kwargs):
        if path.name == 'cgroup.subtree_control' and text.startswith('+'):
            text = ' '.join(sorted(set(path.read_text().split()) | {part[1:] for part in text.split()}))
        return write(path, text, *args, **kwargs)
    monkeypatch.setattr(Path, 'mkdir', create)
    monkeypatch.setattr(Path, 'rmdir', remove)
    monkeypatch.setattr(Path, 'write_text', write_control)
    return root, parent, current


def test_delegated_limits_read_back_and_real_probe_membership(hierarchy):
    _, parent, _ = hierarchy
    limits = resources.ProcessLimits(2, 1)
    owned = Path(limits.path)
    assert owned.parent == parent
    assert (owned / 'cpu.max').read_text() == '200000 100000'
    assert (owned / 'memory.max').read_text() == '1073741824'
    limits.verify_probe()  # harmless Python child using fixture files, no model
    assert not limits.evidence()['verified']
    (owned / 'cgroup.procs').write_text('123\n')
    limits.verify(123)
    assert limits.evidence()['verified']
    limits.close()
    assert not owned.exists()


def test_empty_delegation_is_safely_repaired_and_rechecked(hierarchy):
    _, parent, _ = hierarchy
    (parent / 'cgroup.subtree_control').write_text('memory')
    report = resources.preflight_resources([1], [None], '')
    assert report.repaired
    assert set((parent / 'cgroup.subtree_control').read_text().split()) == {'cpu', 'memory'}
    assert not list(parent.glob('acprof-*'))


@pytest.mark.parametrize(('condition', 'code'), [
    ('denied', 'permission_denied'), ('partial', 'permission_denied'),
    ('no_cpu', 'controller_missing'), ('undelegated', 'delegation_missing'),
    ('occupied', 'delegation_missing'), ('cpu_parent', 'parent_limit'),
    ('memory_parent', 'parent_limit'), ('v1', 'unsupported'),
])
def test_unenforceable_limits_are_distinguished_before_mutation(hierarchy, monkeypatch, condition, code):
    from acprof.host.runtimes import cgroups
    root, parent, _ = hierarchy
    if condition in {'denied', 'partial'}:
        access = cgroups.os.access
        monkeypatch.setattr(cgroups.os, 'access', lambda path, mode: False if (
            condition == 'denied' or Path(path).name == 'cgroup.subtree_control') else access(path, mode))
    elif condition == 'no_cpu':
        (root / 'cgroup.controllers').write_text('memory')
    elif condition == 'undelegated':
        (parent / 'cgroup.controllers').write_text('memory')
    elif condition == 'occupied':
        (parent / 'cgroup.subtree_control').write_text('')
        (parent / 'cgroup.procs').write_text('99999')
    elif condition == 'cpu_parent':
        (root / 'cpu.max').write_text('100000 100000')
    elif condition == 'memory_parent':
        (root / 'memory.max').write_text('536870912')
    elif condition == 'v1':
        (root / 'cgroup.controllers').unlink()
        cgroups.PROC_CGROUP.write_text('1:cpu,memory:/legacy')
    with pytest.raises(cgroups.ProcessResourceError) as error:
        resources.ProcessLimits(2, 1)
    assert error.value.diagnostic.code == code
    assert not list(parent.glob('acprof-*'))


def test_cpu_only_does_not_require_memory_controller(hierarchy):
    root, parent, _ = hierarchy
    for path in (root, parent):
        (path / 'cgroup.controllers').write_text('cpu')
        (path / 'cgroup.subtree_control').write_text('cpu')
    limits = resources.ProcessLimits(1, None)
    assert limits.diagnostic.required == ('cpu',)
    limits.close()


def test_quota_readback_mismatch_cleans_owned_group(hierarchy, monkeypatch):
    from acprof.host.runtimes.cgroups import ProcessResourceError
    _, parent, _ = hierarchy
    write = Path.write_text
    def ignore_quota(path, value, *args, **kwargs):
        return 0 if path.name == 'cpu.max' and path.parent.name.startswith('acprof-') else write(path, value, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', ignore_quota)
    with pytest.raises(ProcessResourceError) as error:
        resources.ProcessLimits(1, None)
    assert error.value.diagnostic.code == 'verification_failed'
    assert not list(parent.glob('acprof-*'))


def test_membership_failure_blocks_verification(hierarchy):
    from acprof.host.runtimes.cgroups import ProcessResourceError
    limits = resources.ProcessLimits(1, 1)
    with pytest.raises(ProcessResourceError):
        limits.verify(123)
    assert not limits.evidence()['verified']
    limits.close()


def test_find_existing_delegated_ancestor_without_changing_current_group(hierarchy, monkeypatch):
    from acprof.host.runtimes import cgroups
    _, parent, current = hierarchy
    monkeypatch.delenv('ACPROF_CGROUP_ROOT')
    access = cgroups.os.access
    monkeypatch.setattr(cgroups.os, 'access', lambda path, mode: False if Path(path) == current else access(path, mode))
    resources.preflight_resources([1], [1], '')
    assert cgroups.current_cgroup() == current
    assert (current / 'cgroup.procs').read_text() == ''
    assert not list(parent.glob('acprof-*'))


@pytest.mark.parametrize('hierarchy', ['app.service'], indirect=True)
@pytest.mark.parametrize(('mode', 'status', 'code', 'picked'), [
    ('automatic', 'ok', 'ready', 'app.service'),
    ('already-has-workers', 'ok', 'ready', 'app.service'),
    ('explicit-parent', 'ok', 'ready', 'app.service'),
    ('explicit-leaf', 'fail', 'delegation_missing', 'controller'),
    ('ancestor-capped', 'fail', 'parent_limit', 'app.service'),
    ('leaf-capped', 'fail', 'parent_limit', 'app.service'),
    ('leaf-limit-unreadable', 'fail', 'permission_denied', 'app.service'),
])
def test_process_preflight_chooses_a_usable_parent_for_a_populated_leaf(
    hierarchy, monkeypatch, mode, status, code, picked,
):
    from dataclasses import replace

    from acprof.experiments.configuration import RunConfig
    from acprof.host.runtimes import cgroups
    from acprof.tui.run.diagnostics import quick_preflight

    root, parent, current = hierarchy
    monkeypatch.delenv('ACPROF_CGROUP_ROOT')
    if mode in {'explicit-parent', 'explicit-leaf'}:
        monkeypatch.setenv('ACPROF_CGROUP_ROOT', str(parent if mode == 'explicit-parent' else current))
    if mode in {'ancestor-capped', 'leaf-capped'}:
        capped = parent if mode == 'ancestor-capped' else current
        (capped / 'memory.max').write_text(str(512 * 1024**2))
    elif mode == 'leaf-limit-unreadable':
        read = Path.read_text

        def read_limit(path, *args, **kwargs):
            if path == current / 'memory.max':
                raise PermissionError(13, 'fixture caller quota unreadable', str(path))
            return read(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'read_text', read_limit)
    members = str(cgroups.os.getpid()) + ('\n99999' if mode == 'already-has-workers' else '')
    (current / 'cgroup.procs').write_text(members)
    (current / 'cgroup.subtree_control').write_text('')
    monkeypatch.setattr('acprof.host.runtimes.process_control.require_process_supervision', lambda: None)
    monkeypatch.setattr('acprof.host.runtimes.capabilities.probe_collection_plan', lambda *a, **kw: {'probes': {}})
    monkeypatch.setattr(cgroups, 'with_guidance', lambda diagnostic: diagnostic)
    config = replace(RunConfig.smoke('fixture/model'), runtime='process', profiling_mode='basic',
                     cpus='2', mems='1', gpus='off')
    row, = (row for row in quick_preflight(config, project_dir=root) if row.check_id == 'process_resources')
    assert row.status == status
    assert row.resource_diagnostic.code == code
    assert Path(row.resource_diagnostic.parent).name == picked
    assert (current / 'cgroup.procs').read_text() == members
    assert (current / 'cgroup.subtree_control').read_text() == ''
    assert not list(root.rglob('acprof-*'))
    if mode == 'explicit-leaf':
        assert 'other processes' not in row.resource_diagnostic.detail
    if mode == 'leaf-limit-unreadable':
        assert str(current / 'memory.max') in row.resource_diagnostic.detail


@pytest.mark.parametrize('hierarchy', ['app.service'], indirect=True)
def test_missing_controller_on_leaf_can_use_empty_delegation_root(hierarchy, monkeypatch):
    from acprof.host.runtimes import cgroups
    _, parent, current = hierarchy
    monkeypatch.delenv('ACPROF_CGROUP_ROOT')
    (parent / 'cgroup.subtree_control').write_text('memory')
    (current / 'cgroup.controllers').write_text('memory')
    (current / 'cgroup.subtree_control').write_text('')
    (current / 'cgroup.procs').write_text(str(cgroups.os.getpid()))
    diagnostic = resources.preflight_resources([1], [None], '')
    assert diagnostic.parent == str(parent)
    assert diagnostic.repaired
    assert set((parent / 'cgroup.subtree_control').read_text().split()) == {'cpu', 'memory'}


def test_unreadable_caller_memory_limit_does_not_block_cpu_only(hierarchy, monkeypatch):
    from acprof.host.runtimes import cgroups
    _, parent, current = hierarchy
    monkeypatch.delenv('ACPROF_CGROUP_ROOT')
    (current / 'cgroup.subtree_control').write_text('')
    (current / 'cgroup.procs').write_text(str(cgroups.os.getpid()))
    read = Path.read_text

    def read_cpu_only(path, *args, **kwargs):
        if path == current / 'memory.max':
            raise PermissionError(13, 'fixture unreadable memory limit', str(path))
        return read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read_cpu_only)
    diagnostic = resources.preflight_resources([1], [None], '')
    assert diagnostic.code == 'ready'
    assert diagnostic.parent == str(parent)


@pytest.mark.parametrize('hierarchy', ['user@1000.service', 'system.slice', 'init.scope'], indirect=True)
def test_automatic_discovery_excludes_manager_owned_ancestors(hierarchy, monkeypatch):
    from acprof.host.runtimes import cgroups
    root, manager, current = hierarchy
    monkeypatch.delenv('ACPROF_CGROUP_ROOT')
    access = cgroups.os.access
    monkeypatch.setattr(cgroups.os, 'access', lambda path, mode:
                        not Path(path).is_relative_to(current) and access(path, mode))
    diagnostic = cgroups.inspect(1, 1, current)
    assert diagnostic.code == 'permission_denied'
    assert diagnostic.parent == str(current)
    assert not list(root.rglob('acprof-*'))
    assert (manager / 'cgroup.procs').read_text() == ''


def test_inherited_mode_needs_no_controllers_or_repair(monkeypatch):
    monkeypatch.setattr(resources, 'inherited_resources', lambda: {'affinity': [0]})
    monkeypatch.setattr('acprof.host.runtimes.cgroups.inspect', Mock(side_effect=AssertionError('no quota checks')))
    report = resources.preflight_resources([None], [None], '')
    assert report.code == 'inherited'


def test_systemd_self_only_delegation_can_create_controller_leaf(hierarchy, monkeypatch):
    from acprof.host.runtimes import cgroups
    root, parent, _ = hierarchy
    is_dir = Path.is_dir
    monkeypatch.setattr(Path, 'is_dir', lambda path: True if str(path) == '/run/systemd/system' else is_dir(path))
    # A standalone systemd user service contains only the application PID.
    unit = root / 'acprof.service'
    parent.rename(unit)
    cgroups.PROC_CGROUP.write_text('0::/acprof.service')
    monkeypatch.setenv('ACPROF_CGROUP_ROOT', str(unit))
    (unit / 'cgroup.procs').write_text(str(cgroups.os.getpid()))
    (unit / 'cgroup.subtree_control').write_text('')
    write = Path.write_text
    def migrate(path, text, *args, **kwargs):
        if path.name == 'cgroup.procs' and path.parent.name.startswith('acprof-controller-'):
            write(unit / 'cgroup.procs', '')
        return write(path, text, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', migrate)
    before = cgroups.inspect(1, 1, unit)
    assert before.code == 'ready'
    after = cgroups.enable_controllers(before)
    assert after.repaired
    assert (unit / 'cgroup.procs').read_text() == ''
    leaf = unit / f'acprof-controller-{cgroups.os.getpid()}'
    assert (leaf / 'cgroup.procs').read_text() == str(cgroups.os.getpid())
    assert set((unit / 'cgroup.subtree_control').read_text().split()) == {'cpu', 'memory'}


def test_never_escapes_current_systemd_unit_to_use_user_manager(hierarchy, monkeypatch):
    from acprof.host.runtimes import cgroups
    _, parent, current = hierarchy
    unit = parent / 'app.service'
    current.rename(unit)
    cgroups.PROC_CGROUP.write_text('0::/delegated/app.service')
    monkeypatch.delenv('ACPROF_CGROUP_ROOT')
    access = cgroups.os.access
    monkeypatch.setattr(cgroups.os, 'access', lambda path, mode: False if Path(path).is_relative_to(unit) else access(path, mode))
    report = cgroups.inspect(1, 1, unit)
    assert report.code == 'permission_denied'
    assert report.parent == str(unit)


def test_probe_failure_cleans_group_and_retains_original_quotas(hierarchy, monkeypatch):
    import subprocess

    from acprof.host.runtimes.cgroups import ProcessResourceError
    _, parent, _ = hierarchy
    monkeypatch.setattr('acprof.system.command.run_command', lambda *args, **kwargs:
                        subprocess.CompletedProcess([], 13, '', 'fixture migration denied'))
    with pytest.raises(ProcessResourceError) as error:
        resources.preflight_resources([2], [1], '')
    assert error.value.diagnostic.code == 'permission_denied'
    assert not list(parent.glob('acprof-*'))


@pytest.mark.parametrize('session', ['ssh', 'user-service', 'no-systemd'])
def test_guidance_only_reads_state_and_never_runs_administrator_commands(monkeypatch, session):
    import subprocess

    from acprof.host.runtimes import cgroups
    monkeypatch.delenv('ACPROF_CGROUP_ROOT', raising=False)
    is_dir = Path.is_dir
    monkeypatch.setattr(Path, 'is_dir', lambda path: session != 'no-systemd' if str(path) == '/run/systemd/system' else is_dir(path))
    monkeypatch.setattr('shutil.which', lambda name: '/usr/bin/' + name)
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        assert command[0] == 'systemctl'
        assert 'show' in command
        assert kwargs['timeout'] == 3
        return subprocess.CompletedProcess(command, 1 if session == 'ssh' else 0,
                                           'Delegate=no\n', 'Failed to connect to bus' if session == 'ssh' else '')
    monkeypatch.setattr('acprof.system.command.run_command', run)
    current = (f'/sys/fs/cgroup/user.slice/user-{cgroups.os.getuid()}.slice/user@{cgroups.os.getuid()}.service/app.slice/acprof.service'
               if session == 'user-service' else '/sys/fs/cgroup/user.slice/session-3.scope')
    report = cgroups.with_guidance(cgroups.ResourceDiagnostic(code='permission_denied', version='v2', current=current))
    guidance = '\n'.join(report.guidance)
    if session == 'no-systemd':
        assert not commands
        assert 'ACPROF_CGROUP_ROOT' in guidance
    else:
        assert len(commands) == 1
        assert ('--user' in commands[0]) == (session == 'user-service')
        assert 'Delegate=yes' in guidance
        assert 'sudo busctl' in guidance
        assert 'Slice s system.slice' in guidance
        assert 'set -eu' in guidance
    assert '重新检查' in guidance


def test_partial_read_permissions_preserve_current_path_for_guidance(hierarchy, monkeypatch):
    from acprof.host.runtimes import cgroups
    _, parent, current = hierarchy
    read = Path.read_text
    def denied(path, *args, **kwargs):
        if path == parent / 'cgroup.controllers':
            raise PermissionError(13, 'Permission denied')
        return read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', denied)
    report = cgroups.inspect(1, 1, current)
    assert report.code == 'permission_denied'
    assert report.current == str(current)
    assert report.parent == str(parent)


def test_requested_affinity_cannot_silently_reduce_numeric_cpu_quota(hierarchy):
    with pytest.raises(resources.ProcessResourceError) as error:
        resources.ProcessLimits(2, None, '0')
    assert error.value.diagnostic.code == 'parent_limit'


def test_workload_affinity_change_invalidates_previous_quota_evidence(hierarchy, monkeypatch):
    limits = resources.ProcessLimits(2, None)
    (Path(limits.path) / 'cgroup.procs').write_text('123')
    limits.verify(123)
    monkeypatch.setattr(resources.os, 'sched_getaffinity', lambda pid: {0})
    with pytest.raises(resources.ProcessResourceError) as error:
        limits.verify(123)
    assert error.value.diagnostic.code == 'verification_failed'
    assert not limits.evidence()['verified']
    limits.close()


def test_parent_limit_change_invalidates_previous_quota_evidence(hierarchy):
    _, parent, _ = hierarchy
    limits = resources.ProcessLimits(2, 1)
    (Path(limits.path) / 'cgroup.procs').write_text('123')
    limits.verify(123)
    (parent / 'memory.max').write_text('536870912')
    with pytest.raises(resources.ProcessResourceError) as error:
        limits.verify(123)
    assert error.value.diagnostic.code == 'parent_limit'
    assert not limits.evidence()['verified']
    limits.close()
