"""Linux lock lifetime, identity and unlinked pathname protection."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from acprof.host import run_state


@pytest.mark.parametrize('replace', [False, True])
def test_unlinking_held_file_does_not_allow_another_measurement(replace):
    with run_state.MeasurementLock() as held:
        held.path.unlink()
        if replace:
            held.path.touch(mode=0o600)
        with pytest.raises(run_state.RunStateError, match='测量锁'):
            with run_state.MeasurementLock():
                pytest.fail('unlinked pathname allowed overlapping measurements')
    with run_state.MeasurementLock():
        pass


@pytest.mark.parametrize('kind', ['symlink', 'directory', 'hardlink', 'wrong-owner'])
def test_file_validation_still_fails_closed_and_releases_kernel_guard(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(run_state, 'MEASUREMENT_LOCK_ROOT', tmp_path)
    lock = run_state.MeasurementLock()
    target = tmp_path / 'target'
    target.touch()
    if kind == 'symlink':
        lock.path.symlink_to(target)
    elif kind == 'directory':
        lock.path.mkdir()
    elif kind == 'hardlink':
        lock.path.hardlink_to(target)
    else:
        lock.path.touch()
    with monkeypatch.context() as context:
        if kind == 'wrong-owner':
            uid = os.getuid()
            context.setattr(run_state.os, 'getuid', lambda: uid + 1)
        with pytest.raises(run_state.RunStateError, match='测量锁'):
            with lock:
                pytest.fail('malformed lock accepted')
    lock.path.rmdir() if lock.path.is_dir() else lock.path.unlink()
    with run_state.MeasurementLock():
        pass


def test_legacy_file_holder_blocks_new_guard_without_leaking_it():
    import fcntl

    path = run_state.MeasurementLock().path
    with path.open('a+') as legacy:
        fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(run_state.RunStateError, match='测量锁'):
            with run_state.MeasurementLock():
                pytest.fail('legacy file lock ignored')
    with run_state.MeasurementLock():
        pass


@pytest.mark.parametrize('exit_mode', ['normal', 'killed'])
def test_launch_environments_share_identity_and_process_exit_releases_it(tmp_path, exit_mode):
    code = '''
import sys
from pathlib import Path
from acprof.host import run_state
run_state.MEASUREMENT_LOCK_ROOT = Path(sys.argv[1])
run_state.MEASUREMENT_LOCK_NAMESPACE = bytes.fromhex(sys.argv[2])
try:
    with run_state.MeasurementLock():
        print('acquired', flush=True)
        sys.stdin.read(1)
except run_state.RunStateError:
    print('blocked', flush=True)
'''
    first_dir, second_dir = tmp_path / 'ssh-worktree', tmp_path / 'service-output'
    first_dir.mkdir()
    second_dir.mkdir()
    command = [sys.executable, '-c', code, str(run_state.MEASUREMENT_LOCK_ROOT), run_state.MEASUREMENT_LOCK_NAMESPACE.hex()]
    env = {**os.environ, 'PYTHONPATH': str(Path.cwd())}
    first = subprocess.Popen(command, cwd=first_dir, env={**env, 'TMPDIR': str(first_dir), 'XDG_RUNTIME_DIR': str(first_dir), 'SSH_CONNECTION': 'test'}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert first.stdout.readline().strip() == 'acquired'
        second = subprocess.run(command, cwd=second_dir, env={**env, 'TMPDIR': str(second_dir), 'XDG_RUNTIME_DIR': str(second_dir), 'INVOCATION_ID': 'test'}, input='\n', capture_output=True, text=True, timeout=10)
        assert second.returncode == 0, second.stderr
        assert second.stdout.strip() == 'blocked'
        if exit_mode == 'killed':
            first.kill()
        first.communicate('\n' if exit_mode == 'normal' else None, timeout=10)
    finally:
        if first.poll() is None:
            first.kill()
            first.communicate(timeout=10)
    released = subprocess.run(command, env=env, input='\n', capture_output=True, text=True, timeout=10)
    assert released.returncode == 0, released.stderr
    assert released.stdout.strip() == 'acquired'


def test_kernel_identity_survives_a_different_temporary_mount_location(tmp_path, monkeypatch):
    # Simulate service PrivateTmp without changing the kernel network namespace.
    with run_state.MeasurementLock():
        monkeypatch.setattr(run_state, 'MEASUREMENT_LOCK_ROOT', tmp_path)
        with pytest.raises(run_state.RunStateError, match='测量锁'):
            with run_state.MeasurementLock():
                pytest.fail('a second file location split the lock identity')


def test_path_replacement_during_acquisition_fails_and_releases_guard(monkeypatch):
    import fcntl

    lock = run_state.MeasurementLock()
    original = fcntl.flock

    def replace_after_lock(fd, operation):
        original(fd, operation)
        lock.path.unlink()
        lock.path.touch(mode=0o600)

    with monkeypatch.context() as context:
        context.setattr(fcntl, 'flock', replace_after_lock)
        with pytest.raises(run_state.RunStateError, match='pathname changed'):
            lock.__enter__()
    with run_state.MeasurementLock():
        pass
