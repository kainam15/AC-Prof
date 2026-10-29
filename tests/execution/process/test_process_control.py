"""Real tiny child processes exercise cleanup without model or Docker dependencies."""
import os
import sys
import time
from types import SimpleNamespace

import pytest

from acprof.host.runtimes.process_control import ManagedProcess
from acprof.process_identity import same_process


def start(tmp_path, code):
    return ManagedProcess([sys.executable, "-c", code], python=sys.executable,
                          env=dict(os.environ), directory=tmp_path, name="test")


def test_exit_code_and_logs_are_retained(tmp_path):
    child = start(tmp_path, "import sys; print('observed', flush=True); sys.exit(7)")
    try:
        assert child.wait(5) == 7
        assert "observed" in child.log_tail()
    finally:
        child.close()
    assert not same_process(child.identity)


def test_cleanup_kills_term_resistant_descendants(tmp_path):
    child = start(tmp_path, "import os,signal,time; os.fork(); signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(90)")
    time.sleep(0.1)
    child.close()
    assert not same_process(child.identity)
    assert not child.state().get("cleanup_error")
    child.close()


def test_timeout_keeps_evidence_and_can_be_cleaned(tmp_path):
    import subprocess
    child = start(tmp_path, "import time; time.sleep(90)")
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            child.wait(0.05)
    finally:
        child.close()
    assert not same_process(child.identity)


def test_pid_reuse_identity_never_matches_live_pid(tmp_path):
    child = start(tmp_path, "import time; time.sleep(90)")
    try:
        assert not same_process({**child.identity, "start_ticks": 0})
        assert same_process(child.identity)
    finally:
        child.close()


def test_supervisor_holds_model_lease_until_cleanup(tmp_path):
    import fcntl

    from acprof.host.model_store.store import ModelStoreMount
    path = tmp_path / 'model.lease'
    stream = path.open('a+b')
    fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
    mount = ModelStoreMount([], stream)
    child = ManagedProcess([sys.executable, '-c', 'import time; time.sleep(90)'],
                           python=sys.executable, env=dict(os.environ), directory=tmp_path,
                           name='lease', pass_fds=mount.pass_fds)
    mount.close()
    try:
        with path.open('a+b') as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        child.close()
    with path.open('a+b') as probe:
        fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_supervision_prerequisites_run_in_a_separate_bounded_helper():
    from acprof.host.runtimes.process_control import require_process_supervision
    assert require_process_supervision() == 'pidfd and child-subreaper available'


def test_supervisor_refuses_to_append_through_log_symlink(tmp_path):
    outside = tmp_path / "outside.log"
    outside.write_text("preserve\n")
    (tmp_path / "test.log").symlink_to(outside)

    def start_and_close():
        child = start(tmp_path, "print('redirected-output', flush=True)")
        try:
            child.wait(5)
        finally:
            child.close()

    with pytest.raises(RuntimeError, match="symbolic links"):
        start_and_close()

    assert outside.read_text() == "preserve\n"


def test_supervisor_does_not_follow_fixed_status_temp_symlink(tmp_path):
    outside = tmp_path / "outside-status.json"
    outside.write_text("preserve")
    (tmp_path / "test.process.tmp").symlink_to(outside)

    child = start(tmp_path, "print('status-test', flush=True)")
    try:
        assert child.wait(5) == 0
    finally:
        child.close()

    assert outside.read_text() == "preserve"
    assert not (tmp_path / "test.process.json").is_symlink()


def test_log_tail_rejects_symlinked_diagnostic_file(tmp_path):
    outside = tmp_path / "outside.log"
    outside.write_text("diagnostic-secret")
    path = tmp_path / "process.log"
    path.symlink_to(outside)

    observed = ManagedProcess.log_tail(SimpleNamespace(log_path=path))

    assert "diagnostic-secret" not in observed
    assert "unavailable" in observed


def test_log_tail_is_bounded_when_the_log_grows_during_read(tmp_path, monkeypatch):
    path = tmp_path / "growing.log"
    original = b"a" * 16384
    path.write_bytes(original)
    from acprof.host.runtimes import process_control

    read_fd = os.read
    grew = False

    def read_growing_log(descriptor, size):
        nonlocal grew
        if not grew:
            with path.open("ab") as writer:
                writer.write(b"new diagnostics" * 10000)
            grew = True
        return read_fd(descriptor, size)

    monkeypatch.setattr(process_control.os, "read", read_growing_log)
    tail = ManagedProcess.log_tail(SimpleNamespace(log_path=path))
    assert tail == original.decode()
    assert path.stat().st_size > len(original)


def test_log_tail_uses_the_open_file_when_the_path_is_replaced(tmp_path, monkeypatch):
    path = tmp_path / "rotating.log"
    retained = tmp_path / "retained.log"
    path.write_bytes(b"old prefix" * 10000 + b"tail" * 4096)
    from acprof.host.runtimes import process_control

    seek_fd = os.lseek
    rotated = False

    def rotate_after_size(descriptor, offset, whence):
        nonlocal rotated
        result = seek_fd(descriptor, offset, whence)
        if whence == os.SEEK_END and not rotated:
            path.rename(retained)
            path.write_bytes(b"new")
            rotated = True
        return result

    monkeypatch.setattr(process_control.os, "lseek", rotate_after_size)
    assert ManagedProcess.log_tail(SimpleNamespace(log_path=path)) == "tail" * 4096
