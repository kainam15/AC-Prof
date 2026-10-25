"""Real tiny child processes exercise cleanup without model or Docker dependencies."""
import os
import sys
import time

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

    from acprof.host.model_store import ModelStoreMount
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
