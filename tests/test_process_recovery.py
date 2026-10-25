"""Kernel-backed ownership faults without model downloads or hardware collection."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from acprof.host.runtimes.process_control import recover_process
from acprof.process_identity import same_process


def test_controller_sigkill_reaps_thread_forked_session(tmp_path):
    workload = '''
import json, os, threading, time
from pathlib import Path
def worker():
    if os.fork() == 0:
        os.setsid()
        if os.fork() == 0:
            Path("descendant.pid").write_text(str(os.getpid()))
            time.sleep(60)
        os._exit(0)
threading.Thread(target=worker).start()
time.sleep(60)
'''
    controller = '''
import json, os, sys, time
from pathlib import Path
from acprof.host.runtimes.process_control import ManagedProcess
directory = Path(sys.argv[1])
os.chdir(directory)
child = ManagedProcess([sys.executable, "-c", sys.argv[2]], python=sys.executable,
                       env=dict(os.environ), directory=directory, name="owned")
print(json.dumps(child.evidence()), flush=True)
time.sleep(60)
'''
    parent = subprocess.Popen([sys.executable, '-c', controller, str(tmp_path), workload],
                              stdout=subprocess.PIPE, text=True)
    try:
        evidence = json.loads(parent.stdout.readline())
        deadline = time.monotonic() + 10
        while not (tmp_path / 'descendant.pid').exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert (tmp_path / 'descendant.pid').exists()
        from acprof.process_identity import process_identity
        descendant = process_identity(int((tmp_path / 'descendant.pid').read_text()))
        parent.send_signal(signal.SIGKILL)
        parent.wait(5)
        status = {}
        while time.monotonic() < deadline:
            status = json.loads(Path(evidence['status_file']).read_text())
            if status.get('cleanup_complete'):
                break
            time.sleep(.02)
        assert status.get('cleanup_complete'), status
        assert not same_process(descendant)
        assert not same_process(evidence['workload'])
        recover_process(evidence)
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(5)
        parent.stdout.close()


def test_recovery_rejects_live_owner_and_unproven_cleanup(tmp_path):
    import pytest

    from acprof.host.runtimes.process_control import ProcessCleanupError
    from acprof.process_identity import process_identity
    identity = process_identity(os.getpid())
    evidence = {'owner': identity, 'workload': identity, 'status_file': str(tmp_path / 'missing')}
    with pytest.raises(ProcessCleanupError, match='supervisor'):
        recover_process(evidence)
    evidence['owner'] = {**identity, 'start_ticks': 0}
    with pytest.raises(ProcessCleanupError, match='cleanup'):
        recover_process(evidence)


def test_missing_startup_status_is_not_forgotten_on_resume(tmp_path):
    import pytest

    from acprof.host.runtimes.process_control import ProcessCleanupError, verify_process_cleanup
    (tmp_path / 'starting.process.owner.json').write_text(json.dumps({
        'launch_state': 'started', 'owner_pid': 123, 'status_file': 'starting.process.json'}))
    with pytest.raises(ProcessCleanupError, match='cleanup evidence'):
        verify_process_cleanup(tmp_path)
