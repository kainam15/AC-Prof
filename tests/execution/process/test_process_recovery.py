"""Kernel-backed ownership faults without model downloads or hardware collection."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from acprof.artifacts import MAX_JSON_ARTIFACT_BYTES
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
        deadline = time.monotonic() + 10
        status = {}
        while time.monotonic() < deadline:
            status = json.loads(Path(evidence['status_file']).read_text())
            # Final status is published before the supervisor itself exits.
            if status.get('cleanup_complete') and not same_process(evidence['owner']):
                break
            time.sleep(.02)
        assert status.get('cleanup_complete'), status
        assert not same_process(evidence['owner'])
        assert not same_process(descendant)
        assert not same_process(evidence['workload'])
        recover_process(evidence, directory=tmp_path)
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(5)
        parent.stdout.close()


def test_recovery_rejects_live_owner_and_unproven_cleanup(tmp_path):
    from acprof.host.runtimes.process_control import ProcessCleanupError
    from acprof.process_identity import process_identity
    identity = process_identity(os.getpid())
    evidence = {'owner': identity, 'workload': identity, 'status_file': str(tmp_path / 'missing')}
    with pytest.raises(ProcessCleanupError, match='supervisor'):
        recover_process(evidence, directory=tmp_path)
    evidence['owner'] = {**identity, 'start_ticks': 0}
    with pytest.raises(ProcessCleanupError, match='cleanup'):
        recover_process(evidence, directory=tmp_path)


def _cleanup_evidence(status_file: Path) -> tuple[dict, dict]:
    owner = {"pid": 2147483647, "start_ticks": 1}
    workload = {"pid": 2147483646, "start_ticks": 1}
    evidence = {
        "kind": "process",
        "owner": owner,
        "workload": workload,
        "status_file": str(status_file),
        "log_file": str(status_file.with_suffix(".log")),
    }
    status = {
        "cleanup_complete": True,
        "cleanup_error": "",
        "owner": owner,
        "workload": workload,
    }
    return evidence, status


def test_recovery_rejects_status_outside_expected_directory(tmp_path):
    from acprof.host.runtimes.process_control import ProcessCleanupError

    logs = tmp_path / "logs"
    logs.mkdir()
    path = tmp_path / "outside.process.json"
    evidence, status = _cleanup_evidence(path)
    path.write_text(json.dumps(status))

    with pytest.raises(ProcessCleanupError, match="outside expected directory"):
        recover_process(evidence, directory=logs)


def test_recovery_rejects_symlink_and_non_regular_status(tmp_path):
    from acprof.host.runtimes.process_control import ProcessCleanupError

    logs = tmp_path / "logs"
    logs.mkdir()
    target = tmp_path / "outside.process.json"
    evidence, status = _cleanup_evidence(logs / "owned.process.json")
    target.write_text(json.dumps(status))
    Path(evidence["status_file"]).symlink_to(target)
    with pytest.raises(ProcessCleanupError, match="cleanup evidence is unavailable"):
        recover_process(evidence, directory=logs)

    Path(evidence["status_file"]).unlink()
    Path(evidence["status_file"]).mkdir()
    with pytest.raises(ProcessCleanupError, match="regular file"):
        recover_process(evidence, directory=logs)


@pytest.mark.parametrize(("content", "message"), [
    (b" " * (MAX_JSON_ARTIFACT_BYTES + 1), "4 MiB"),
    (b'{"cleanup_complete":true,"value":NaN}', "invalid process status JSON"),
    (b"[]", "top-level JSON value must be an object"),
])
def test_recovery_rejects_unbounded_or_invalid_status(tmp_path, content, message):
    from acprof.host.runtimes.process_control import ProcessCleanupError

    path = tmp_path / "owned.process.json"
    evidence, _ = _cleanup_evidence(path)
    path.write_bytes(content)
    with pytest.raises(ProcessCleanupError, match=message):
        recover_process(evidence, directory=tmp_path)


def test_startup_recovery_accepts_internal_regular_cleanup_evidence(tmp_path):
    from acprof.host.runtimes.process_control import verify_process_cleanup

    status_path = tmp_path / "owned.process.json"
    _, status = _cleanup_evidence(status_path)
    status_path.write_text(json.dumps(status))
    (tmp_path / "owned.process.owner.json").write_text(json.dumps({
        "launch_state": "started",
        "owner_pid": status["owner"]["pid"],
        "status_file": status_path.name,
    }))

    verify_process_cleanup(tmp_path)


def test_startup_recovery_rejects_symlinked_launch_ledger(tmp_path):
    from acprof.host.runtimes.process_control import ProcessCleanupError, verify_process_cleanup

    target = tmp_path / "outside.json"
    target.write_text(json.dumps({"launch_state": "failed"}))
    (tmp_path / "owned.process.owner.json").symlink_to(target)

    with pytest.raises(ProcessCleanupError, match="complete cleanup evidence"):
        verify_process_cleanup(tmp_path)


def test_startup_recovery_detects_deleted_launch_ledger_from_inventory(tmp_path):
    from acprof.host.runtimes.process_control import ProcessCleanupError, verify_process_cleanup

    (tmp_path / "owned.process.inventory.json").write_text(json.dumps({
        "schema_version": 2,
        "owner_file": "owned.process.owner.json",
        "status_file": "owned.process.json",
        "launch_state": "starting",
    }))

    with pytest.raises(ProcessCleanupError, match="complete cleanup evidence"):
        verify_process_cleanup(tmp_path)


@pytest.mark.parametrize(("inventory_name", "status_name"), [
    ("alias.process.inventory.json", "owned.process.json"),
    ("owned.process.inventory.json", "other.process.json"),
])
def test_startup_recovery_binds_inventory_to_its_filename(tmp_path, inventory_name, status_name):
    from acprof.host.runtimes.process_control import ProcessCleanupError, verify_process_cleanup

    owner_file = "owned.process.owner.json"
    status_path = tmp_path / status_name
    _, status = _cleanup_evidence(status_path)
    status_path.write_text(json.dumps(status))
    (tmp_path / owner_file).write_text(json.dumps({
        "schema_version": 2,
        "owner_file": owner_file,
        "status_file": status_name,
        "launch_state": "started",
        "owner_pid": status["owner"]["pid"],
    }))
    (tmp_path / inventory_name).write_text(json.dumps({
        "schema_version": 2,
        "owner_file": owner_file,
        "status_file": status_name,
        "launch_state": "starting",
    }))

    with pytest.raises(ProcessCleanupError, match="invalid process launch inventory"):
        verify_process_cleanup(tmp_path)


def test_startup_recovery_rejects_symlinked_or_mismatched_inventory(tmp_path):
    from acprof.host.runtimes.process_control import ProcessCleanupError, verify_process_cleanup

    target = tmp_path / "outside-inventory.json"
    target.write_text(json.dumps({
        "schema_version": 2,
        "owner_file": "owned.process.owner.json",
        "status_file": "owned.process.json",
        "launch_state": "starting",
    }))
    inventory = tmp_path / "owned.process.inventory.json"
    inventory.symlink_to(target)
    with pytest.raises(ProcessCleanupError, match="complete cleanup evidence"):
        verify_process_cleanup(tmp_path)

    inventory.unlink()
    inventory.write_text(target.read_text())
    (tmp_path / "owned.process.owner.json").write_text(json.dumps({
        "schema_version": 2,
        "owner_file": "owned.process.owner.json",
        "status_file": "other.process.json",
        "launch_state": "failed",
    }))
    with pytest.raises(ProcessCleanupError, match="does not match"):
        verify_process_cleanup(tmp_path)


def test_missing_startup_status_is_not_forgotten_on_resume(tmp_path):
    from acprof.host.runtimes.process_control import ProcessCleanupError, verify_process_cleanup
    (tmp_path / 'starting.process.owner.json').write_text(json.dumps({
        'launch_state': 'started', 'owner_pid': 123, 'status_file': 'starting.process.json'}))
    with pytest.raises(ProcessCleanupError, match='cleanup evidence'):
        verify_process_cleanup(tmp_path)
