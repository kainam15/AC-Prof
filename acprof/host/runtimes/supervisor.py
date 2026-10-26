"""Standard-library-only supervisor, executed by the isolated runtime Python.

Its pipe closes when the controller exits (including SIGKILL). Subreaping keeps
double-forked descendants owned until cleanup. No sampling happens here.
"""
from __future__ import annotations

import ctypes
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path


def identity(pid):
    raw = Path(f"/proc/{pid}/stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    return {"pid": pid, "start_ticks": int(fields[19]), "pgid": int(fields[2]), "state": fields[0]}


def descendants(pid):
    pending = [pid]
    found: set[int] = set()
    while pending:
        parent = pending.pop()
        # A child belongs to the thread that forked it, not necessarily the
        # thread-group leader. Include workers created by background threads.
        for path in Path(f"/proc/{parent}/task").glob("*/children"):
            try:
                children = {int(child) for child in path.read_text().split()}
            except FileNotFoundError:
                continue
            pending.extend(children - found)
            found.update(children)
    return sorted(found)


def verify_supervision():
    """Exercise ownership syscalls in this helper, never in the UI/controller."""
    descriptor = os.pidfd_open(os.getpid())
    try:
        signal.pidfd_send_signal(descriptor, 0)
    finally:
        os.close(descriptor)
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot establish descendant ownership")


def main():
    control, status_file, log_file, cpuset, cgroup, *command = sys.argv[1:]
    status_path = Path(status_file)
    status = {"owner": identity(os.getpid()), "returncode": None}

    def publish():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status))
        os.replace(temporary, status_path)

    workload = None
    try:
        # Verify safe signalling before taking ownership of any model process.
        # Old kernels or seccomp restrictions must fail before launch.
        verify_supervision()
        with open(log_file, "ab", buffering=0) as log:
            # Resource controls are installed in a tiny launch process before exec.
            launcher = str(Path(__file__).with_name("launch.py"))
            descriptors = (int(os.environ["ACPROF_SERVER_FD"]),) if os.environ.get("ACPROF_SERVER_FD") else ()
            workload = subprocess.Popen([sys.executable, launcher, cpuset, cgroup, *command],
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                        start_new_session=True, close_fds=True, pass_fds=descriptors)
            for descriptor in descriptors:
                os.close(descriptor)
            status["workload"] = identity(workload.pid)
            publish()
            while workload.poll() is None:
                readable, _, _ = select.select([int(control)], [], [], 0.2)
                if readable and not os.read(int(control), 1):
                    break
            status["returncode"] = workload.poll()
    except BaseException as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            for sig, grace in ((signal.SIGTERM, 2), (signal.SIGKILL, 3)):
                deadline = time.monotonic() + grace
                while time.monotonic() < deadline:
                    for pid in descendants(os.getpid()):
                        try:
                            # Rescan until empty: a thread may fork while its
                            # parent's termination signal is being delivered.
                            descriptor = os.pidfd_open(pid)
                            try:
                                if pid in descendants(os.getpid()):
                                    signal.pidfd_send_signal(descriptor, sig)
                            finally:
                                os.close(descriptor)
                        except ProcessLookupError:
                            pass
                    if workload is not None:
                        workload.poll()
                    try:
                        while True:
                            pid, wait_status = os.waitpid(-1, os.WNOHANG)
                            if not pid:
                                break
                            if workload is not None and pid == workload.pid:
                                # Cleanup may win the race with Popen.poll().
                                # Keep the consumed status before ECHILD could
                                # make a later poll report a successful exit.
                                workload.returncode = os.waitstatus_to_exitcode(wait_status)
                    except ChildProcessError:
                        break
                    time.sleep(0.02)
                if not descendants(os.getpid()):
                    break
            if descendants(os.getpid()):
                raise RuntimeError("descendant processes survived cleanup")
            status['cleanup_complete'] = True
            if workload is not None:
                status["returncode"] = workload.poll()
        except BaseException as exc:
            status["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        publish()


if __name__ == "__main__":
    if sys.argv[1:] == ["--probe"]:
        verify_supervision()
        print("pidfd and child-subreaper available")
    else:
        main()
