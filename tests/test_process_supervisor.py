"""Preserve kernel exit status when cleanup reaps a workload before Popen does."""
import json
import os
import signal
import subprocess
import sys

import pytest

from acprof.host.runtimes import supervisor


@pytest.mark.parametrize("exit_code", [0, 23, -signal.SIGTERM])
def test_cleanup_reaper_preserves_workload_exit_status(tmp_path, monkeypatch, exit_code):
    source = (f"import os; os._exit({exit_code})" if exit_code >= 0 else
              f"import os; os.kill(os.getpid(), {-exit_code})")
    child = subprocess.Popen([sys.executable, "-I", "-c", source],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    descendant = subprocess.Popen([sys.executable, "-I", "-c", "import os; os._exit(91)"],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    real_poll, real_waitpid = child.poll, os.waitpid
    reaped = set()

    def delayed_poll():
        # Fix the scheduler interleaving: cleanup wins the race to waitpid.
        return real_poll() if child.pid in reaped else None

    def reap_owned_child(pid, options):
        assert pid == -1
        # Keep the test's reaper away from any other pytest-owned children.
        for process in (child, descendant):
            if process.pid not in reaped:
                result = real_waitpid(process.pid, options)
                if result[0]:
                    reaped.add(result[0])
                return result
        raise ChildProcessError

    status_path = tmp_path / "status.json"
    try:
        # Observe exit without consuming the real kernel status.
        for process in (child, descendant):
            os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
        with monkeypatch.context() as patch:
            patch.setattr(sys, "argv", ["supervisor", str(read_fd), str(status_path),
                                      str(tmp_path / "child.log"), "", "", "fixture"])
            patch.delenv("ACPROF_SERVER_FD", raising=False)
            patch.setattr(supervisor, "verify_supervision", lambda: None)
            patch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: child)
            patch.setattr(child, "poll", delayed_poll)
            patch.setattr(supervisor.os, "waitpid", reap_owned_child)
            patch.setattr(supervisor, "descendants",
                          lambda pid: [p.pid for p in (child, descendant) if p.pid not in reaped])
            supervisor.main()
    finally:
        os.close(read_fd)
        for process in (child, descendant):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)

    status = json.loads(status_path.read_text())
    assert status.get("cleanup_complete"), status
    assert "cleanup_error" not in status, status
    assert status["returncode"] == exit_code
