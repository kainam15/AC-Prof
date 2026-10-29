"""Preparation timeouts must stop installer helpers before discarding staging."""
import os
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from acprof.dependency_locks import content_digest
from acprof.host.runtimes import environment
from acprof.process_identity import process_identity, same_process


@pytest.mark.parametrize("phase", ["venv", "install", "inspect"])
@pytest.mark.parametrize("failure", ["timeout", "cancel"])
def test_preparation_failure_stops_installer_descendants(tmp_path, monkeypatch, phase, failure):
    pid_path = tmp_path / "helper.pid"
    installer = tmp_path / "uv-fixture"
    installer.write_text(f'''#!{sys.executable}
import os, pathlib, shutil, signal, subprocess, sys, time
phase = "venv" if sys.argv[1] == "venv" else "install" if sys.argv[1] == "pip" else "inspect"
if phase == {phase!r}:
    child = subprocess.Popen([sys.executable, "-I", "-c",
        "import os,pathlib,time; pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    def stop(*args):
        child.wait(timeout=3)
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    time.sleep(60)
elif phase == "venv":
    python = pathlib.Path(sys.argv[-1]) / "bin/python"
    python.parent.mkdir(parents=True)
    shutil.copyfile(__file__, python)
    python.chmod(0o755)
elif phase == "inspect":
    print("{{}}")
''')
    installer.chmod(0o755)
    root = tmp_path / "environments"
    identity = {"artifacts": []}
    declaration = SimpleNamespace(requirements_lock="requirements.lock",
                                  platform=SimpleNamespace(python_version="3.12"))
    monkeypatch.setattr(environment, "environment_plan", lambda *args: (root, identity, declaration))
    monkeypatch.setattr(environment.shutil, "which", lambda name: str(installer))
    run_command = environment.run_command

    def bounded_command(command, **kwargs):
        return run_command(command, **{**kwargs, "timeout": 0.5})

    monkeypatch.setattr(environment, "run_command", bounded_command)
    if failure == "cancel":
        communicate = subprocess.Popen.communicate

        def interrupted_communicate(process, *args, **kwargs):
            try:
                return communicate(process, *args, **kwargs)
            except subprocess.TimeoutExpired:
                # Cancel only after the same bounded startup has launched a helper.
                raise KeyboardInterrupt from None

        monkeypatch.setattr(subprocess.Popen, "communicate", interrupted_communicate)
    try:
        expected_error = KeyboardInterrupt if failure == "cancel" else subprocess.TimeoutExpired
        with pytest.raises(expected_error):
            environment.prepare_environment(None, tmp_path)
        assert pid_path.exists(), "fixture did not reach the helper launch"
        try:
            identity_after_timeout = process_identity(int(pid_path.read_text()))
        except (FileNotFoundError, ProcessLookupError):
            identity_after_timeout = None
        assert identity_after_timeout is None or not same_process(identity_after_timeout), (
            "installer helper survived preparation failure", identity_after_timeout)
        assert not (root / content_digest(identity)).exists()
        assert not list(root.glob(".preparing-*"))
    finally:
        if pid_path.exists():
            # The failing baseline leaves a helper behind; stop only that fixture.
            try:
                descriptor = os.pidfd_open(int(pid_path.read_text()))
            except ProcessLookupError:
                pass
            else:
                try:
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                finally:
                    os.close(descriptor)
