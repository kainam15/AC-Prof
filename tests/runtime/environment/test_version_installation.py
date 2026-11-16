"""Real isolated uv tool updates plus deterministic recovery failures."""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.version import Version

from acprof.host.run_state import MeasurementLock, RunStateError
from acprof.updates.installation import Installation, detect_installation
from acprof.updates.releases import ReleaseVersion
from acprof.updates.service import launch_update, make_plan, read_state, update_active
from acprof.updates.worker import (
    Transaction,
    atomic_json,
    failure_kind,
    read_receipt,
    verify_installation,
)
from tests.support.update_wheels import write_wheel


def test_unsupported_platform_does_not_import_linux_lock(tmp_path, monkeypatch):
    import builtins

    original = builtins.__import__

    def platform_import(name, *args, **kwargs):
        if name == "fcntl":
            raise ModuleNotFoundError("No module named 'fcntl'")
        return original(name, *args, **kwargs)

    monkeypatch.setattr("acprof.updates.service.sys.platform", "win32")
    monkeypatch.setattr(builtins, "__import__", platform_import)
    assert not update_active(tmp_path)


@pytest.fixture
def installed_tool(tmp_path, monkeypatch):
    uv = shutil.which("uv")
    if not uv:
        pytest.skip("Real isolated tool test requires uv")
    index = tmp_path / "index"
    write_wheel(index, "1.0.0")
    tools, bins = tmp_path / "tools", tmp_path / "bin"
    monkeypatch.setenv("UV_TOOL_DIR", str(tools))
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(bins))
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "cache"))
    subprocess.run([uv, "--no-config", "tool", "install", "acprof==1.0.0", "--default-index", index.as_uri(),
                    "--python", str(Path(sys.executable).resolve()), "--no-python-downloads"],
                   check=True, capture_output=True, text=True, timeout=20)
    installation = Installation("uv tool", tools / "acprof", tools / "acprof/bin/python", bins / "acprof", Path(uv))
    state = tmp_path / "state"
    state.mkdir()
    return installation, state, index


def plan_for(installed_tool, current, target, channel):
    installation, state, index = installed_tool
    release = ReleaseVersion(Version(current), Version(target) if channel == "stable" else None,
                             Version(target) if channel == "preview" else None, channel)
    return replace(make_plan(installation, release, state), index_url=index.as_uri(), timeout=15)


def run_plan(plan):
    process = launch_update(plan)
    try:
        process.wait(timeout=25)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return read_state(Path(plan.state_dir))


def test_real_uv_switches_upgrade_downgrade_and_unpinned_future(installed_tool, tmp_path):
    installation, state, index = installed_tool
    protected = [tmp_path / "settings.json", tmp_path / "results.csv", tmp_path / "model-cache"]
    for path in protected:
        path.write_bytes(b"original user bytes")
    current = "1.0.0"
    for target, channel in [("1.1.0a1", "preview"), ("1.1.0b1", "preview"),
                            ("1.1.0rc1", "preview"), ("1.0.0", "stable"),
                            ("1.1.0rc1", "preview"), ("1.1.0", "preview"), ("1.1.0", "stable")]:
        write_wheel(index, target)
        plan = plan_for(installed_tool, current, target, channel)
        outcome = run_plan(plan)
        assert outcome["status"] == "success", outcome
        verify_installation(plan, target, updated=True)
        assert read_state(state, "channel.json")["channel"] == channel
        assert read_receipt(installation.prefix)["requirements"] == [{"name": "acprof"}]
        current = target
    # A later ordinary uv upgrade must not be trapped by the confirmed version.
    write_wheel(index, "1.2.0")
    subprocess.run([str(installation.manager), "--no-config", "tool", "upgrade", "acprof"],
                   check=True, capture_output=True, timeout=20)
    verify_installation(plan, "1.2.0")
    assert all(path.read_bytes() == b"original user bytes" for path in protected)
    assert not update_active(state)
    with MeasurementLock():
        pass


@pytest.mark.parametrize("failure", ["missing", "wrong-entry", "newer-release"])
def test_real_uv_failure_restores_version_receipt_and_channel(installed_tool, failure):
    installation, state, index = installed_tool
    original_receipt = (installation.prefix / "uv-receipt.toml").read_bytes()
    atomic_json(state / "channel.json", {"channel": "stable", "prefix": str(installation.prefix)})
    original_channel = (state / "channel.json").read_bytes()
    if failure == "wrong-entry":
        write_wheel(index, "1.1.0a1", command_version="99.0.0")
    elif failure == "newer-release":
        write_wheel(index, "1.1.0a1")
        write_wheel(index, "1.1.0b1")
    plan = plan_for(installed_tool, "1.0.0", "1.1.0a1", "preview")
    if failure == "missing":
        plan = replace(plan, index_url=(index / "missing").as_uri())
    outcome = run_plan(plan)
    assert outcome["status"] == "failed", outcome
    assert outcome["rollback"] == "restored", outcome
    if failure == "wrong-entry":
        assert "command returned 'AC-Prof 99.0.0'" in outcome["error"]
    elif failure == "newer-release":
        assert "got" in outcome["error"]
        assert "1.1.0b1" in outcome["error"]
    else:
        assert "not found in the package registry" in outcome["error"]
    verify_installation(plan, "1.0.0")
    assert (installation.prefix / "uv-receipt.toml").read_bytes() == original_receipt
    assert (state / "channel.json").read_bytes() == original_channel


@pytest.mark.parametrize(("error", "expected"), [
    (PermissionError("denied"), "permission"), (OSError("No space left on device"), "disk"),
    (TimeoutError("deadline"), "timeout"), (RuntimeError("DNS connection failed"), "network"),
    (ValueError("Verification failed"), "environment"), (RuntimeError("installer exited"), "install"),
])
def test_failure_classification(error, expected):
    assert failure_kind(error) == expected


def test_snapshot_failure_never_runs_installer(installed_tool, monkeypatch):
    plan = plan_for(installed_tool, "1.0.0", "1.1.0a1", "preview")
    Path(plan.job_dir).mkdir(parents=True)
    monkeypatch.setattr(shutil, "copytree", lambda *a, **k: (_ for _ in ()).throw(PermissionError("denied")))
    monkeypatch.setattr(Transaction, "install", lambda *a: pytest.fail("must not mutate without a backup"))
    assert not Transaction(plan).run()
    state = read_state(Path(plan.state_dir))
    assert state["error_kind"] == "permission"
    assert state["rollback"] == "not_needed"
    verify_installation(plan, "1.0.0")


def test_child_retains_both_measurement_guards_after_parent_closes(tmp_path):
    with MeasurementLock() as lock:
        process = subprocess.Popen([sys.executable, "-I", "-c", "import sys;sys.stdin.read()"],
                                   stdin=subprocess.PIPE, pass_fds=lock.inherited_descriptors)
    try:
        with pytest.raises(RunStateError), MeasurementLock():
            pytest.fail("child must still exclude measurements")
    finally:
        process.communicate(timeout=5)
    with MeasurementLock():
        pass


def test_actual_current_uv_paths_and_mismatched_override(installed_tool, monkeypatch, tmp_path):
    installation, _, _ = installed_tool
    monkeypatch.setattr(sys, "prefix", str(installation.prefix))
    monkeypatch.setattr(sys, "executable", str(installation.python))
    monkeypatch.setattr("acprof.__file__", str(installation.prefix / "lib/python3.12/site-packages/acprof/__init__.py"))
    monkeypatch.setattr("importlib.metadata.distribution", lambda _: SimpleNamespace(read_text=lambda name: None))
    assert detect_installation().supported
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path / "other-tools"))
    wrong = detect_installation()
    assert not wrong.supported
    assert "does not match" in wrong.reason


def test_entrypoint_of_another_tool_is_not_overwritten(installed_tool, tmp_path):
    installation, _, _ = installed_tool
    other = tmp_path / "other-acprof"
    other.write_text("other installation")
    installation.entrypoint.unlink()
    installation.entrypoint.symlink_to(other)
    with pytest.raises(ValueError, match="different installation"):
        plan_for(installed_tool, "1.0.0", "1.1.0a1", "preview")
    assert other.read_text() == "other installation"


@pytest.mark.parametrize("stop", ["timeout", "terminate"])
def test_install_timeout_reaps_child_blocks_parallel_and_restores(installed_tool, stop):
    installation, state, _ = installed_tool
    manager = state / "slow-uv"
    ready = state / "ready"
    manager.write_text(f"#!{Path(sys.executable).resolve()}\nimport time\nfrom pathlib import Path\nPath({str(ready)!r}).touch()\ntime.sleep(60)\n")
    manager.chmod(0o700)
    plan = replace(plan_for(installed_tool, "1.0.0", "1.1.0a1", "preview"), manager=str(manager), timeout=2)
    process = launch_update(plan)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        assert update_active(state)
        with pytest.raises(RunStateError):
            launch_update(replace(plan, job_dir=str(state / "other-job")))
        with pytest.raises(RunStateError), MeasurementLock():
            pytest.fail("installation must exclude collection")
        if stop == "terminate":
            process.terminate()
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    outcome = read_state(state)
    assert outcome["error_kind"] == ("timeout" if stop == "timeout" else "install")
    assert outcome["rollback"] == "restored"
    assert not update_active(state)
    verify_installation(plan, "1.0.0")


def test_disk_exhaustion_and_recovery_failure_are_explicit(installed_tool, monkeypatch):
    plan = plan_for(installed_tool, "1.0.0", "1.1.0a1", "preview")
    Path(plan.job_dir).mkdir(parents=True)
    with monkeypatch.context() as context:
        context.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
        assert not Transaction(plan).run()
    outcome = read_state(Path(plan.state_dir))
    assert outcome["error_kind"] == "disk"
    assert outcome["rollback"] == "not_needed"
    monkeypatch.setattr(Transaction, "install", lambda _: (_ for _ in ()).throw(RuntimeError("install failed")))
    monkeypatch.setattr(Transaction, "restore", lambda _: (_ for _ in ()).throw(PermissionError("backup unavailable")))
    assert not Transaction(plan).run()
    outcome = read_state(Path(plan.state_dir))
    assert outcome["rollback"] == "failed"
    assert "backup unavailable" in outcome["error"]
    assert (Path(plan.job_dir) / "backup/uv-receipt.toml").is_file()
    assert not (Path(plan.state_dir) / "channel.json").exists()
