"""Explicit affinity must be verified after cgroup membership is applied."""
import pytest

from acprof.host.runtimes import resources


def test_tighter_effective_affinity_does_not_claim_the_requested_cpu_set(monkeypatch):
    monkeypatch.setattr(resources, 'inherited_resources', lambda: {'affinity': [0, 1]})
    limits = resources.ProcessLimits(None, None, '0,1')
    monkeypatch.setattr(resources.os, 'sched_getaffinity', lambda pid: {0})
    with pytest.raises(RuntimeError, match='affinity verification'):
        limits.verify(123)
    assert not limits.evidence()['verified']


def test_process_limits_close_retries_transient_ebusy_and_fails_closed(monkeypatch, tmp_path):
    import errno
    from pathlib import Path

    monkeypatch.setattr(resources, 'inherited_resources', lambda: {'affinity': [0, 1]})
    monkeypatch.setattr(resources, 'CGROUP_RMDIR_TIMEOUT_S', 0.05)
    monkeypatch.setattr(resources, 'CGROUP_RMDIR_POLL_INTERVAL_S', 0.0)

    cgroup_dir = tmp_path / "cgroup"
    cgroup_dir.mkdir()
    limits = resources.ProcessLimits(None, None)
    limits.path = str(cgroup_dir)

    attempts = 0
    original_rmdir = Path.rmdir

    def transient_ebusy(self):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError(errno.EBUSY, "Device or resource busy")
        original_rmdir(self)

    monkeypatch.setattr(Path, "rmdir", transient_ebusy)
    limits.close()
    assert attempts == 3
    assert limits.path == ""
    assert not cgroup_dir.exists()

    persistent_dir = tmp_path / "cgroup_stuck"
    persistent_dir.mkdir()
    limits.path = str(persistent_dir)
    monkeypatch.setattr(Path, "rmdir", lambda _self: (_ for _ in ()).throw(OSError(errno.EBUSY, "Device or resource busy")))
    with pytest.raises(OSError, match="Device or resource busy") as exc_info:
        limits.close()
    assert exc_info.value.errno == errno.EBUSY
    assert limits.path == str(persistent_dir)

    eperm_calls = 0

    def fail_eperm(_self):
        nonlocal eperm_calls
        eperm_calls += 1
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(Path, "rmdir", fail_eperm)
    with pytest.raises(OSError, match="Operation not permitted") as eperm_info:
        limits.close()
    assert eperm_info.value.errno == errno.EPERM
    assert eperm_calls == 1
    assert limits.path == str(persistent_dir)
