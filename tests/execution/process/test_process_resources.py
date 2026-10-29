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
