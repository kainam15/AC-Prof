"""Verified resource enforcement is separate from inherited allocation evidence."""
from __future__ import annotations

import errno
import os
import time
import uuid
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from acprof.host.runtimes import cgroups
from acprof.host.runtimes.cgroups import ProcessResourceError, ResourceDiagnostic, current_cgroup
from acprof.system.cleanup import cleanup_on_exit

CGROUP_RMDIR_TIMEOUT_S = 1.0
CGROUP_RMDIR_POLL_INTERVAL_S = 0.02


def _read_limit(path: Path) -> dict[str, str]:
    try:
        return {'status': 'available', 'value': path.read_text().strip()}
    except (FileNotFoundError, PermissionError) as exc:
        return {'status': 'permission_denied' if isinstance(exc, PermissionError) else 'unavailable', 'error': str(exc)}


def _quota_bound(field: str, value: str | None) -> int | Fraction | None:
    if value is None or value == 'max':
        return None
    if field == 'cpu.max':
        quota, period = value.split()
        if quota == 'max':
            return None
        if int(period) <= 0:
            raise ValueError(f'Invalid cpu.max period: {period}')
        return Fraction(int(quota), int(period))
    return int(value)


def _visible_limits(root: Path, *, outside_ancestry_of: Path | None = None) -> list[dict[str, Any]]:
    return [
        {'path': str(directory), 'cpu.max': _read_limit(directory / 'cpu.max'),
         'memory.max': _read_limit(directory / 'memory.max')}
        for directory in cgroups.ancestors(root)
        if outside_ancestry_of is None or not outside_ancestry_of.is_relative_to(directory)
    ]


def inherited_resources(*, pid: int | None = None) -> dict:
    cgroup_error = ""
    try:
        root = current_cgroup() if pid is None else cgroups.process_cgroup(Path(f"/proc/{pid}/cgroup"))
    except OSError as exc:
        root = None
        cgroup_error = f"{type(exc).__name__}: {exc}"
    result: dict[str, Any] = {"affinity": sorted(os.sched_getaffinity(pid or 0)), "cgroup": str(root) if root else None,
              "cpu_max": None, "memory_max": None, "allocation_source": "visible_process_namespace",
              "exclusive": "unknown", "hidden_ancestor_limits": "unknown"}
    result['cgroup_error'] = cgroup_error
    ancestors = _visible_limits(root) if root is not None else []
    if ancestors:
        result['cpu_max'] = ancestors[0]['cpu.max'].get('value')
        result['memory_max'] = ancestors[0]['memory.max'].get('value')
    cpu_limits, memory_limits = [float(len(result["affinity"]))], []
    for ancestor in ancestors:
        cpu = _quota_bound('cpu.max', ancestor['cpu.max'].get('value'))
        memory = _quota_bound('memory.max', ancestor['memory.max'].get('value'))
        if cpu is not None:
            cpu_limits.append(float(cpu))
        if memory is not None:
            memory_limits.append(int(memory))
    result.update(visible_ancestor_limits=ancestors, effective_cpu_upper_bound=min(cpu_limits),
                  effective_memory_upper_bound=min(memory_limits) if memory_limits else None)
    return result


class ProcessLimits:
    def __init__(self, cpu, memory, affinity=""):
        self.cpu, self.memory, self.affinity = cpu, memory, affinity
        self.path = ""
        self.verified = False
        self.inherited = inherited_resources()
        self.oom_before = 0
        self.diagnostic = ResourceDiagnostic(code="inherited")
        if affinity:
            from acprof.cpu_affinity import parse_cpu_set
            if not parse_cpu_set(affinity) <= set(self.inherited['affinity']):
                raise ValueError('requested affinity exceeds the visible CPU allocation')
        if cpu is None and memory is None:
            return
        self.diagnostic = ResourceDiagnostic()
        try:
            self.diagnostic = cgroups.inspect(cpu, memory, current_cgroup())
            if self.diagnostic.code != 'ready':
                raise ProcessResourceError(self.diagnostic)
            self._check_parent_limits()
            self.diagnostic = cgroups.enable_controllers(self.diagnostic)
            owned = Path(self.diagnostic.parent) / ("acprof-" + uuid.uuid4().hex)
            owned.mkdir()
            self.path = str(owned)
            if cpu is not None:
                (owned / "cpu.max").write_text(f"{cpu * 100000} 100000")
            if memory is not None:
                (owned / "memory.max").write_text(str(memory * 1024**3))
            self._verify_quotas()
            self.oom_before = self._oom_count()
        except (OSError, ValueError, RuntimeError) as exc:
            error = cgroups.failure(self.diagnostic, exc)
            # Keep the original failure if cleanup also fails; never erase debt.
            with cleanup_on_exit(self.close, description="Process cgroup allocation"):
                if error is exc:
                    raise
                raise error from exc

    def _check_parent_limits(self):
        cpu, memory = self.cpu, self.memory
        failure = ''
        if cpu is not None and self.affinity:
            from acprof.cpu_affinity import parse_cpu_set
            if cpu > len(parse_cpu_set(self.affinity)):
                failure = 'requested CPU quota exceeds requested affinity'
        if cpu is not None and cpu > len(self.inherited['affinity']):
            failure = 'requested CPU quota exceeds inherited allocation'
        requested = {'cpu.max': cpu, 'memory.max': memory * 1024**3 if memory is not None else None}
        target = Path(self.diagnostic.parent)
        target_limits = _visible_limits(target)
        # The caller can reside in a sibling. Re-read limits on both paths
        # when the workload starts; keep the original allocation evidence.
        caller_limits = []
        if self.diagnostic.current and not target.is_relative_to(self.diagnostic.current):
            caller_limits = _visible_limits(Path(self.diagnostic.current), outside_ancestry_of=target)
        evidence = []
        for ancestor in target_limits:
            limits = {'path': ancestor['path']}
            for field in requested:
                limit = ancestor[field]
                limits[field] = limit.get('value', limit['status'])
            evidence.append(limits)
        self.inherited['delegated_ancestor_limits'] = evidence
        for field, amount in requested.items():
            if amount is None:
                continue
            for ancestor in (*caller_limits, *target_limits):
                limit = ancestor[field]
                if limit['status'] == 'permission_denied':
                    raise ProcessResourceError(replace(self.diagnostic, code='permission_denied', detail=limit['error']))
                bound = _quota_bound(field, limit.get('value'))
                if bound is not None and amount > bound:
                    failure = f"{ancestor['path']}: {field}={limit['value']}"
        if failure:
            raise ProcessResourceError(replace(self.diagnostic, code='parent_limit', detail=failure))
        for field, amount in requested.items():
            if amount is None:
                self._check_inherited_axis(field, caller_limits, target_limits)

    def _check_inherited_axis(
        self, field: str, caller_limits: list[dict[str, Any]], target_limits: list[dict[str, Any]],
    ) -> None:
        # A workload outside the caller's subtree must retain any visible
        # finite bound on its inherited axis, including at final verification.
        if not self.diagnostic.current:
            raise ProcessResourceError(replace(self.diagnostic, code='verification_failed',
                                               detail=f'cannot locate caller cgroup to verify inherited {field}'))
        if not caller_limits:
            return
        target = Path(self.diagnostic.parent)
        target_values = [ancestor[field] for ancestor in target_limits]
        bounds = (_quota_bound(field, limit.get('value')) for limit in target_values)
        target_bound = min((bound for bound in bounds if bound is not None), default=None)
        for ancestor in caller_limits:
            limit = ancestor[field]
            if limit['status'] == 'permission_denied':
                raise ProcessResourceError(replace(self.diagnostic, code='permission_denied', detail=limit['error']))
            caller_bound = _quota_bound(field, limit.get('value'))
            if caller_bound is None or (target_bound is not None and target_bound <= caller_bound):
                continue
            for target_limit in target_values:
                if target_limit['status'] == 'permission_denied':
                    raise ProcessResourceError(replace(self.diagnostic, code='permission_denied', detail=target_limit['error']))
            raise ProcessResourceError(replace(self.diagnostic, code='parent_limit', detail=(
                f"{target} cannot preserve inherited {field}={limit['value']} from {ancestor['path']}"
            )))

    def _verify_quotas(self):
        if not self.path:
            return
        root = Path(self.path)
        if self.cpu is not None and (root / "cpu.max").read_text().split() != [str(self.cpu * 100000), "100000"]:
            raise RuntimeError("Process CPU quota verification failed")
        if self.memory is not None and int((root / "memory.max").read_text()) != self.memory * 1024**3:
            raise RuntimeError("Process memory limit verification failed")

    def verify_probe(self):
        """A bounded, model-free child proves migration permissions before preparation.

        The probe is not workload evidence and cannot set self.verified.
        """
        if not self.path:
            return
        import subprocess
        import sys

        from acprof.system.command import run_command
        self._verify_quotas()
        script = """import os, sys
from pathlib import Path
try:
    path = Path(sys.argv[1]) / 'cgroup.procs'
    path.write_text(str(os.getpid()))
    sys.exit(0 if str(os.getpid()) in path.read_text().split() else 1)
except OSError as exc:
    print(str(exc), file=sys.stderr)
    sys.exit(13 if exc.errno in (1, 13, 30) else 1)
"""
        try:
            result = run_command([sys.executable, '-I', '-c', script, self.path],
                                 capture_output=True, text=True, timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise cgroups.failure(self.diagnostic, exc, code='verification_failed') from exc
        if result.returncode:
            raise cgroups.failure(self.diagnostic, RuntimeError(result.stderr),
                                  code='permission_denied' if result.returncode == 13 else 'verification_failed')
        self._verify_quotas()

    def verify(self, pid):
        self.verified = False
        if self.path:
            try:
                root = Path(self.path)
                if str(pid) not in (root / "cgroup.procs").read_text().split():
                    raise RuntimeError("Process cgroup membership verification failed")
                self._verify_quotas()
                if self.cpu is not None and self.cpu > len(os.sched_getaffinity(pid)):
                    raise RuntimeError("Process CPU quota exceeds workload affinity")
                # Ancestors may have changed since the earlier preflight.
                self._check_parent_limits()
            except (OSError, ValueError, RuntimeError) as exc:
                if isinstance(exc, ProcessResourceError):
                    raise
                raise cgroups.failure(self.diagnostic, exc, code='verification_failed') from exc
        if self.affinity:
            from acprof.cpu_affinity import parse_cpu_set
            if os.sched_getaffinity(pid) != parse_cpu_set(self.affinity):
                raise RuntimeError("Process affinity verification failed")
        self.verified = True

    def _oom_count(self):
        if not self.path:
            return 0
        try:
            counts = dict(line.split() for line in (Path(self.path) / "memory.events").read_text().splitlines())
            return int(counts['oom_kill'])
        except (FileNotFoundError, PermissionError, KeyError):
            return None

    def oom_killed(self):
        after = self._oom_count()
        return bool(self.path) and after is not None and self.oom_before is not None and after > self.oom_before

    def evidence(self):
        return {"requested": {"cpu_quota": self.cpu, "memory_gib": self.memory, "affinity": self.affinity or None},
                "inherited": self.inherited, "method": "cgroup_v2" if self.path else "affinity" if self.affinity else "inherited",
                "verified": self.verified, "cgroup": self.path or None,
                "scope": "workload_tree" if self.path else "observed_process_tree"}

    def close(self):
        if not self.path:
            return
        deadline = time.monotonic() + CGROUP_RMDIR_TIMEOUT_S
        while True:
            try:
                Path(self.path).rmdir()
            except OSError as exc:
                remaining = deadline - time.monotonic()
                if exc.errno != errno.EBUSY or remaining <= 0:
                    raise
                time.sleep(min(CGROUP_RMDIR_POLL_INTERVAL_S, remaining))
            else:
                self.path = ""
                return


def preflight_resources(cpus, memories, affinity):
    """Exercise requested limit files before any dependency/model preparation."""
    diagnostic = ResourceDiagnostic(code="inherited")
    repaired = False
    try:
        for cpu in cpus:
            for memory in memories:
                limits = ProcessLimits(cpu, memory, affinity)
                diagnostic = limits.diagnostic
                with cleanup_on_exit(limits.close, description="Process resource preflight"):
                    limits.verify_probe()
                    repaired = repaired or diagnostic.repaired
    except (OSError, ValueError, RuntimeError) as exc:
        error = cgroups.failure(diagnostic, exc)
        if error is exc:
            raise
        raise error from exc
    return replace(diagnostic, repaired=repaired)
