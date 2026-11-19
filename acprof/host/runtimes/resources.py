"""Verified resource enforcement is separate from inherited allocation evidence."""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from acprof.host.runtimes import cgroups
from acprof.host.runtimes.cgroups import ProcessResourceError, ResourceDiagnostic, current_cgroup
from acprof.system.cleanup import cleanup_on_exit


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
    ancestors = []
    if root is not None:
        for directory in cgroups.ancestors(root):
            limits: dict[str, Any] = {"path": str(directory)}
            for name in ("cpu.max", "memory.max"):
                try:
                    value = (directory / name).read_text().strip()
                    limits[name] = {"status": "available", "value": value}
                    if directory == root:
                        result[name.replace(".", "_")] = value
                except (FileNotFoundError, PermissionError) as exc:
                    limits[name] = {"status": "permission_denied" if isinstance(exc, PermissionError) else "unavailable", "error": str(exc)}
            ancestors.append(limits)
    cpu_limits, memory_limits = [float(len(result["affinity"]))], []
    for ancestor in ancestors:
        cpu = ancestor['cpu.max'].get('value', '').split()
        memory = ancestor['memory.max'].get('value')
        if cpu and cpu[0] != 'max':
            cpu_limits.append(int(cpu[0]) / int(cpu[1]))
        if memory and memory != 'max':
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
        from dataclasses import replace
        cpu, memory = self.cpu, self.memory
        evidence = []
        failure = ''
        if cpu is not None and self.affinity:
            from acprof.cpu_affinity import parse_cpu_set
            if cpu > len(parse_cpu_set(self.affinity)):
                failure = 'requested CPU quota exceeds requested affinity'
        if cpu is not None and cpu > self.inherited['effective_cpu_upper_bound']:
            failure = 'requested CPU quota exceeds inherited allocation'
        if memory is not None and self.inherited['effective_memory_upper_bound'] is not None and memory * 1024**3 > self.inherited['effective_memory_upper_bound']:
            failure = 'requested memory quota exceeds inherited allocation'
        for ancestor in cgroups.ancestors(Path(self.diagnostic.parent)):
            limits = {'path': str(ancestor)}
            for field in ('cpu.max', 'memory.max'):
                try:
                    value = (ancestor / field).read_text().strip()
                except FileNotFoundError:
                    value = 'unavailable'
                limits[field] = value
                if value in {'max', 'unavailable'}:
                    continue
                if field == 'cpu.max' and cpu is not None:
                    quota, period = value.split()
                    if quota != 'max' and int(quota) / int(period) < cpu:
                        failure = f'{ancestor}: cpu.max={value}'
                elif field == 'memory.max' and memory is not None and int(value) < memory * 1024**3:
                    failure = f'{ancestor}: memory.max={value}'
            evidence.append(limits)
        self.inherited['delegated_ancestor_limits'] = evidence
        if failure:
            raise ProcessResourceError(replace(self.diagnostic, code='parent_limit', detail=failure))

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
        if self.path:
            Path(self.path).rmdir()
            self.path = ""


def preflight_resources(cpus, memories, affinity):
    """Exercise requested limit files before any dependency/model preparation."""
    from dataclasses import replace
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
