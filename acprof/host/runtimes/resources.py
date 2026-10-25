"""Verified resource enforcement is separate from inherited allocation evidence."""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any


def current_cgroup() -> Path | None:
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            relative = line[3:].lstrip("/")
            candidate = Path("/sys/fs/cgroup") / relative
            if candidate.is_dir() and ".." not in candidate.parts:
                return candidate
    return None


def inherited_resources() -> dict:
    cgroup_error = ""
    try:
        root = current_cgroup()
    except OSError as exc:
        root = None
        cgroup_error = f"{type(exc).__name__}: {exc}"
    result: dict[str, Any] = {"affinity": sorted(os.sched_getaffinity(0)), "cgroup": str(root) if root else None,
              "cpu_max": None, "memory_max": None, "allocation_source": "visible_process_namespace",
              "exclusive": "unknown", "hidden_ancestor_limits": "unknown"}
    result['cgroup_error'] = cgroup_error
    ancestors = []
    if root is not None:
        for directory in (root, *root.parents):
            if not directory.is_relative_to('/sys/fs/cgroup'):
                break
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
        if affinity:
            from acprof.cpu_affinity import parse_cpu_set
            if not parse_cpu_set(affinity) <= set(self.inherited['affinity']):
                raise ValueError('requested affinity exceeds the visible CPU allocation')
        if cpu is None and memory is None:
            return
        if cpu is not None and cpu > self.inherited['effective_cpu_upper_bound']:
            raise ValueError('requested CPU quota exceeds an inherited CPU limit')
        if memory is not None and self.inherited['effective_memory_upper_bound'] is not None and memory * 1024**3 > self.inherited['effective_memory_upper_bound']:
            raise ValueError('requested memory quota exceeds an inherited memory limit')
        parent = Path(os.environ["ACPROF_CGROUP_ROOT"]) if os.environ.get("ACPROF_CGROUP_ROOT") else current_cgroup()
        if parent is None:
            raise ValueError("Process limits require a delegated cgroup v2; use --cpus auto --mems auto")
        parent = parent.resolve()
        mounts = []
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            before, after = line.split(' - ', 1)
            if after.split()[0] == 'cgroup2':
                mounts.append(Path(before.split()[4]))
        if not any(parent.is_relative_to(mount) for mount in mounts):
            raise ValueError('ACPROF_CGROUP_ROOT must be on a cgroup v2 filesystem')
        # A delegated root may have tighter ancestors than the controller's own
        # group. Verify every visible level before claiming the requested cap.
        delegated = []
        for ancestor in (parent, *parent.parents):
            if not any(ancestor.is_relative_to(mount) for mount in mounts):
                break
            evidence = {'path': str(ancestor)}
            for field in ('cpu.max', 'memory.max'):
                try:
                    value = (ancestor / field).read_text().strip()
                except FileNotFoundError:
                    value = 'unavailable'
                evidence[field] = value
                if value in {'max', 'unavailable'}:
                    continue
                if field == 'cpu.max' and cpu is not None:
                    parts = value.split()
                    if parts[0] != 'max' and int(parts[0]) / int(parts[1]) < cpu:
                        raise ValueError('requested CPU quota exceeds a delegated ancestor limit')
                elif field == 'memory.max' and memory is not None and int(value) < memory * 1024**3:
                    raise ValueError('requested memory quota exceeds a delegated ancestor limit')
            delegated.append(evidence)
        self.inherited['delegated_ancestor_limits'] = delegated
        owned = parent / ("acprof-" + uuid.uuid4().hex)
        try:
            owned.mkdir()
            self.path = str(owned)
            if cpu is not None:
                if cpu > len(os.sched_getaffinity(0)):
                    raise ValueError('requested CPU quota exceeds visible affinity')
                parent_quota = (parent / "cpu.max").read_text().split()
                if parent_quota[0] != "max" and int(parent_quota[0]) / int(parent_quota[1]) < cpu:
                    raise ValueError("requested CPU quota exceeds the inherited quota")
                (owned / "cpu.max").write_text(f"{cpu * 100000} 100000")
            if memory is not None:
                limit = (parent / "memory.max").read_text().strip()
                if limit != "max" and int(limit) < memory * 1024**3:
                    raise ValueError("requested memory limit exceeds the inherited limit")
                (owned / "memory.max").write_text(str(memory * 1024**3))
            self.oom_before = self._oom_count()
        except (OSError, ValueError) as exc:
            self.close()
            raise ValueError(f"cannot enforce Process quotas in delegated cgroup: {exc}; use --cpus auto --mems auto") from exc

    def verify(self, pid):
        if self.path:
            root = Path(self.path)
            if str(pid) not in (root / "cgroup.procs").read_text().split():
                raise RuntimeError("Process cgroup membership verification failed")
            if self.cpu is not None and (root / "cpu.max").read_text().split() != [str(self.cpu * 100000), "100000"]:
                raise RuntimeError("Process CPU quota verification failed")
            if self.memory is not None and int((root / "memory.max").read_text()) != self.memory * 1024**3:
                raise RuntimeError("Process memory limit verification failed")
        if self.affinity:
            from acprof.cpu_affinity import parse_cpu_set
            if not os.sched_getaffinity(pid) <= parse_cpu_set(self.affinity):
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
    for cpu in cpus:
        for memory in memories:
            limits = ProcessLimits(cpu, memory, affinity)
            limits.close()
