"""Apply verified process controls before importing any model dependency."""
import os
import sys
from pathlib import Path


def main():
    cpuset, cgroup, *command = sys.argv[1:]
    if cpuset:
        cpus = {int(cpu) for cpu in cpuset.split(",")}
        os.sched_setaffinity(0, cpus)
        if os.sched_getaffinity(0) != cpus:
            raise RuntimeError("requested affinity did not take effect")
    if cgroup:
        (Path(cgroup) / "cgroup.procs").write_text(str(os.getpid()))
        members = (Path(cgroup) / "cgroup.procs").read_text().split()
        if str(os.getpid()) not in members:
            raise RuntimeError("workload did not enter its owned cgroup")
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
