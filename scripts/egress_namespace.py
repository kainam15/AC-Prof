"""Privileged helper for the isolated egress experiment, never the host namespace.

Invoked by validate_measurement_egress.py with a Docker PID and its start ticks.
The namespace FD is pinned and process identity checked before entering it.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

RULES = '''table inet acprof_review_egress {
    chain output {
        type filter hook output priority -10; policy drop;
        oifname "lo" ip daddr 127.0.0.1 accept
        oifname "lo" ip6 daddr ::1 accept
        ct state established,related ct direction reply accept
        counter reject with icmpx type admin-prohibited
    }
}
'''


def process_start_ticks(pid: int) -> int:
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[19])


def enter_namespace(pid: int, start_ticks: int) -> None:
    if pid <= 1 or start_ticks <= 0:
        raise ValueError('a live experiment container PID and start ticks are required')
    if process_start_ticks(pid) != start_ticks:
        raise RuntimeError('container process identity changed')
    descriptor = os.open(f'/proc/{pid}/ns/net', os.O_RDONLY | os.O_CLOEXEC)
    try:
        target = os.fstat(descriptor)
        current = os.stat('/proc/self/ns/net')
        if (target.st_dev, target.st_ino) == (current.st_dev, current.st_ino):
            raise RuntimeError('refusing to alter the host network namespace')
        if process_start_ticks(pid) != start_ticks:
            raise RuntimeError('container process identity changed while opening namespace')
        os.setns(descriptor, 0)
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pid', type=int)
    parser.add_argument('start_ticks', type=int)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    enter_namespace(args.pid, args.start_ticks)
    if not args.inspect:
        subprocess.run(['nft', '-f', '-'], input=RULES, text=True, check=True, timeout=10)
    subprocess.run(['nft', '-j', 'list', 'table', 'inet', 'acprof_review_egress'], check=True, timeout=10)


if __name__ == '__main__':
    main()
