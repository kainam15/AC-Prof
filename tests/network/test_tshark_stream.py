import subprocess
import sys
import time
from pathlib import Path

import pytest

from acprof.packet.tshark import stream_tshark


def child_script(tmp_path, body):
    path = tmp_path / 'child.py'
    path.write_text('import os, sys, time, signal\n' + body)
    return [sys.executable, str(path)]


def test_drains_large_stderr_and_yields_stdout_before_child_exits(tmp_path):
    release = tmp_path / 'release'
    command = child_script(tmp_path, f'''
os.write(2, b'x' * 200000)
os.write(1, b'first\\n')
while not os.path.exists({str(release)!r}):
    time.sleep(0.01)
os.write(1, b'last without newline')
''')
    lines = stream_tshark(command, timeout=5)
    assert next(lines) == 'first'
    release.touch()
    assert list(lines) == ['last without newline']


def test_nonzero_exit_retains_bounded_diagnostics(tmp_path):
    command = child_script(tmp_path, "os.write(2, b'x' * 200000 + b'failure detail')\nsys.exit(7)\n")
    with pytest.raises(subprocess.CalledProcessError) as caught:
        list(stream_tshark(command, timeout=5))
    assert caught.value.returncode == 7
    assert caught.value.stderr.endswith('failure detail')
    assert 'truncated' in caught.value.stderr
    assert len(caught.value.stderr) < 66000


@pytest.mark.parametrize('stop', ['timeout', 'consumer-close'])
def test_stops_and_reaps_child_ignoring_sigterm(tmp_path, stop):
    pid_path = tmp_path / 'pid'
    command = child_script(tmp_path, f'''
signal.signal(signal.SIGTERM, signal.SIG_IGN)
open({str(pid_path)!r}, 'w').write(str(os.getpid()))
print('ready', flush=True)
time.sleep(30)
''')
    lines = stream_tshark(command, timeout=0.4)
    assert next(lines) == 'ready'
    started = time.monotonic()
    if stop == 'timeout':
        with pytest.raises(subprocess.TimeoutExpired):
            list(lines)
    else:
        lines.close()
    assert time.monotonic() - started < 2
    assert not Path(f'/proc/{int(pid_path.read_text())}').exists()


@pytest.mark.parametrize("ending", [b"", b"\n"])
def test_fragmented_stdout_preserves_line_boundaries_without_recopying_prefixes(tmp_path, monkeypatch, ending):
    import os

    # The first byte of 界 is the last byte of a 64-byte read.
    prefix = "x" * (8192 - 2)
    payload = b"\n" + prefix.encode() + "界".encode() + b"\r\n\n\xfftail" + ending
    path = tmp_path / "frames"
    path.write_bytes(payload)
    command = child_script(tmp_path, f"with open({str(path)!r}, 'rb') as f: sys.stdout.buffer.write(f.read())")
    copied_prefix_bytes = 0
    read = os.read

    class Chunk(bytes):
        def __radd__(self, prefix):
            nonlocal copied_prefix_bytes
            copied_prefix_bytes += len(prefix)
            return prefix + bytes(self)

    def fragment(fd, size):
        return Chunk(read(fd, min(size, 64)))

    monkeypatch.setattr(os, "read", fragment)
    assert list(stream_tshark(command, timeout=5)) == ["", prefix + "界\r", "", "\ufffdtail"]
    assert copied_prefix_bytes <= len(payload)
