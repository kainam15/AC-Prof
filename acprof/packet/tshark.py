"""Stream tshark output while draining diagnostics and bounding its lifetime."""

import os
import selectors
import subprocess
import time
from collections.abc import Generator, Sequence


def stream_tshark(command: Sequence[str], *, timeout: float) -> Generator[str, None, None]:
    # Inherit the parser's process group: the postprocessing supervisor must
    # still be able to stop tshark if the parser itself is killed.
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    deadline = time.monotonic() + timeout
    pending = b""
    diagnostics = bytearray()
    truncated = False
    try:
        with selectors.DefaultSelector() as selector:
            for pipe in (process.stdout, process.stderr):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                for key, _ in selector.select(remaining):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if key.fileobj is process.stderr:
                        diagnostics.extend(chunk)
                        if len(diagnostics) > 65536:
                            del diagnostics[:-65536]
                            truncated = True
                    else:
                        pending += chunk
                        lines = pending.split(b"\n")
                        pending = lines.pop()
                        for line in lines:
                            yield line.decode("utf-8", errors="replace")
            if pending:
                yield pending.decode("utf-8", errors="replace")
            returncode = process.wait(timeout=max(0, deadline - time.monotonic()))
            if returncode:
                message = diagnostics.decode("utf-8", errors="replace").strip()
                if truncated:
                    message = "[earlier tshark stderr truncated]\n" + message
                raise subprocess.CalledProcessError(returncode, command, stderr=message)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stdout.close()
        process.stderr.close()
