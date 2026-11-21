"""Stream tshark output while draining diagnostics and bounding its lifetime."""

import os
import selectors
import subprocess
import time
from collections.abc import Generator, Sequence


def tshark_fields_command(pcap: str, server_port: int) -> list[str]:
    """Use the same decoder, field list and TCP port filter for preflight and postprocessing."""
    port = str(server_port)

    command = [
        "tshark", "-l", "-r", pcap,
        "-o", "tcp.desegment_tcp_streams:TRUE",
        "-o", "http.desegment_body:TRUE",
        "-d", f"tcp.port=={port},http",
        "-Y", f"tcp.port=={port}",
        "-T", "fields", "-E", "separator=\t",
    ]
    for field in (
        "tcp.stream", "tcp.srcport", "tcp.dstport", "frame.len", "tcp.len",
        "frame.number", "frame.time_epoch", "http.request", "http.request.uri",
        "http.request.line", "http.response", "http.request_in", "http.response.code",
    ):
        command.extend(("-e", field))
    return command


def stream_tshark(command: Sequence[str], *, timeout: float) -> Generator[str, None, None]:
    # Inherit the parser's process group: the postprocessing supervisor must
    # still be able to stop tshark if the parser itself is killed.
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    deadline = time.monotonic() + timeout
    fragments: list[bytes] = []
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
                        lines = chunk.split(b"\n")
                        tail = lines.pop()
                        if lines and fragments:
                            fragments.append(lines[0])
                            lines[0] = b"".join(fragments)
                            fragments.clear()
                        for line in lines:
                            yield line.decode("utf-8", errors="replace")
                        if tail:
                            fragments.append(tail)
            if fragments:
                yield b"".join(fragments).decode("utf-8", errors="replace")
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
