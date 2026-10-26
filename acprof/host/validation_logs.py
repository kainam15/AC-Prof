"""File-backed validation output and bounded parsing outside measurement windows."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import IO, TextIO

from acprof.artifacts import atomic_write

_READ_CHARS = 65536


def iter_prefixed_records(stream: IO[str], prefix: str) -> Iterator[str]:
    """Yield whole records while discarding arbitrarily long diagnostic lines."""
    current: list[str] | None = None
    line_start = True
    while fragment := stream.readline(max(_READ_CHARS, len(prefix))):
        if line_start:
            current = [fragment[len(prefix):]] if fragment.startswith(prefix) else None
        elif current is not None:
            current.append(fragment)
        line_start = fragment.endswith("\n")
        if line_start and current is not None:
            yield "".join(current)
            current = None
    if current is not None:
        yield "".join(current)


def last_prefixed_record(stream: IO[str], prefix: str) -> str | None:
    last = None
    for last in iter_prefixed_records(stream, prefix):
        pass
    return last


class ValidationLog:
    """Capture separate channels, then publish stdout + newline + stderr atomically.

    Owners close this object in their cleanup finally block, including on cancel.
    Files live alongside the destination so large output does not fill /tmp.
    """

    def __init__(self, path: Path):
        self.path = path
        self.dispatched = False
        self._resources = ExitStack()
        self._streams: tuple[IO[str], IO[str]] | None = None
        self._closed = False

    def run(self, command: Sequence[str], runner: Callable[..., subprocess.CompletedProcess[str]],
            *, timeout: float) -> subprocess.CompletedProcess[str]:
        if self._closed or self._streams is not None:
            raise RuntimeError("validation output can capture only one command")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        streams = [self._resources.enter_context(tempfile.TemporaryFile(
            mode="w+", encoding="utf-8", errors="replace", dir=self.path.parent)) for _ in range(2)]
        self._streams = (streams[0], streams[1])
        self.dispatched = True
        return runner(command, capture_output=False, text=True, timeout=timeout,
                      stdout=streams[0], stderr=streams[1])

    def records(self, prefix: str, *, include_stderr: bool = False) -> Iterator[str]:
        if self._streams is None:
            return
        for stream in self._streams if include_stderr else self._streams[:1]:
            stream.seek(0)
            yield from iter_prefixed_records(stream, prefix)

    def last_record(self, prefix: str) -> str | None:
        if self._streams is None:
            return None
        self._streams[0].seek(0)
        return last_prefixed_record(self._streams[0], prefix)

    def tail(self, limit: int = 4000) -> str:
        if self._streams is None:
            return ""
        tails = []
        for stream in self._streams:
            stream.flush()
            end = os.fstat(stream.fileno()).st_size
            size = min(end, limit * 4)
            raw = os.pread(stream.fileno(), size, end - size)
            tails.append(raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n"))
        return "\n".join(tails)[-limit:]

    def close(self) -> None:
        if self._closed:
            return

        def write(stream: TextIO) -> None:
            if self._streams is not None:
                for index, source in enumerate(self._streams):
                    if index:
                        stream.write("\n")
                    source.seek(0)
                    shutil.copyfileobj(source, stream, length=_READ_CHARS)

        try:
            atomic_write(self.path, write)
        finally:
            self._resources.close()
            self._closed = True
