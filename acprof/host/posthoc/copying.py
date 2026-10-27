"""Copy complete backup bytes while preserving filesystem-reported holes."""
from __future__ import annotations

import errno
import os
import shutil
import stat
from collections.abc import Iterator
from itertools import chain
from pathlib import Path

_BUFFER_BYTES = 1024 * 1024
_UNSUPPORTED_SEEK = {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS}


def _data_ranges(descriptor: int, size: int) -> Iterator[tuple[int, int]]:
    position = 0
    while position < size:
        try:
            start = os.lseek(descriptor, position, os.SEEK_DATA)
        except OSError as error:
            if error.errno == errno.ENXIO:
                return
            raise
        end = os.lseek(descriptor, start, os.SEEK_HOLE)
        if not position <= start < end <= size:
            raise OSError(errno.EIO, "invalid or changed backup source extent")
        yield start, end
        position = end


def _copy_range(source: int, destination: int, start: int, end: int) -> None:
    os.lseek(source, start, os.SEEK_SET)
    os.lseek(destination, start, os.SEEK_SET)
    remaining = end - start
    while remaining:
        block = os.read(source, min(_BUFFER_BYTES, remaining))
        if not block:
            raise OSError(errno.EIO, "backup source ended inside a data extent")
        pending = memoryview(block)
        while pending:
            written = os.write(destination, pending)
            if written <= 0:
                raise OSError(errno.EIO, "backup copy made no write progress")
            pending = pending[written:]
        remaining -= len(block)


def _copy_sparse(source: Path, destination: Path) -> bool:
    with source.open("rb", buffering=0) as reader:
        before = os.fstat(reader.fileno())
        extents = _data_ranges(reader.fileno(), before.st_size)
        try:
            first = next(extents, None)
        except OSError as error:
            # Capability fallback is allowed only before touching the destination.
            if error.errno in _UNSUPPORTED_SEEK:
                return False
            raise
        if first == (0, before.st_size):
            return False  # Compressed files can look sparse without reporting holes.
        with destination.open("wb", buffering=0) as writer:
            if first is not None:
                for start, end in chain((first,), extents):
                    _copy_range(reader.fileno(), writer.fileno(), start, end)
            os.ftruncate(writer.fileno(), before.st_size)
        after = os.fstat(reader.fileno())
        if ((before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise OSError(errno.EIO, "backup source changed during copy")
    shutil.copystat(source, destination)
    return True


def copy_backup_file(source: Path, destination: Path) -> None:
    """Preserve all bytes and copy2 metadata; callers own fsync and publication."""
    info = source.stat()
    sparse = (stat.S_ISREG(info.st_mode) and hasattr(os, "SEEK_DATA") and hasattr(os, "SEEK_HOLE")
              and getattr(info, "st_blocks", info.st_size) * 512 < info.st_size)
    if sparse:
        try:
            same_file = source.samefile(destination)
        except FileNotFoundError:
            same_file = False
        if same_file:
            raise shutil.SameFileError(f"{source!s} and {destination!s} are the same file")
        if _copy_sparse(source, destination):
            return
    shutil.copy2(source, destination)
