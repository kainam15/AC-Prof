"""Atomic artifact publication preserves competing files and exposes I/O failures."""
from __future__ import annotations

import errno
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from threading import Barrier
from typing import TextIO
from unittest.mock import patch

import pytest

from acprof.artifacts import atomic_write


def create_competing_entry(destination: Path, kind: str) -> None:
    if kind == "file":
        destination.write_bytes(b"previous output\n")
        destination.chmod(0o600)
    elif kind == "directory":
        destination.mkdir()
    else:
        destination.symlink_to("missing-target")


@pytest.mark.parametrize("overwrite", [False, True])
def test_atomic_write_publishes_complete_file_with_requested_mode(tmp_path, overwrite):
    destination = tmp_path / "result.csv"
    opened = []

    def write(stream):
        opened.append(stream)
        stream.write('诊断,"quoted"\r\n')
        assert not destination.exists()

    atomic_write(destination, write, default_mode=0o640, overwrite=overwrite)
    assert destination.read_bytes() == '诊断,"quoted"\r\n'.encode()
    assert destination.stat().st_mode & 0o777 == 0o640
    assert opened[0].closed
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_write_default_replacement_preserves_existing_mode(tmp_path):
    destination = tmp_path / "result.csv"
    create_competing_entry(destination, "file")

    def write(stream: TextIO) -> None:
        stream.write("replacement\n")

    atomic_write(destination, write, default_mode=0o644)
    assert destination.read_bytes() == b"replacement\n"
    assert destination.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
@pytest.mark.parametrize("timing", ["before", "during"])
def test_exclusive_write_preserves_existing_directory_entries(tmp_path, kind, timing):
    destination = tmp_path / "result.csv"
    if timing == "before":
        create_competing_entry(destination, kind)

    def write(stream):
        stream.write("new output\n")
        if timing == "during":
            create_competing_entry(destination, kind)

    with pytest.raises(FileExistsError):
        atomic_write(destination, write, overwrite=False)
    if kind == "file":
        assert destination.read_bytes() == b"previous output\n"
        assert destination.stat().st_mode & 0o777 == 0o600
    elif kind == "directory":
        assert destination.is_dir()
        assert not list(destination.iterdir())
    else:
        assert destination.is_symlink()
        assert destination.readlink() == Path("missing-target")
    assert list(tmp_path.iterdir()) == [destination]


def test_exclusive_writers_publish_only_one_complete_output(tmp_path):
    destination = tmp_path / "result.csv"
    barrier = Barrier(2)
    payloads = [f"writer {index}: 诊断\n" * 2048 for index in range(2)]

    def attempt(payload):
        def write(stream):
            stream.write(payload)
            barrier.wait(timeout=10)
        try:
            atomic_write(destination, write, overwrite=False)
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, payloads))
    assert results.count(True) == 1
    assert destination.read_bytes() == payloads[results.index(True)].encode()
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("phase", ["write", "cancel", "flush", "mode", "file_sync", "publish"])
def test_failure_before_publication_closes_and_removes_temporary(tmp_path, monkeypatch, overwrite, phase):
    destination = tmp_path / "result.csv"
    if overwrite:
        create_competing_entry(destination, "file")
    failure = KeyboardInterrupt("cancelled export") if phase == "cancel" else OSError(errno.EIO, phase)
    opened = []

    def fail(*_args, **_kwargs):
        raise failure

    if phase in {"mode", "file_sync", "publish"}:
        operation = {"mode": "fchmod", "file_sync": "fsync",
                     "publish": "replace" if overwrite else "link"}[phase]
        monkeypatch.setattr(os, operation, fail)

    with ExitStack() as patches:
        def write(stream: TextIO) -> None:
            opened.append(stream)
            stream.write("partial output\n")
            if phase in {"write", "cancel"}:
                raise failure
            if phase == "flush":
                patches.enter_context(patch.object(stream, "flush", side_effect=failure))

        with pytest.raises(type(failure)) as caught:
            atomic_write(destination, write, overwrite=overwrite)
    assert caught.value is failure
    assert opened[0].closed
    if overwrite:
        assert destination.read_bytes() == b"previous output\n"
        assert destination.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == ([destination] if overwrite else [])


def test_exclusive_publication_syncs_complete_file_before_link_and_directory_after(tmp_path, monkeypatch):
    destination = tmp_path / "result.csv"
    events = []
    real_fsync, real_link = os.fsync, os.link

    def sync(fd):
        events.append("directory_sync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_sync")
        real_fsync(fd)

    def publish(source, target):
        assert Path(source).read_bytes() == b"complete output\n"
        assert not destination.exists()
        events.append("publish")
        real_link(source, target)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "link", publish)

    def write(stream: TextIO) -> None:
        stream.write("complete output\n")

    atomic_write(destination, write, overwrite=False)
    assert events == ["file_sync", "publish", "directory_sync"]
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("phase", ["unlink", "directory_sync"])
def test_error_after_exclusive_publication_keeps_complete_output(tmp_path, monkeypatch, phase):
    destination = tmp_path / "result.csv"
    failure = OSError(errno.EIO, phase)
    real_unlink = Path.unlink
    unlinks = []

    def unlink(path, **kwargs):
        unlinks.append(path)
        if len(unlinks) == 1:
            raise failure
        real_unlink(path, **kwargs)

    def fail_sync(_directory):
        raise failure

    if phase == "unlink":
        monkeypatch.setattr(Path, "unlink", unlink)
    else:
        monkeypatch.setattr("acprof.artifacts.sync_directory", fail_sync)

    def write(stream: TextIO) -> None:
        stream.write("complete output\n")

    with pytest.raises(OSError, match=phase) as caught:
        atomic_write(destination, write, overwrite=False)
    assert caught.value is failure
    assert destination.read_bytes() == b"complete output\n"
    assert list(tmp_path.iterdir()) == [destination]
